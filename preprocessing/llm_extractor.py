"""
llm_extractor.py  –  Task 1.2: LLM-Based Information Extraction
================================================================

WHY THIS FILE EXISTS
--------------------
The plan (Section 1.2) requires the system to:
  • Design a structured prompt to extract: personal_info, education[],
    experience[], skills[], publications[], supervision[], books[], patents[]
  • Use JSON-structured output from the LLM
  • Validate extracted JSON; flag missing or null fields
  • For each institution, look up THE and QS world university rankings

DESIGN DECISIONS
----------------
- The extraction prompt is designed precisely following the plan's exact
  field list and the spec's Section 3 requirements.
- We use a two-mode approach:
    * If the CV text is ≤ 24,000 chars (~6000 tokens), one API call extracts everything.
    * If longer, we chunk the text and merge results (plan tip: "run extraction
      per-section if the CV is long > 4000 tokens").
- Groq models (llama-3.3-70b-versatile) have a 128K token context window,
  so we use a much larger chunk size than the original 14K.
- We do NOT use response_format=json_object because not all Groq-hosted models
  support that parameter. Instead, the prompt instructs the model to reply with
  only JSON, and we parse/repair the response robustly.
- THE and QS rankings are looked up via Groq as a second LLM call per institution,
  since live web scraping would require credentials. The LLM uses its training
  knowledge and is explicitly instructed to return null if uncertain.
- Every field that the LLM cannot find is set to null, not omitted.
- A ValidationReport is produced per candidate listing every null/missing field.

IMPORTANT: This module requires a Groq (or OpenAI-compatible) API key set in the
environment variable OPENAI_API_KEY plus OPENAI_BASE_URL for Groq.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Groq llama-3.3-70b-versatile context window is 128K tokens (~512K chars).
# We use 24K chars (~6K tokens) per chunk so there is ample room for the
# system prompt and the response. For most single CVs this means one call.
_MAX_CHARS_PER_CALL = 24_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Tracks which fields were null/missing after extraction."""
    candidate_filename: str
    missing_top_level: List[str] = field(default_factory=list)   # e.g. "education"
    missing_personal_fields: List[str] = field(default_factory=list)
    empty_arrays: List[str] = field(default_factory=list)         # arrays with 0 items
    notes: List[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(
            self.missing_top_level
            or self.missing_personal_fields
            or self.empty_arrays
        )

    def summary(self) -> str:
        parts = []
        if self.missing_top_level:
            parts.append(f"Missing keys: {self.missing_top_level}")
        if self.missing_personal_fields:
            parts.append(f"Missing personal fields: {self.missing_personal_fields}")
        if self.empty_arrays:
            parts.append(f"Empty arrays: {self.empty_arrays}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return " | ".join(parts) if parts else "OK"


@dataclass
class ExtractionResult:
    """Contains the extracted structured data for one candidate."""
    candidate_filename: str
    success: bool
    data: Optional[Dict[str, Any]]    # The full extracted JSON dict
    validation: Optional[ValidationReport]
    raw_response: Optional[str] = None  # Raw LLM response text (for debugging)
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Main extraction prompt.
# Follows the plan exactly: all 8 top-level keys, all sub-fields, JSON-only response.
# Also instructs the model to extract institution names for THE/QS lookup.
_SYSTEM_PROMPT = """You are a highly accurate CV data extractor for a university recruitment system called TALASH.
Your task is to read CV text and return a structured JSON object.

STRICT RULES:
1. Return ONLY a valid JSON object. No markdown fences, no explanation, no extra text before or after.
2. Use EXACTLY the keys shown in the schema. Do not add extra keys.
3. If a field is not found in the CV text, set its value to null.
4. For array fields, return an empty array [] if no items are found.
5. For dates, use YYYY-MM format when both year and month are available, or YYYY if only year is known.
6. Do NOT invent or guess data that is not explicitly stated in the CV.

JSON SCHEMA:
{
  "personal_info": {
    "name": null,
    "email": null,
    "phone": null,
    "address": null,
    "cnic": null
  },
  "education": [
    {
      "level": null,
      "degree": null,
      "specialization": null,
      "institution": null,
      "country": null,
      "start_year": null,
      "end_year": null,
      "marks_percentage": null,
      "cgpa": null,
      "cgpa_scale": null,
      "board": null
    }
  ],
  "experience": [
    {
      "job_title": null,
      "organization": null,
      "start_date": null,
      "end_date": null,
      "employment_type": null,
      "description": null
    }
  ],
  "skills": [
    {
      "skill_name": null,
      "category": null
    }
  ],
  "publications": [
    {
      "type": null,
      "title": null,
      "venue": null,
      "year": null,
      "authors": null,
      "doi": null,
      "url": null,
      "issn": null
    }
  ],
  "supervision": [
    {
      "student_name": null,
      "level": null,
      "role": null,
      "year": null,
      "thesis_title": null
    }
  ],
  "books": [
    {
      "title": null,
      "authors": null,
      "isbn": null,
      "publisher": null,
      "year": null,
      "link": null
    }
  ],
  "patents": [
    {
      "patent_number": null,
      "title": null,
      "date": null,
      "inventors": null,
      "country": null,
      "link": null
    }
  ]
}

FIELD CONSTRAINTS:
- education.level: one of [SSC, HSSC, BS, BSc, MS, MPhil, PhD, Other]
- education.country: the country where the institution is located (e.g., Pakistan, USA, UK)
- experience.employment_type: one of [Full-time, Part-time, Contract, Visiting, Research, Internship, Other]
- publications.type: one of [journal, conference, book-chapter, other]
- publications.issn: extract ISSN if mentioned, otherwise null
- supervision.level: one of [MS, MPhil, PhD, BS, Other]
- supervision.role: one of [main, co-supervisor, external]
- skills.category: one of [technical, soft, language, tool, domain, other]
"""

_USER_PROMPT_TEMPLATE = """Extract all information from the following CV text and return the JSON object as specified.
Important: Output ONLY the JSON object. Start your response with {{ and end with }}.

CV TEXT:
{cv_text}
"""

# Prompt for THE/QS ranking lookup via LLM
_RANKING_SYSTEM_PROMPT = """You are an expert on global university rankings with deep knowledge of
the Times Higher Education (THE) World University Rankings and QS World University Rankings.

Your task: given a list of institution names, return a JSON object mapping each institution name
to its known ranking information.

STRICT RULES:
1. Return ONLY a valid JSON object. No markdown, no explanation.
2. Only use information from your training knowledge. Do NOT guess or hallucinate rankings.
3. If you are not confident about the exact ranking, set the value to null.
4. Rankings change yearly — return the most recent ranking you know from your training data.
5. Use the institution name exactly as provided as the JSON key.

JSON SCHEMA for each institution:
{
  "INSTITUTION_NAME": {
    "the_rank": null,
    "the_rank_range": null,
    "qs_rank": null,
    "qs_rank_range": null,
    "known_as": null,
    "ranking_notes": null
  }
}

- the_rank: integer or null (e.g., 201 means rank 201)
- the_rank_range: string for banded ranks (e.g., "601-800", "1001+") or null
- qs_rank: integer or null
- qs_rank_range: string for banded ranks (e.g., "601-650") or null
- known_as: common short name if the institution is better known by another name (e.g., "MIT")
- ranking_notes: any important note (e.g., "Pakistani national ranking only", "not in THE/QS")

IMPORTANT: For Pakistani universities like NUST, COMSATS, UET, LUMS, GCU, etc., include their
known THE/QS rankings. Many Pakistani universities appear in the 801+ or 1001+ band.
"""

_RANKING_USER_PROMPT = """Look up THE and QS World University Rankings for these institutions:

{institutions}

Return a single JSON object with one key per institution name."""

# Prompt for chunked merge
_MERGE_PROMPT_TEMPLATE = """You are merging multiple partial JSON extractions of the same CV.
Combine them into one final JSON object following the same schema.
RULES:
1. Return ONLY valid JSON. No markdown, no explanation.
2. Start with {{ and end with }}.
3. Prefer non-null values over null values when there is a conflict.
4. Merge arrays by combining all items (deduplicate by title/name where obvious).

PARTIAL EXTRACTIONS:
{partial_jsons}
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    Uses an LLM (Groq/OpenAI-compatible) to extract structured data from CV text.
    Also performs THE/QS university ranking enrichment via a second LLM call.

    Usage
    -----
    extractor = LLMExtractor()
    result = extractor.extract(candidate_filename="john_doe", cv_text="...")
    if result.success:
        print(result.data["personal_info"])
        print(result.data["education"][0]["the_rank"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        base_url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            API key. Falls back to OPENAI_API_KEY environment variable.
        model : str
            Model name (e.g. llama-3.3-70b-versatile for Groq).
        temperature : float
            0.0 means deterministic output (best for structured extraction).
        base_url : str, optional
            Custom base URL for alternative providers (like Groq or xAI).
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", None)

        if not self.api_key:
            logger.warning(
                "No API key found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to LLMExtractor()."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, candidate_filename: str, cv_text: str) -> ExtractionResult:
        """
        Extract structured information from CV text, then enrich education
        entries with THE/QS university rankings.

        Parameters
        ----------
        candidate_filename : str
            The PDF filename (stem), used as a reference identifier.
        cv_text : str
            Raw text extracted from the PDF.

        Returns
        -------
        ExtractionResult
        """
        if not cv_text or not cv_text.strip():
            return ExtractionResult(
                candidate_filename=candidate_filename,
                success=False,
                data=None,
                validation=None,
                error_message="Empty CV text provided. Cannot extract data.",
            )

        try:
            # ── Step 1: Extract structured data from CV text ──────────────
            if len(cv_text) <= _MAX_CHARS_PER_CALL:
                raw_json = self._extract_single(cv_text)
            else:
                logger.info(
                    "'%s' is long (%d chars). Using chunked extraction.",
                    candidate_filename, len(cv_text)
                )
                raw_json = self._extract_chunked(cv_text)

            data = self._parse_json(raw_json, candidate_filename)
            if data is None:
                return ExtractionResult(
                    candidate_filename=candidate_filename,
                    success=False,
                    data=None,
                    validation=None,
                    raw_response=raw_json,
                    error_message="LLM returned invalid JSON that could not be repaired.",
                )

            # Ensure all top-level keys exist (fill with empty/null defaults)
            data = self._fill_missing_keys(data)

            # ── Step 2: Enrich education with THE/QS rankings ─────────────
            data = self._enrich_with_rankings(data, candidate_filename)

            validation = self._validate(candidate_filename, data)
            if validation.has_issues():
                logger.info(
                    "Validation issues for '%s': %s",
                    candidate_filename, validation.summary()
                )

            return ExtractionResult(
                candidate_filename=candidate_filename,
                success=True,
                data=data,
                validation=validation,
                raw_response=raw_json,
            )

        except Exception as e:
            logger.exception("LLM extraction failed for '%s'.", candidate_filename)
            return ExtractionResult(
                candidate_filename=candidate_filename,
                success=False,
                data=None,
                validation=None,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # Internal: API calls
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        """Make one chat completion call and return the response text."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            # NOTE: We intentionally do NOT pass response_format={"type": "json_object"}
            # because Groq-hosted open-source models (llama, mixtral) do not all support
            # that parameter. Instead, the prompt explicitly instructs JSON-only output.
        )
        return response.choices[0].message.content or ""

    def _extract_single(self, cv_text: str) -> str:
        """Extract data from the entire CV in one API call."""
        user_prompt = _USER_PROMPT_TEMPLATE.format(cv_text=cv_text)
        return self._call_llm(_SYSTEM_PROMPT, user_prompt)

    def _extract_chunked(self, cv_text: str) -> str:
        """
        Split the CV into overlapping chunks, extract from each, then merge.
        Used when the CV text exceeds _MAX_CHARS_PER_CALL.
        """
        chunks = self._split_into_chunks(cv_text, max_chars=_MAX_CHARS_PER_CALL, overlap=500)
        logger.debug("Chunked into %d parts.", len(chunks))

        partial_jsons = []
        for i, chunk in enumerate(chunks):
            logger.debug("Extracting chunk %d/%d …", i + 1, len(chunks))
            user_prompt = _USER_PROMPT_TEMPLATE.format(cv_text=chunk)
            raw = self._call_llm(_SYSTEM_PROMPT, user_prompt)
            partial_jsons.append(raw)

        if len(partial_jsons) == 1:
            return partial_jsons[0]

        # Merge all partial JSONs into one final result
        merge_user = _MERGE_PROMPT_TEMPLATE.format(
            partial_jsons="\n---\n".join(partial_jsons)
        )
        return self._call_llm(_SYSTEM_PROMPT, merge_user)

    # ------------------------------------------------------------------
    # Internal: THE/QS ranking enrichment
    # ------------------------------------------------------------------

    def _enrich_with_rankings(self, data: Dict, candidate_filename: str) -> Dict:
        """
        For each unique institution in the education list, look up its
        THE and QS world university rankings via the LLM.

        Adds these fields to each education entry:
          the_rank, the_rank_range, qs_rank, qs_rank_range, ranking_notes
        """
        education = data.get("education") or []
        if not education:
            return data

        # Collect unique non-null institution names
        institutions = list({
            edu["institution"]
            for edu in education
            if edu.get("institution")
        })

        if not institutions:
            return data

        logger.info(
            "  [1.2b] Looking up THE/QS rankings for %d institution(s): %s",
            len(institutions), ", ".join(institutions[:5])
        )

        try:
            institution_list = "\n".join(f"- {inst}" for inst in institutions)
            user_prompt = _RANKING_USER_PROMPT.format(institutions=institution_list)
            raw = self._call_llm(_RANKING_SYSTEM_PROMPT, user_prompt)
            rankings = self._parse_json(raw, f"{candidate_filename}_rankings")
        except Exception as e:
            logger.warning("Ranking lookup failed for '%s': %s", candidate_filename, e)
            rankings = {}

        if not rankings:
            rankings = {}

        # Attach ranking data to each education entry
        for edu in education:
            inst = edu.get("institution")
            if inst and inst in rankings:
                r = rankings[inst] or {}
                edu["the_rank"] = r.get("the_rank")
                edu["the_rank_range"] = r.get("the_rank_range")
                edu["qs_rank"] = r.get("qs_rank")
                edu["qs_rank_range"] = r.get("qs_rank_range")
                edu["ranking_notes"] = r.get("ranking_notes")
            else:
                edu.setdefault("the_rank", None)
                edu.setdefault("the_rank_range", None)
                edu.setdefault("qs_rank", None)
                edu.setdefault("qs_rank_range", None)
                edu.setdefault("ranking_notes", None)

        data["education"] = education
        return data

    # ------------------------------------------------------------------
    # Internal: JSON parsing & repair
    # ------------------------------------------------------------------

    def _parse_json(self, raw: str, name: str) -> Optional[Dict]:
        """Parse JSON from LLM response; attempt simple repair on failure."""
        if not raw:
            return None

        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Attempt to extract JSON from within markdown fences (sometimes LLM adds them)
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Final fallback: find first { ... } block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.error("Could not parse JSON for '%s'. Raw:\n%s", name, raw[:500])
        return None

    # ------------------------------------------------------------------
    # Internal: defaults & validation
    # ------------------------------------------------------------------

    _REQUIRED_KEYS = [
        "personal_info", "education", "experience", "skills",
        "publications", "supervision", "books", "patents"
    ]
    _PERSONAL_FIELDS = ["name", "email", "phone", "address", "cnic"]
    _ARRAY_KEYS = [
        "education", "experience", "skills",
        "publications", "supervision", "books", "patents"
    ]

    def _fill_missing_keys(self, data: Dict) -> Dict:
        """Ensure all required top-level keys are present."""
        for key in self._REQUIRED_KEYS:
            if key not in data:
                data[key] = [] if key in self._ARRAY_KEYS else None
        if "personal_info" not in data or data["personal_info"] is None:
            data["personal_info"] = {f: None for f in self._PERSONAL_FIELDS}
        else:
            for f in self._PERSONAL_FIELDS:
                if f not in data["personal_info"]:
                    data["personal_info"][f] = None
        return data

    def _validate(self, candidate_filename: str, data: Dict) -> ValidationReport:
        """Check for null or empty fields and produce a validation report."""
        report = ValidationReport(candidate_filename=candidate_filename)

        # Check top-level keys
        for key in self._REQUIRED_KEYS:
            if data.get(key) is None:
                report.missing_top_level.append(key)

        # Check personal_info sub-fields
        pinfo = data.get("personal_info") or {}
        for f in self._PERSONAL_FIELDS:
            if not pinfo.get(f):
                report.missing_personal_fields.append(f)

        # Check arrays that are empty
        for key in self._ARRAY_KEYS:
            val = data.get(key)
            if isinstance(val, list) and len(val) == 0:
                report.empty_arrays.append(key)

        return report

    # ------------------------------------------------------------------
    # Internal: text chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(text: str, max_chars: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks at paragraph boundaries where possible.
        Correctly terminates at end of text without infinite looping.
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # Try to break at a paragraph boundary (double newline)
            if end < len(text):
                boundary = text.rfind("\n\n", start, end)
                if boundary != -1 and boundary > start + max_chars // 2:
                    end = boundary + 2
            chunks.append(text[start:end])

            if end >= len(text):
                break

            start = end - overlap  # overlap to avoid losing context at boundaries
        return chunks
