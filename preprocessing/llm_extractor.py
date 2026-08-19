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
    flags: List[Dict[str, Any]] = field(default_factory=list)     # data quality flags
    notes: List[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(
            self.missing_top_level
            or self.missing_personal_fields
            or self.empty_arrays
            or self.flags
        )

    def summary(self) -> str:
        parts = []
        if self.missing_top_level:
            parts.append(f"Missing keys: {self.missing_top_level}")
        if self.missing_personal_fields:
            parts.append(f"Missing personal fields: {self.missing_personal_fields}")
        if self.empty_arrays:
            parts.append(f"Empty arrays: {self.empty_arrays}")
        if self.flags:
            flags_summary = [f"{f['entry_ref']}:{f['flag_type']} ({f['explanation']})" for f in self.flags]
            parts.append(f"Flags: [{'; '.join(flags_summary)}]")
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
        model: Optional[str] = None,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            API key. Falls back to OPENAI_API_KEY environment variable.
        model : str, optional
            Model name. Defaults to OPENAI_MODEL env var or groq/compound-mini.
        temperature : float
            0.0 means deterministic output (best for structured extraction).
        base_url : str, optional
            Custom base URL for alternative providers (like Groq or xAI).
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "groq/compound-mini")
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
            # Try single-call first; fall back to chunked if JSON is invalid
            if len(cv_text) <= _MAX_CHARS_PER_CALL:
                raw_json = self._extract_single(cv_text)
                data = self._parse_json(raw_json, candidate_filename)
                if data is None:
                    # Retry 1: force chunked even for short CVs (model may have truncated)
                    logger.warning(
                        "'%s' single-call returned invalid JSON — retrying with chunked extraction.",
                        candidate_filename
                    )
                    raw_json = self._extract_chunked(cv_text)
                    data = self._parse_json(raw_json, candidate_filename)
            else:
                logger.info(
                    "'%s' is long (%d chars). Using chunked extraction.",
                    candidate_filename, len(cv_text)
                )
                raw_json = self._extract_chunked(cv_text)
                data = self._parse_json(raw_json, candidate_filename)
                if data is None:
                    # Retry: smaller chunks
                    logger.warning(
                        "'%s' chunked extraction returned invalid JSON — retrying with smaller chunks.",
                        candidate_filename
                    )
                    raw_json = self._extract_chunked_small(cv_text)
                    data = self._parse_json(raw_json, candidate_filename)

            if data is None:
                logger.warning(
                    "'%s' LLM extraction returned empty/invalid JSON — falling back to rule-based heuristic extraction.",
                    candidate_filename
                )
                data = self._extract_heuristic(candidate_filename, cv_text)

            # Normalize flat/non-standard model output to required schema structure
            data = self._normalize_to_schema(data)

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
            logger.warning("LLM extraction exception for '%s': %s. Falling back to heuristic extraction.", candidate_filename, e)
            fallback_data = self._extract_heuristic(candidate_filename, cv_text)
            fallback_data = self._normalize_to_schema(fallback_data)
            fallback_data = self._fill_missing_keys(fallback_data)
            validation = self._validate(candidate_filename, fallback_data)
            return ExtractionResult(
                candidate_filename=candidate_filename,
                success=True,
                data=fallback_data,
                validation=validation,
                raw_response=None,
            )

    # ------------------------------------------------------------------
    # Internal: API calls
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        """Make one chat completion call with retry, backoff, and model fallback on 429 rate limits."""
        import time
        try:
            from openai import OpenAI, APIError, RateLimitError
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        models_to_try = [self.model]
        if "groq/compound-mini" in models_to_try and "qwen/qwen3.6-27b" not in models_to_try:
            models_to_try.append("qwen/qwen3.6-27b")

        for m in models_to_try:
            for attempt in range(4):
                try:
                    response = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=self.temperature,
                        max_tokens=4000,
                    )
                    text = response.choices[0].message.content or ""
                    if text.strip():
                        return text
                except RateLimitError as e:
                    sleep_time = 4.0 * (attempt + 1)
                    logger.warning("Rate limit on model '%s' (attempt %d/4). Waiting %.1fs...", m, attempt + 1, sleep_time)
                    time.sleep(sleep_time)
                except APIError as e:
                    logger.warning("API error on model '%s': %s. Retrying...", m, e)
                    time.sleep(2.0)
                except Exception as e:
                    logger.error("LLM call error on model '%s': %s", m, e)
                    break
        return ""

    def _extract_single(self, cv_text: str) -> str:
        """
        Two-phase extraction to avoid hitting max_tokens limits.
        Phase 1: personal_info, education, experience, skills (compact).
        Phase 2: publications, supervision, books, patents (potentially large).
        Results are merged into one JSON.
        """
        phase1_prompt = (
            "Extract ONLY the following sections from the CV and return a JSON object.\n"
            "Return ONLY valid JSON starting with { and ending with }.\n"
            "Sections needed: personal_info, education, experience, skills.\n\n"
            "Schema:\n"
            '{"personal_info":{"name":null,"email":null,"phone":null,"address":null,"cnic":null},'
            '"education":[{"level":null,"degree":null,"specialization":null,"institution":null,'
            '"country":null,"start_year":null,"end_year":null,"marks_percentage":null,"cgpa":null,"cgpa_scale":null,"board":null}],'
            '"experience":[{"job_title":null,"organization":null,"start_date":null,"end_date":null,"employment_type":null,"description":null}],'
            '"skills":[{"skill_name":null,"category":null}]}\n\n'
            f"CV TEXT:\n{cv_text}"
        )

        phase2_prompt = (
            "Extract ONLY the following sections from the CV and return a JSON object.\n"
            "Return ONLY valid JSON starting with { and ending with }.\n"
            "Sections needed: publications, supervision, books, patents.\n\n"
            "Schema:\n"
            '{"publications":[{"type":null,"title":null,"venue":null,"year":null,"authors":null,"doi":null,"url":null,"issn":null}],'
            '"supervision":[{"student_name":null,"level":null,"role":null,"year":null,"thesis_title":null}],'
            '"books":[{"title":null,"authors":null,"isbn":null,"publisher":null,"year":null,"link":null}],'
            '"patents":[{"patent_number":null,"title":null,"date":null,"inventors":null,"country":null,"link":null}]}\n\n'
            f"CV TEXT:\n{cv_text}"
        )

        raw1 = self._call_llm(
            "You are a precise CV data extractor. Return ONLY valid JSON. No markdown, no explanation.",
            phase1_prompt
        )
        raw2 = self._call_llm(
            "You are a precise CV data extractor. Return ONLY valid JSON. No markdown, no explanation.",
            phase2_prompt
        )

        # Merge both JSONs
        data1 = self._parse_json(raw1, "phase1") or {}
        data2 = self._parse_json(raw2, "phase2") or {}
        merged = {**data1, **data2}
        import json
        return json.dumps(merged)

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

    def _extract_chunked_small(self, cv_text: str) -> str:
        """
        Final fallback: uses very small 6000-char chunks for complex/long CVs.
        Extracts each chunk independently and merges results.
        """
        small_chunk = 6_000
        chunks = self._split_into_chunks(cv_text, max_chars=small_chunk, overlap=300)
        logger.debug("Small-chunked into %d parts (chunk_size=%d).", len(chunks), small_chunk)

        partial_jsons = []
        for i, chunk in enumerate(chunks):
            logger.debug("Small-chunk extracting %d/%d …", i + 1, len(chunks))
            user_prompt = _USER_PROMPT_TEMPLATE.format(cv_text=chunk)
            raw = self._call_llm(_SYSTEM_PROMPT, user_prompt)
            # Only keep parseable partial results
            if self._parse_json(raw, f"small_chunk_{i}") is not None:
                partial_jsons.append(raw)

        if not partial_jsons:
            return "{}"
        if len(partial_jsons) == 1:
            return partial_jsons[0]

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
        if not raw or not raw.strip():
            return None

        # Remove thinking blocks (<think>...</think>) from models like qwen
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

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

    def _extract_heuristic(self, candidate_filename: str, cv_text: str) -> Dict[str, Any]:
        """
        Rule-based / regex fallback extraction when LLM API calls are unavailable or rate-limited.
        Ensures CV extraction ALWAYS succeeds and produces a valid record.
        """
        # Extract candidate name
        name = None
        m = re.search(r"\bName\s+([A-Z][A-Z\s]{2,50}?)(?:\s{2,}|\s+Father|\s+Date|\n)", cv_text)
        if m:
            name = m.group(1).strip()
        if not name:
            m = re.search(r"Name[:\s]+([A-Z][a-zA-Z\s]{2,50}?)\n", cv_text)
            if m:
                name = m.group(1).strip()
        if not name:
            clean_fn = candidate_filename.replace(".pdf", "").replace("_CV", "")
            clean_fn = re.sub(r"^\d+_", "", clean_fn)
            name = clean_fn.replace("_", " ").title()

        # Email
        email = None
        em = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cv_text)
        if em:
            email = em.group(0)

        # Phone
        phone = None
        pm = re.search(r"(\+?\d{1,3}[\s-]?)?\(?\d{3,4}\)?[\s-]?\d{6,8}", cv_text)
        if pm:
            phone = pm.group(0)

        # Education
        education = []
        if re.search(r"\b(Ph\.?D|Doctor of Philosophy)\b", cv_text, re.I):
            education.append({"level": "PhD", "degree": "Doctor of Philosophy", "institution": None})
        if re.search(r"\b(M\.?S|M\.?Sc|Master|MPhil)\b", cv_text, re.I):
            education.append({"level": "MS", "degree": "Master of Science", "institution": None})
        if re.search(r"\b(B\.?S|B\.?Sc|Bachelor|BIT)\b", cv_text, re.I):
            education.append({"level": "BS", "degree": "Bachelor of Science", "institution": None})

        return {
            "personal_info": {
                "name": name,
                "email": email,
                "phone": phone,
                "address": None,
                "cnic": None
            },
            "education": education,
            "experience": [],
            "skills": [],
            "publications": [],
            "supervision": [],
            "books": [],
            "patents": []
        }

    def _normalize_to_schema(self, data: Dict) -> Dict:
        """
        Normalize model output to the required schema structure.
        Some models (e.g. groq/compound-mini) return a flat dict or
        non-nested structure instead of the required nested schema.
        This method remaps keys to ensure compliance.
        """
        # Already has the required top-level structure — return as-is
        has_schema_keys = any(k in data for k in ("personal_info", "education", "experience", "publications"))
        # But if personal_info is missing, the model may have placed fields at top level
        has_flat_keys = any(k in data for k in ("name", "email", "phone", "father_name", "date_of_birth"))

        if has_schema_keys and not has_flat_keys:
            return data  # Already correct structure

        # Build normalized output
        normalized: Dict[str, Any] = {}

        # ── personal_info ────────────────────────────────────────────────
        if "personal_info" in data and isinstance(data["personal_info"], dict):
            normalized["personal_info"] = data["personal_info"]
        else:
            # Try to extract personal fields from flat structure
            pinfo = data.get("personal_info") or {}
            if not pinfo or not isinstance(pinfo, dict):
                pinfo = {}
            # Supplement with flat-level fields if missing
            for flat_key, schema_key in [("name", "name"), ("email", "email"), ("phone", "phone"), ("address", "address"), ("cnic", "cnic")]:
                if schema_key not in pinfo or not pinfo[schema_key]:
                    # Try common flat variants
                    val = data.get(flat_key) or data.get(f"personal_{flat_key}")
                    if val:
                        pinfo[schema_key] = val
            normalized["personal_info"] = pinfo

        # ── education ────────────────────────────────────────────────────
        edu_raw = data.get("education") or []
        if isinstance(edu_raw, list):
            edu_list = []
            for item in edu_raw:
                if isinstance(item, dict):
                    edu_list.append(item)
                elif isinstance(item, str):
                    # Model returned plain strings — wrap as minimal dict
                    edu_list.append({"degree": item, "institution": None, "level": None,
                                     "specialization": None, "country": None,
                                     "start_year": None, "end_year": None,
                                     "marks_percentage": None, "cgpa": None, "cgpa_scale": None, "board": None})
            normalized["education"] = edu_list
        else:
            normalized["education"] = []

        # ── experience ───────────────────────────────────────────────────
        exp_raw = data.get("experience") or []
        if isinstance(exp_raw, list):
            exp_list = []
            for item in exp_raw:
                if isinstance(item, dict):
                    exp_list.append(item)
                elif isinstance(item, str):
                    exp_list.append({"job_title": item, "organization": None,
                                     "start_date": None, "end_date": None,
                                     "employment_type": None, "description": None})
            normalized["experience"] = exp_list
        else:
            normalized["experience"] = []

        # ── publications ─────────────────────────────────────────────────
        pub_raw = data.get("publications") or []
        if isinstance(pub_raw, list):
            pub_list = []
            for item in pub_raw:
                if isinstance(item, dict):
                    pub_list.append(item)
                elif isinstance(item, str):
                    pub_list.append({"title": item, "type": None, "venue": None,
                                     "year": None, "authors": None, "doi": None,
                                     "url": None, "issn": None})
            normalized["publications"] = pub_list
        else:
            normalized["publications"] = []

        # ── skills, supervision, books, patents — pass through ───────────
        for key in ("skills", "supervision", "books", "patents"):
            val = data.get(key)
            if isinstance(val, list):
                # Normalize string items to minimal dicts
                result = []
                for item in val:
                    if isinstance(item, dict):
                        result.append(item)
                    elif isinstance(item, str) and item.strip():
                        result.append({"title": item} if key in ("books",) else {"skill_name": item, "category": None} if key == "skills" else {"patent_number": item} if key == "patents" else {"student_name": item})
                normalized[key] = result
            else:
                normalized[key] = val if val is not None else []

        return normalized

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

        # Check individual education entries for data quality
        education = data.get("education") or []
        for idx, edu in enumerate(education):
            ref = f"education[{idx}]"
            deg_title = edu.get("degree") or "Unknown Degree"

            if not edu.get("institution"):
                report.flags.append({
                    "entry_ref": ref,
                    "flag_type": "MISSING_INSTITUTION",
                    "explanation": f"Institution name is missing for '{deg_title}'."
                })
            if not edu.get("degree"):
                report.flags.append({
                    "entry_ref": ref,
                    "flag_type": "MISSING_DEGREE_TITLE",
                    "explanation": f"Degree title is missing at index {idx}."
                })
            
            # Check years
            start_yr = edu.get("start_year")
            end_yr = edu.get("end_year")
            if not start_yr and not end_yr:
                report.flags.append({
                    "entry_ref": ref,
                    "flag_type": "MISSING_YEARS",
                    "explanation": f"Both start and end years are missing for '{deg_title}'."
                })
            elif start_yr and end_yr:
                try:
                    s_val = int(float(start_yr))
                    e_val = int(float(end_yr))
                    if s_val > e_val:
                        report.flags.append({
                            "entry_ref": ref,
                            "flag_type": "INVALID_YEAR_RANGE",
                            "explanation": f"Start year ({start_yr}) is after end year ({end_yr}) for '{deg_title}'."
                        })
                except (ValueError, TypeError):
                    pass

            # Check grades
            if not edu.get("cgpa") and not edu.get("marks_percentage"):
                report.flags.append({
                    "entry_ref": ref,
                    "flag_type": "MISSING_GRADES",
                    "explanation": f"Academic score (CGPA or marks percentage) is missing for '{deg_title}'."
                })

            # Check specialization
            if not edu.get("specialization"):
                report.flags.append({
                    "entry_ref": ref,
                    "flag_type": "MISSING_SPECIALIZATION",
                    "explanation": f"Specialization/discipline is missing for '{deg_title}'."
                })

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
