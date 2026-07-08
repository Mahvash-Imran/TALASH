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

DESIGN DECISIONS
----------------
- The extraction prompt is designed precisely following the plan's exact
  field list and the spec's Section 3 requirements.
- We use a two-mode approach:
    * If the CV text is ≤ 3500 tokens (~14,000 chars), one API call extracts everything.
    * If longer, we chunk the text and merge results (plan tip: "run extraction
      per-section if the CV is long > 4000 tokens").
- We request `response_format={"type": "json_object"}` (OpenAI JSON mode)
  so the model always returns valid JSON – no need for regex parsing.
- Every field that the LLM cannot find is set to `null`, not omitted.
- A ValidationReport is produced per candidate listing every null/missing field.

IMPORTANT: This module requires an OpenAI API key set in the environment
variable OPENAI_API_KEY (or passed directly).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Approximate characters per token (conservative estimate)
_CHARS_PER_TOKEN = 4
# Max characters to send in one call (3500 tokens × 4 chars/token)
_MAX_CHARS_PER_CALL = 14_000


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

# The main extraction prompt.
# Follows the plan exactly: all 8 top-level keys, all sub-fields, JSON-only response.
_SYSTEM_PROMPT = """You are a highly accurate CV data extractor for a university recruitment system called TALASH.
Your task is to read CV text and return a structured JSON object.

RULES:
1. Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
2. Use EXACTLY the keys shown below. Do not add extra keys.
3. If a field is not found in the CV text, set its value to null.
4. For array fields, return an empty array [] if no items are found.
5. For dates, use YYYY-MM format when both year and month are available, or YYYY if only year is known.

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
      "url": null
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

NOTES ON SPECIFIC FIELDS:
- education.level: one of [SSE, HSSC, BS, BSc, MS, MPhil, PhD, Other]
- experience.employment_type: one of [Full-time, Part-time, Contract, Visiting, Research, Internship, Other]
- publications.type: one of [journal, conference, book-chapter, other]
- supervision.level: one of [MS, MPhil, PhD, BS, Other]
- supervision.role: one of [main, co-supervisor, external]
- skills.category: one of [technical, soft, language, tool, domain, other]
"""

_USER_PROMPT_TEMPLATE = """Extract all information from the following CV text and return the JSON object as specified.

CV TEXT:
{cv_text}
"""

# Prompt for chunked extraction (long CVs)
_MERGE_PROMPT_TEMPLATE = """You are merging multiple partial JSON extractions of the same CV.
Combine them into one final JSON object following the same schema.
Prefer non-null values over null values when there is a conflict.
Merge arrays by combining all items (deduplicate by title/name where obvious).
Return ONLY valid JSON.

PARTIAL EXTRACTIONS:
{partial_jsons}
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMExtractor:
    """
    Uses an LLM (OpenAI GPT by default) to extract structured data from CV text.

    Usage
    -----
    extractor = LLMExtractor(api_key="sk-...", model="gpt-4o-mini")
    result = extractor.extract(candidate_filename="john_doe", cv_text="...")
    if result.success:
        print(result.data["personal_info"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
        model : str
            OpenAI model name. gpt-4o-mini is cost-effective and accurate.
            Use gpt-4o for higher accuracy on complex CVs.
        temperature : float
            0.0 means deterministic output (best for structured extraction).
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.temperature = temperature

        if not self.api_key:
            logger.warning(
                "No OpenAI API key found. Set the OPENAI_API_KEY environment "
                "variable or pass api_key= to LLMExtractor()."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, candidate_filename: str, cv_text: str) -> ExtractionResult:
        """
        Extract structured information from CV text.

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
            # Choose single-call or chunked extraction based on text length
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

    def _call_openai(self, system: str, user: str) -> str:
        """Make one OpenAI chat completion call and return the response text."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},  # Forces valid JSON output
        )
        return response.choices[0].message.content or ""

    def _extract_single(self, cv_text: str) -> str:
        """Extract data from the entire CV in one API call."""
        user_prompt = _USER_PROMPT_TEMPLATE.format(cv_text=cv_text)
        return self._call_openai(_SYSTEM_PROMPT, user_prompt)

    def _extract_chunked(self, cv_text: str) -> str:
        """
        Split the CV into overlapping chunks, extract from each, then merge.
        This is used when the CV text is too long for a single call.
        Overlap of 200 chars ensures context at boundaries is not lost.
        """
        chunks = self._split_into_chunks(cv_text, max_chars=_MAX_CHARS_PER_CALL, overlap=200)
        logger.debug("Chunked into %d parts.", len(chunks))

        partial_jsons = []
        for i, chunk in enumerate(chunks):
            logger.debug("Extracting chunk %d/%d …", i + 1, len(chunks))
            user_prompt = _USER_PROMPT_TEMPLATE.format(cv_text=chunk)
            raw = self._call_openai(_SYSTEM_PROMPT, user_prompt)
            partial_jsons.append(raw)

        # Merge all partial JSONs into one final result
        merge_user = _MERGE_PROMPT_TEMPLATE.format(
            partial_jsons="\n---\n".join(partial_jsons)
        )
        return self._call_openai(_SYSTEM_PROMPT, merge_user)

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
            start = end - overlap  # overlap to avoid losing context at boundaries
        return chunks
