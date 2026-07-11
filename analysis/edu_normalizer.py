"""
edu_normalizer.py  –  Tasks 2.1, 2.2, 2.3, 2.4
=================================================

WHY THIS FILE EXISTS
--------------------
Tasks 2.1–2.4 are pure data-processing steps on the rows already stored in
education.csv by Module 1.  No LLM is needed here.

  2.1  Extract SSC/HSSC records; flag missing as "Not provided"
  2.2  Extract UG/PG/PhD records; support multiple Pakistani degree pathways
  2.3  Normalize marks: percentage, CGPA/4, CGPA/5, CGPA/10, division text
       Output: marks_original, marks_scale, marks_normalized (0-100%)
  2.4  Map raw degree strings to canonical level labels
       Output: standard_level in {SSC, HSSC, UG, PG, PhD, Other}

DESIGN DECISIONS
----------------
- We classify by matching the raw 'level' field from Module 1 first, then fall
  back to keyword-matching the 'degree' string.  This handles the common case
  where the LLM put "BS Computer Science" in the 'degree' column and left
  'level' as null.
- Marks normalization converts everything to a 0-100 float so gap analysis and
  trend analysis in tasks 2.6-2.7 can work numerically.
- Division labels ("First", "Second", "Third") are mapped to midpoints of
  their conventional percentage ranges.
- All original values are preserved; only new columns are added.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level classification keyword maps
# ---------------------------------------------------------------------------

_SSC_KEYWORDS = {
    "ssc", "matric", "matriculation", "secondary school certificate",
    "secondary", "o level", "o-level", "class 10", "grade 10", "sse",
    "middle", "9th", "10th",
}
_HSSC_KEYWORDS = {
    "hssc", "intermediate", "fsc", "fa", "inter", "f.sc", "f.a",
    "a level", "a-level", "class 12", "grade 12", "higher secondary",
    "pre-engineering", "pre-medical", "ics", "icom",
}
_UG_KEYWORDS = {
    "bs", "be", "beng", "b.e", "bsc", "b.sc", "ba", "b.a", "bba", "bcs",
    "bachelor", "undergraduate", "b.tech", "btech", "bcom", "b.com",
    "b.arch", "barch", "bpharm", "bfa",
}
_PG_KEYWORDS = {
    "ms", "msc", "m.sc", "ma", "m.a", "mphil", "m.phil", "mba", "mpa",
    "master", "postgraduate", "pg", "m.tech", "mtech", "med", "mcs",
    "llm", "mfin",
}
_PHD_KEYWORDS = {
    "phd", "ph.d", "doctorate", "doctoral", "dphil", "d.phil",
    "doctor of philosophy", "doctor of science", "doctorat",
}


# ---------------------------------------------------------------------------
# Marks normalization tables
# ---------------------------------------------------------------------------

_DIVISION_MAP = {
    "first":  75.0, "1st": 75.0, "first division":  75.0,
    "second": 62.5, "2nd": 62.5, "second division": 62.5,
    "third":  52.5, "3rd": 52.5, "third division":  52.5,
    "distinction": 85.0,
    "pass": 45.0,
}


class EduNormalizer:
    """
    Tasks 2.1-2.4: Classify and normalize education records.

    Usage
    -----
    normalizer = EduNormalizer()
    enriched_rows = normalizer.process(education_rows)
    # enriched_rows is a list of dicts with new columns added:
    #   standard_level, marks_original, marks_scale, marks_normalized
    """

    def process(self, education_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of raw education row dicts (from education.csv).

        Parameters
        ----------
        education_rows : list of dict
            Each dict is one row from education.csv.

        Returns
        -------
        list of dict
            Same rows with these extra keys added:
              - standard_level      : SSC | HSSC | UG | PG | PhD | Other | Not provided
              - marks_original      : raw value as string (e.g. "3.8/4.0", "78%")
              - marks_scale         : "percentage" | "cgpa_4" | "cgpa_5" | "cgpa_10" | "division" | null
              - marks_normalized    : float 0-100, or null if cannot be computed
        """
        result = []
        for row in education_rows:
            enriched = dict(row)
            enriched["standard_level"] = self._classify_level(row)
            marks_orig, scale, norm = self._normalize_marks(row)
            enriched["marks_original"]   = marks_orig
            enriched["marks_scale"]      = scale
            enriched["marks_normalized"] = norm
            result.append(enriched)
        return result

    # ------------------------------------------------------------------
    # Tasks 2.1 & 2.2 & 2.4: Level Classification
    # ------------------------------------------------------------------

    def _classify_level(self, row: Dict) -> str:
        """
        Map a raw education row to one of: SSC, HSSC, UG, PG, PhD, Other.
        Strategy: check 'level' field first (Module 1 LLM output), then
        fall back to keyword-matching 'degree' field.
        Returns "Not provided" if the row is essentially empty.
        """
        raw_level  = str(row.get("level")  or "").strip().lower()
        raw_degree = str(row.get("degree") or "").strip().lower()

        if not raw_level and not raw_degree:
            return "Not provided"

        # Direct level field match
        if raw_level:
            if raw_level in {"ssc", "sse", "matric", "secondary"}:
                return "SSC"
            if raw_level in {"hssc", "intermediate", "fsc", "fa"}:
                return "HSSC"
            if raw_level in {"bs", "bsc", "be", "beng", "ba", "bba", "bachelor"}:
                return "UG"
            if raw_level in {"ms", "msc", "mphil", "ma", "master", "pg", "mba"}:
                return "PG"
            if raw_level in {"phd", "doctorate"}:
                return "PhD"

        # Fall back to keyword scan of degree string
        candidate = raw_level + " " + raw_degree
        tokens = set(re.findall(r"\b\w+\b", candidate))

        if tokens & _PHD_KEYWORDS:
            return "PhD"
        if tokens & _PG_KEYWORDS:
            return "PG"
        if tokens & _UG_KEYWORDS:
            return "UG"
        if tokens & _HSSC_KEYWORDS:
            return "HSSC"
        if tokens & _SSC_KEYWORDS:
            return "SSC"

        return "Other"

    # ------------------------------------------------------------------
    # Task 2.3: Marks Normalization
    # ------------------------------------------------------------------

    def _normalize_marks(self, row: Dict):
        """
        Returns (marks_original: str, scale: str | None, normalized: float | None).

        Handles:
          - Percentages (50–100 range, or explicit % sign)
          - CGPA values with explicit scale (e.g. "3.8/4.0")
          - CGPA values without scale (guessed from magnitude)
          - Division text ("First", "Second Division")
          - null / missing
        """
        cgpa       = row.get("cgpa")
        cgpa_scale = row.get("cgpa_scale")
        marks_pct  = row.get("marks_percentage")

        # --- Percentage from marks_percentage field --------------------
        if marks_pct is not None and str(marks_pct).strip():
            raw_str = str(marks_pct).strip()
            pct = self._parse_percentage(raw_str)
            if pct is not None:
                return (raw_str, "percentage", round(pct, 2))

        # --- CGPA with explicit scale ---------------------------------
        if cgpa is not None and str(cgpa).strip():
            cgpa_str = str(cgpa).strip()
            scale_str = str(cgpa_scale).strip() if cgpa_scale else ""
            cgpa_val = self._parse_float(cgpa_str)

            if cgpa_val is not None:
                # Explicit scale provided by LLM
                scale_val = self._parse_float(scale_str)
                if scale_val:
                    norm = (cgpa_val / scale_val) * 100.0
                    marks_orig = f"{cgpa_val}/{scale_val}"
                    return (marks_orig, f"cgpa_{scale_val:.0f}", round(min(norm, 100.0), 2))

                # Infer scale from magnitude
                if cgpa_val <= 4.0:
                    return (cgpa_str, "cgpa_4", round((cgpa_val / 4.0) * 100.0, 2))
                elif cgpa_val <= 5.0:
                    return (cgpa_str, "cgpa_5", round((cgpa_val / 5.0) * 100.0, 2))
                elif cgpa_val <= 10.0:
                    return (cgpa_str, "cgpa_10", round((cgpa_val / 10.0) * 100.0, 2))
                elif cgpa_val <= 100.0:
                    return (cgpa_str, "percentage", round(cgpa_val, 2))

        # --- Division text -------------------------------------------
        combined = " ".join(str(v) for v in [
            row.get("marks_percentage"), row.get("cgpa"), row.get("degree")
        ] if v).lower()
        for key, pct in _DIVISION_MAP.items():
            if key in combined:
                return (key, "division", pct)

        return (None, None, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_percentage(s: str) -> Optional[float]:
        """Parse a string that might represent a percentage (e.g. '78%', '78.5', '780/1000')."""
        s = s.replace(",", ".").strip()
        # Explicit % sign
        m = re.match(r"^(\d+(?:\.\d+)?)\s*%$", s)
        if m:
            return float(m.group(1))
        # Fraction like 769/1000
        m = re.match(r"^(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            if den > 0:
                return (num / den) * 100.0
        # Bare number in percentage range
        m = re.match(r"^(\d+(?:\.\d+)?)$", s)
        if m:
            val = float(m.group(1))
            if val > 10.0:   # likely a percentage, not a CGPA
                return min(val, 100.0)
        return None

    @staticmethod
    def _parse_float(s: str) -> Optional[float]:
        """Parse a plain float string, return None on failure."""
        try:
            return float(str(s).replace(",", ".").strip())
        except (ValueError, TypeError):
            return None
