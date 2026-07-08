"""
normalizer.py  –  Task 1.3: Data Cleaning & Normalization
==========================================================

WHY THIS FILE EXISTS
--------------------
The plan (Section 1.3) requires:
  • Strip extra whitespace, fix encoding issues
  • Standardize date formats → YYYY-MM or YYYY
  • Normalize CGPA scales (e.g., /4.0, /5.0) to a common 4.0 scale
  • Normalize percentage → record original + flag scale
  • Deduplicate publications by title similarity

DESIGN DECISIONS
----------------
- CGPA normalization: we convert all CGPAs to /4.0 (the most common
  Pakistani/international scale). The original value AND original scale are
  both stored so no information is lost (the plan explicitly says to preserve
  the original).
- Date parsing is lenient: it handles "Sep 2018", "2018-09", "September 2018",
  "2018", "Present", "Ongoing", etc. All are normalized to YYYY-MM or YYYY.
  "Present" and "Ongoing" become the sentinel string "present".
- Publication deduplication uses Jaccard similarity on title tokens — a simple
  but effective approach that requires no extra dependencies. Two publications
  with >70% token overlap are considered duplicates; the one with more
  non-null fields is kept.
- Text normalization fixes common PDF extraction artefacts: ligatures (ﬁ→fi),
  non-breaking spaces, control characters, Windows line-endings.
"""

import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CGPA scale detection thresholds
_SCALE_BOUNDARIES = [
    (4.0, 4.0),   # if max raw CGPA ≤ 4.0  → scale 4.0
    (5.0, 5.0),   # if max raw CGPA ≤ 5.0  → scale 5.0
    (10.0, 10.0), # if max raw CGPA ≤ 10.0 → scale 10.0 (rare but used in India)
    (100.0, 100.0),  # percentage disguised as CGPA
]

# Month abbreviation → zero-padded number
_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    "january": "01", "february": "02", "march": "03", "april": "04",
    "june": "06", "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12",
}

# Jaccard similarity threshold for duplicate publication detection
_DUP_THRESHOLD = 0.70

# PDF ligature / encoding fixes
_LIGATURE_MAP = {
    "\ufb01": "fi",  # ﬁ
    "\ufb02": "fl",  # ﬂ
    "\ufb00": "ff",  # ﬀ
    "\ufb03": "ffi", # ﬃ
    "\ufb04": "ffl", # ﬄ
    "\u2013": "-",   # en-dash
    "\u2014": "-",   # em-dash
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Cleans and normalizes structured data extracted by LLMExtractor.

    Takes the raw dict produced by LLMExtractor.extract().data and returns
    a cleaned version ready for export to CSV/Excel.

    Usage
    -----
    norm = Normalizer()
    clean_data = norm.normalize(raw_extracted_dict)
    """

    def normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all normalization steps on the extracted data dict.

        Parameters
        ----------
        data : dict
            The dict from ExtractionResult.data

        Returns
        -------
        dict
            Cleaned and normalized version of data (new object, original unchanged)
        """
        import copy
        d = copy.deepcopy(data)

        # --- 1. Personal info: clean strings ---
        if d.get("personal_info"):
            d["personal_info"] = self._clean_dict_strings(d["personal_info"])

        # --- 2. Education: normalize dates, CGPA, strings ---
        d["education"] = [
            self._normalize_education_entry(e)
            for e in (d.get("education") or [])
        ]

        # --- 3. Experience: normalize dates, clean strings ---
        d["experience"] = [
            self._normalize_experience_entry(e)
            for e in (d.get("experience") or [])
        ]

        # --- 4. Skills: clean strings ---
        d["skills"] = [
            self._clean_dict_strings(s)
            for s in (d.get("skills") or [])
        ]

        # --- 5. Publications: normalize year, clean strings, deduplicate ---
        pubs = [
            self._normalize_publication_entry(p)
            for p in (d.get("publications") or [])
        ]
        before_dedup = len(pubs)
        pubs = self._deduplicate_by_title(pubs, key="title")
        after_dedup = len(pubs)
        if before_dedup != after_dedup:
            logger.info(
                "Deduplicated publications: %d → %d (removed %d duplicates).",
                before_dedup, after_dedup, before_dedup - after_dedup,
            )
        d["publications"] = pubs

        # --- 6. Supervision: normalize year, clean strings ---
        d["supervision"] = [
            self._normalize_supervision_entry(s)
            for s in (d.get("supervision") or [])
        ]

        # --- 7. Books: deduplicate by title, clean strings ---
        books = [
            self._clean_dict_strings(b)
            for b in (d.get("books") or [])
        ]
        d["books"] = self._deduplicate_by_title(books, key="title")

        # --- 8. Patents: clean strings ---
        d["patents"] = [
            self._clean_dict_strings(p)
            for p in (d.get("patents") or [])
        ]

        return d

    # ------------------------------------------------------------------
    # Per-section normalizers
    # ------------------------------------------------------------------

    def _normalize_education_entry(self, entry: Dict) -> Dict:
        entry = self._clean_dict_strings(entry)

        # Normalize years
        entry["start_year"] = self._normalize_year(entry.get("start_year"))
        entry["end_year"]   = self._normalize_year(entry.get("end_year"))

        # Normalize CGPA
        raw_cgpa  = self._parse_float(entry.get("cgpa"))
        raw_scale = self._parse_float(entry.get("cgpa_scale"))
        if raw_cgpa is not None:
            detected_scale, normalized_cgpa = self._normalize_cgpa(raw_cgpa, raw_scale)
            entry["cgpa"]              = raw_cgpa          # keep original
            entry["cgpa_scale"]        = detected_scale    # detected / confirmed scale
            entry["cgpa_normalized_4"] = round(normalized_cgpa, 3)  # /4.0 equivalent
        else:
            entry["cgpa"]              = None
            entry["cgpa_scale"]        = None
            entry["cgpa_normalized_4"] = None

        # Normalize percentage
        raw_pct = self._parse_float(entry.get("marks_percentage"))
        if raw_pct is not None:
            entry["marks_percentage"]          = raw_pct
            entry["marks_percentage_original"] = raw_pct  # preserved
        else:
            entry["marks_percentage"] = None
            entry["marks_percentage_original"] = None

        return entry

    def _normalize_experience_entry(self, entry: Dict) -> Dict:
        entry = self._clean_dict_strings(entry)
        entry["start_date"] = self._normalize_date(entry.get("start_date"))
        entry["end_date"]   = self._normalize_date(entry.get("end_date"))
        return entry

    def _normalize_publication_entry(self, entry: Dict) -> Dict:
        entry = self._clean_dict_strings(entry)
        entry["year"] = self._normalize_year(entry.get("year"))
        # Ensure type is lowercase
        if entry.get("type"):
            entry["type"] = str(entry["type"]).lower().strip()
        return entry

    def _normalize_supervision_entry(self, entry: Dict) -> Dict:
        entry = self._clean_dict_strings(entry)
        entry["year"] = self._normalize_year(entry.get("year"))
        if entry.get("level"):
            entry["level"] = str(entry["level"]).upper().strip()
        if entry.get("role"):
            entry["role"] = str(entry["role"]).lower().strip()
        return entry

    # ------------------------------------------------------------------
    # String cleaning
    # ------------------------------------------------------------------

    def _clean_string(self, value: Any) -> Optional[str]:
        """
        Clean a single string value:
        - Return None if null-like
        - Fix PDF ligatures and encoding artefacts
        - Strip extra whitespace and control characters
        - Collapse multiple spaces/newlines
        """
        if value is None:
            return None
        s = str(value)

        # Fix ligatures and special chars
        for bad, good in _LIGATURE_MAP.items():
            s = s.replace(bad, good)

        # Remove control characters (except normal whitespace)
        s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", s)

        # Normalize whitespace
        s = re.sub(r"\r\n|\r", "\n", s)       # normalise line endings
        s = re.sub(r"[ \t]+", " ", s)          # collapse horizontal whitespace
        s = re.sub(r"\n{3,}", "\n\n", s)       # max 2 consecutive newlines
        s = s.strip()

        # Return None for effectively-empty strings
        null_tokens = {"n/a", "na", "none", "null", "not available",
                       "not applicable", "-", "–", "—", ""}
        if s.lower() in null_tokens:
            return None

        return s

    def _clean_dict_strings(self, d: Dict) -> Dict:
        """Apply _clean_string to every string value in a dict."""
        return {
            k: (self._clean_string(v) if isinstance(v, (str, type(None))) else v)
            for k, v in d.items()
        }

    # ------------------------------------------------------------------
    # Date normalization
    # ------------------------------------------------------------------

    def _normalize_date(self, value: Any) -> Optional[str]:
        """
        Normalize a date-like value to YYYY-MM or YYYY.
        Handles 'Present', 'Ongoing', 'current', etc.
        Returns None for unparseable inputs.
        """
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in {"null", "none", "n/a", ""}:
            return None

        # Ongoing / present markers
        if s.lower() in {"present", "ongoing", "current", "to date", "till date",
                         "till now", "to now", "till present", "up to now"}:
            return "present"

        # Try YYYY-MM
        m = re.match(r"^(\d{4})[/-](\d{1,2})$", s)
        if m:
            y, mo = m.group(1), m.group(2).zfill(2)
            if 1 <= int(mo) <= 12:
                return f"{y}-{mo}"

        # Try Month YYYY or YYYY Month
        m = re.match(
            r"^(?:(\w+)\s+(\d{4})|(\d{4})\s+(\w+))$", s, re.IGNORECASE
        )
        if m:
            month_name = (m.group(1) or m.group(4)).lower()
            year       = m.group(2) or m.group(3)
            month_num  = _MONTHS.get(month_name[:3])
            if month_num:
                return f"{year}-{month_num}"

        # Try plain YYYY
        m = re.match(r"^(\d{4})$", s)
        if m:
            return m.group(1)

        # Try DD/MM/YYYY or MM/DD/YYYY → store as YYYY-MM
        m = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
        if m:
            a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
            # Heuristic: if first number > 12 it must be a day
            if a > 12:
                return f"{year}-{str(b).zfill(2)}"
            else:
                return f"{year}-{str(a).zfill(2)}"

        logger.debug("Could not normalize date: '%s'", s)
        return s  # return as-is rather than losing info

    def _normalize_year(self, value: Any) -> Optional[str]:
        """Extract a 4-digit year string from a value."""
        if value is None:
            return None
        s = str(value).strip()
        m = re.search(r"\b(\d{4})\b", s)
        if m:
            return m.group(1)
        if s.lower() == "present":
            return "present"
        return None

    # ------------------------------------------------------------------
    # CGPA / Marks normalization
    # ------------------------------------------------------------------

    def _parse_float(self, value: Any) -> Optional[float]:
        """Safely parse a float from a string or number."""
        if value is None:
            return None
        try:
            cleaned = re.sub(r"[^\d.]", "", str(value))
            return float(cleaned) if cleaned else None
        except (ValueError, TypeError):
            return None

    def _normalize_cgpa(
        self, cgpa: float, declared_scale: Optional[float]
    ):
        """
        Normalize a CGPA value to a /4.0 equivalent.

        Returns
        -------
        (detected_scale, normalized_cgpa_on_4)
        """
        # If the caller provided a scale, trust it
        if declared_scale and declared_scale > 0:
            scale = declared_scale
        else:
            # Auto-detect: pick smallest standard scale that accommodates the value
            scale = 4.0  # default
            for upper, std_scale in _SCALE_BOUNDARIES:
                if cgpa <= upper:
                    scale = std_scale
                    break

        if scale == 0:
            scale = 4.0  # safety

        normalized = (cgpa / scale) * 4.0
        normalized = min(normalized, 4.0)  # clamp to 4.0 max

        return scale, normalized

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_by_title(
        self, items: List[Dict], key: str = "title"
    ) -> List[Dict]:
        """
        Remove duplicate items where two entries have highly similar titles
        (Jaccard similarity ≥ _DUP_THRESHOLD). Keep the one with more
        non-null fields (more complete record).
        """
        if not items:
            return items

        kept: List[Dict] = []
        for candidate in items:
            title_a = self._tokenize(candidate.get(key) or "")
            is_duplicate = False
            for i, existing in enumerate(kept):
                title_b = self._tokenize(existing.get(key) or "")
                if title_a and title_b and self._jaccard(title_a, title_b) >= _DUP_THRESHOLD:
                    # Keep the more complete record
                    if self._completeness(candidate) > self._completeness(existing):
                        kept[i] = candidate
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(candidate)

        return kept

    @staticmethod
    def _tokenize(text: str) -> set:
        """Lowercase word tokens for Jaccard comparison."""
        return set(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        """Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _completeness(d: Dict) -> int:
        """Count non-null fields in a dict (used to pick the richer duplicate)."""
        return sum(1 for v in d.values() if v is not None)
