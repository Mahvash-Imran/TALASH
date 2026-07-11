"""
gap_justifier.py  –  Task 2.8: Gap Justification via Experience
================================================================

WHY THIS FILE EXISTS
--------------------
Task 2.8 requires cross-referencing the educational gaps detected in 2.7
with the candidate's employment/experience records from experience.csv.

A gap is "justified" if any experience record's date range overlaps (or is
adjacent to) the gap period.

DESIGN DECISIONS
----------------
- Pure Python / pandas — no LLM needed for this step.
- Date parsing is lenient: year-only or YYYY-MM formats both work.
- "Overlap" is defined as: experience_start <= gap_end AND
                            experience_end   >= gap_start
  with a 6-month grace window on both sides to handle imprecise dates.
- "present" / "ongoing" end dates are treated as the current year.
"""

import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CURRENT_YEAR = date.today().year


class GapJustifier:
    """
    Task 2.8: Cross-reference educational gaps against experience records.

    Usage
    -----
    justifier = GapJustifier()
    gaps_with_justification = justifier.justify(gaps, experience_rows)
    """

    def justify(
        self,
        gaps: List[Dict[str, Any]],
        experience_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        For each gap, check if any experience record overlaps the gap period.
        Returns the same gap list with 'justified_by' and 'justification_type'
        fields populated.

        Parameters
        ----------
        gaps : list of dict
            Output of ProgressionChecker._detect_gaps().  Each has
            'end_year', 'start_year', 'significant'.
        experience_rows : list of dict
            Rows from experience.csv for the same candidate.  Expected columns:
            job_title, organization, start_date, end_date, employment_type.

        Returns
        -------
        list of dict  (same as input with 'justified_by' and 'justification_type' added)
        """
        parsed_exp = [self._parse_experience(e) for e in experience_rows]

        result = []
        for gap in gaps:
            enriched = dict(gap)
            gap_start = gap.get("end_year")    # year when previous degree ended
            gap_end   = gap.get("start_year")  # year when next degree started

            if gap_start is None or gap_end is None or not gap.get("significant"):
                enriched["justified_by"]        = None
                enriched["justification_type"]  = None
                result.append(enriched)
                continue

            justification = self._find_overlap(gap_start, gap_end, parsed_exp)
            enriched["justified_by"]       = justification["description"]
            enriched["justification_type"] = justification["type"]
            result.append(enriched)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_overlap(
        self,
        gap_start_year: int,
        gap_end_year: int,
        parsed_exp: List[Dict],
    ) -> Dict:
        """
        Check each experience entry for overlap with [gap_start_year, gap_end_year].
        Returns the best matching justification, or 'Unexplained'.
        """
        # Grace window: 6 months ~ 0.5 years — we work in year fractions
        grace = 0.5

        for exp in parsed_exp:
            exp_start = exp.get("start_year_float")
            exp_end   = exp.get("end_year_float")
            if exp_start is None:
                continue

            # Overlap condition: ranges intersect (with grace)
            if exp_start <= (gap_end_year + grace) and (exp_end or _CURRENT_YEAR) >= (gap_start_year - grace):
                org  = exp.get("organization") or "Unknown Organization"
                role = exp.get("job_title") or "Employment"
                etype = str(exp.get("employment_type") or "").strip()

                jtype = self._classify_justification(etype, role)
                return {
                    "description": f"{jtype} at {org} ({int(exp_start) if exp_start else '?'}–{int(exp_end) if exp_end else 'present'})",
                    "type": jtype,
                }

        return {"description": None, "type": "Unexplained"}

    @staticmethod
    def _classify_justification(employment_type: str, job_title: str) -> str:
        """Map employment type + title to a human-readable justification type."""
        combined = (employment_type + " " + job_title).lower()
        if any(k in combined for k in ["research", "phd", "doctoral", "fellow"]):
            return "Research / Doctoral Work"
        if any(k in combined for k in ["teach", "lectur", "professor", "faculty", "instructor"]):
            return "Teaching"
        if any(k in combined for k in ["intern"]):
            return "Internship"
        if any(k in combined for k in ["part", "visiting"]):
            return "Part-time Employment"
        if any(k in combined for k in ["contract"]):
            return "Contract Employment"
        return "Employment"

    @staticmethod
    def _parse_experience(exp_row: Dict) -> Dict:
        """Parse start/end dates from an experience row into year floats."""
        def _to_year_float(s: Optional[str]) -> Optional[float]:
            if not s:
                return None
            s = str(s).strip().lower()
            if s in ("present", "ongoing", "current", "till date", "to date", "nan"):
                return float(_CURRENT_YEAR)
            # YYYY-MM
            m = re.match(r"(\d{4})-(\d{2})", s)
            if m:
                return int(m.group(1)) + (int(m.group(2)) - 1) / 12.0
            # YYYY
            m = re.match(r"(\d{4})", s)
            if m:
                return float(m.group(1))
            return None

        return {
            "job_title":       exp_row.get("job_title"),
            "organization":    exp_row.get("organization"),
            "employment_type": exp_row.get("employment_type"),
            "start_year_float": _to_year_float(str(exp_row.get("start_date") or "")),
            "end_year_float":   _to_year_float(str(exp_row.get("end_date")   or "")),
        }
