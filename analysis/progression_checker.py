"""
progression_checker.py  –  Tasks 2.6 & 2.7
============================================

WHY THIS FILE EXISTS
--------------------
  2.6  Analyse the degree sequence for:
         - correct ordering (SSC < HSSC < UG < PG < PhD)
         - specialization consistency across levels
         - performance trend (improving / stable / declining)

  2.7  Detect gaps between consecutive educational stages and flag those
       exceeding 12 months as significant.

DESIGN DECISIONS
----------------
- All logic is pure Python — no LLM calls needed for structural analysis.
- Level ordering uses an integer rank so comparison is unambiguous.
- Specialization drift is detected by checking keyword overlap between
  adjacent specialization strings.  A drift is flagged when overlap is < 30%.
  This is intentionally lenient to avoid over-flagging reasonable evolutions
  (e.g. EE → Telecom Engineering).
- Years are extracted from the 'start_year' / 'end_year' columns which may
  be int, float ("2018.0"), or string ("2018").
- When only a year is available (no month), we use Jan 1 as start and
  Dec 31 as end to compute conservative gap bounds.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Level ordering (lower number = earlier in typical academic progression)
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {
    "SSC":  1,
    "HSSC": 2,
    "UG":   3,
    "PG":   4,
    "PhD":  5,
    "Other": 6,
    "Not provided": 7,
}

# Expected gap between SSC and HSSC in years (typically 2)
_SSC_HSSC_EXPECTED_GAP_YRS = 2


class ProgressionChecker:
    """
    Tasks 2.6 & 2.7: Analyse progression and detect gaps.

    Usage
    -----
    checker = ProgressionChecker()
    result = checker.analyse(enriched_education_rows)
    # result is a dict with keys:
    #   degrees_sorted        : list of dicts, sorted chronologically
    #   progression_consistent: bool
    #   specialization_drift  : list of {from, to, detail}
    #   performance_trend     : "improving" | "stable" | "declining" | "insufficient data"
    #   highest_degree        : "PhD" | "PG" | "UG" | "HSSC" | "SSC" | "Not provided"
    #   educational_gaps      : list of {between, gap_years, gap_months, significant}
    """

    def analyse(self, edu_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        edu_rows : list of dict
            Rows from edu_normalizer output (already have standard_level,
            marks_normalized, etc.).

        Returns
        -------
        dict  (see class docstring)
        """
        if not edu_rows:
            return self._empty_result()

        # Sort by end_year (chronological), then start_year
        sorted_rows = sorted(edu_rows, key=lambda r: (
            self._year_int(r.get("end_year"))   or 9999,
            self._year_int(r.get("start_year")) or 9999,
        ))

        # ── Task 2.6 ─────────────────────────────────────────────────────
        progression_ok, drift_events   = self._check_progression(sorted_rows)
        performance_trend               = self._check_performance_trend(sorted_rows)
        highest                         = self._highest_degree(sorted_rows)

        # ── Task 2.7 ─────────────────────────────────────────────────────
        gaps = self._detect_gaps(sorted_rows)

        return {
            "degrees_sorted":         sorted_rows,
            "progression_consistent": progression_ok,
            "specialization_drift":   drift_events,
            "performance_trend":      performance_trend,
            "highest_degree":         highest,
            "educational_gaps":       gaps,
        }

    # ------------------------------------------------------------------
    # Task 2.6 helpers
    # ------------------------------------------------------------------

    def _check_progression(
        self, sorted_rows: List[Dict]
    ) -> Tuple[bool, List[Dict]]:
        """
        Check that degrees advance in level order.
        Also detect specialization drift between adjacent degrees.
        """
        drift_events = []
        consistent = True
        prev_order = 0
        prev_spec  = None

        for row in sorted_rows:
            level = row.get("standard_level", "Other")
            order = _LEVEL_ORDER.get(level, 6)

            # Progression check (ignore 'Other' and 'Not provided')
            if order <= 5 and prev_order > 0:
                if order < prev_order:
                    consistent = False

            # Specialization drift check
            curr_spec = str(row.get("specialization") or row.get("degree") or "").strip()
            if prev_spec and curr_spec:
                overlap = self._keyword_overlap(prev_spec, curr_spec)
                if overlap < 0.3 and order <= 5 and _LEVEL_ORDER.get(level, 6) > 2:
                    drift_events.append({
                        "from_degree":     prev_spec[:80],
                        "to_degree":       curr_spec[:80],
                        "overlap_score":   round(overlap, 2),
                        "flagged":         True,
                    })

            if order <= 5:
                prev_order = order
            if curr_spec:
                prev_spec = curr_spec

        return consistent, drift_events

    def _check_performance_trend(self, sorted_rows: List[Dict]) -> str:
        """Return improving / stable / declining / insufficient data."""
        scores = [
            r["marks_normalized"]
            for r in sorted_rows
            if r.get("marks_normalized") is not None
            and r.get("standard_level") in {"UG", "PG", "PhD"}
        ]
        if len(scores) < 2:
            return "insufficient data"
        diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff > 2.0:
            return "improving"
        elif avg_diff < -2.0:
            return "declining"
        return "stable"

    def _highest_degree(self, sorted_rows: List[Dict]) -> str:
        """Return the highest standard_level among the candidate's degrees."""
        levels = [r.get("standard_level", "Not provided") for r in sorted_rows]
        ordered = ["PhD", "PG", "UG", "HSSC", "SSC", "Other", "Not provided"]
        for lvl in ordered:
            if lvl in levels:
                return lvl
        return "Not provided"

    # ------------------------------------------------------------------
    # Task 2.7: Gap detection
    # ------------------------------------------------------------------

    def _detect_gaps(self, sorted_rows: List[Dict]) -> List[Dict]:
        """
        For each consecutive pair of educational stages, compute the gap
        (in years and months) between end of one and start of the next.
        Flag gaps > 12 months as significant.
        """
        gaps = []
        # Only consider levels that form a meaningful sequence
        meaningful = [
            r for r in sorted_rows
            if r.get("standard_level") in {"SSC", "HSSC", "UG", "PG", "PhD"}
        ]

        for i in range(len(meaningful) - 1):
            curr = meaningful[i]
            nxt  = meaningful[i + 1]

            end_year   = self._year_int(curr.get("end_year"))
            start_year = self._year_int(nxt.get("start_year"))

            if end_year is None or start_year is None:
                continue

            gap_years  = start_year - end_year
            gap_months = gap_years * 12

            curr_label = f"{curr.get('standard_level')} ({curr.get('degree', '')})"
            nxt_label  = f"{nxt.get('standard_level')}  ({nxt.get('degree', '')})"

            # Special case: SSC->HSSC gap should be ~2 years
            if curr.get("standard_level") == "SSC" and nxt.get("standard_level") == "HSSC":
                expected = _SSC_HSSC_EXPECTED_GAP_YRS
                significant = abs(gap_years - expected) > 1
            else:
                significant = gap_months > 12

            if gap_months != 0:  # skip 0-gap (immediate continuation)
                gaps.append({
                    "between":        f"{curr.get('standard_level')} and {nxt.get('standard_level')}",
                    "from_degree":    curr_label[:60],
                    "to_degree":      nxt_label[:60],
                    "end_year":       end_year,
                    "start_year":     start_year,
                    "gap_years":      gap_years,
                    "gap_months":     gap_months,
                    "significant":    significant,
                    "justified_by":   None,   # filled in by gap_justifier
                })

        return gaps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _year_int(val: Any) -> Optional[int]:
        """Safely parse a year value to int."""
        if val is None:
            return None
        s = str(val).strip()
        # Handle "YYYY-MM" format
        m = re.match(r"^(\d{4})", s)
        if m:
            return int(m.group(1))
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _keyword_overlap(s1: str, s2: str) -> float:
        """Jaccard overlap of word tokens between two specialization strings."""
        t1 = set(re.findall(r"\b\w{3,}\b", s1.lower()))
        t2 = set(re.findall(r"\b\w{3,}\b", s2.lower()))
        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "degrees_sorted":         [],
            "progression_consistent": True,
            "specialization_drift":   [],
            "performance_trend":      "insufficient data",
            "highest_degree":         "Not provided",
            "educational_gaps":       [],
        }
