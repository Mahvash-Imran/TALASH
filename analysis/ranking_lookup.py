"""
ranking_lookup.py  –  Task 2.5: Institutional Quality Assessment
=================================================================

WHY THIS FILE EXISTS
--------------------
Task 2.5 requires associating each degree institution with its THE and QS
world university ranking.

DESIGN DECISIONS
----------------
- We use a LOCAL static CSV (data/rankings/universities.csv) built from the
  actual institutions mentioned in the dataset.  This satisfies the plan's
  requirement: "Never assume a rank – only record if confirmed from source".
- Fuzzy matching (rapidfuzz) handles spelling variants, abbreviations and
  partial names (e.g. "NUST" vs "National University of Sciences and Technology").
- If no match above the confidence threshold is found we record "Not Ranked"
  for both THE and QS.  We never invent or guess a rank.
- The CSV aliases column (semicolon-separated) lets us pre-register known
  abbreviations so common cases never go unmatched even at high thresholds.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Default path to the ranking CSV
_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "rankings" / "universities.csv"

# Minimum fuzzy-match score (0-100) to accept a match
_MATCH_THRESHOLD = 72

NOT_RANKED = "Not Ranked"


class RankingLookup:
    """
    Task 2.5: Match institution names against a local ranking table.

    Usage
    -----
    lookup = RankingLookup()
    result = lookup.get_ranking("NUST")
    # {"the_rank": 401, "the_rank_range": "401-500",
    #  "qs_rank": 381, "qs_rank_range": "381-390",
    #  "matched_name": "National University of Sciences and Technology",
    #  "match_score": 95}
    """

    def __init__(self, csv_path: Optional[str] = None):
        path = Path(csv_path) if csv_path else _DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"Rankings CSV not found: {path}\n"
                "Expected at: data/rankings/universities.csv"
            )
        self._df = pd.read_csv(path, dtype=str).fillna("")
        self._index = self._build_index()
        logger.info("RankingLookup: loaded %d institutions from %s", len(self._df), path.name)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def get_ranking(self, institution_name: str) -> Dict[str, Any]:
        """
        Look up THE and QS ranking for an institution name.

        Returns
        -------
        dict with keys:
          matched_name    : canonical institution name matched, or null
          match_score     : fuzzy match score 0-100, or null
          the_rank        : int | "Not Ranked"
          the_rank_range  : str | "Not Ranked"
          qs_rank         : int | "Not Ranked"
          qs_rank_range   : str | "Not Ranked"
        """
        if not institution_name or not str(institution_name).strip():
            return self._not_ranked_result(None, None)

        name_clean = self._clean(institution_name)

        # 1. Try exact alias match first (fast path)
        exact = self._alias_exact_match(name_clean)
        if exact is not None:
            return self._row_to_result(self._df.iloc[exact], 100, institution_name)

        # 2. Try fuzzy match against all known names + aliases
        best_row, best_score = self._fuzzy_match(name_clean)
        if best_row is not None and best_score >= _MATCH_THRESHOLD:
            return self._row_to_result(best_row, best_score, institution_name)

        logger.debug("No ranking found for '%s' (best score %s)", institution_name, best_score)
        return self._not_ranked_result(institution_name, best_score)

    def enrich_education_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add ranking columns to a list of education row dicts.
        Adds: matched_institution, match_score, the_rank, the_rank_range,
              qs_rank, qs_rank_range.
        """
        result = []
        seen_lookups: Dict[str, Dict] = {}  # cache per institution

        for row in rows:
            enriched = dict(row)
            inst = str(row.get("institution") or "").strip()

            if inst not in seen_lookups:
                seen_lookups[inst] = self.get_ranking(inst)
            ranking = seen_lookups[inst]

            enriched.update({
                "matched_institution": ranking.get("matched_name"),
                "ranking_match_score": ranking.get("match_score"),
                "the_rank":            ranking.get("the_rank"),
                "the_rank_range":      ranking.get("the_rank_range"),
                "qs_rank":             ranking.get("qs_rank"),
                "qs_rank_range":       ranking.get("qs_rank_range"),
            })
            result.append(enriched)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_index(self) -> List[Tuple[str, List[str], int]]:
        """
        Build a list of (clean_canonical_name, [clean_alias, ...], row_index)
        for fast lookup.
        """
        index = []
        for i, row in self._df.iterrows():
            canonical = self._clean(row["institution_name"])
            aliases_raw = str(row.get("aliases", "") or "")
            aliases = [self._clean(a) for a in aliases_raw.split(";") if a.strip()]
            index.append((canonical, aliases, i))
        return index

    def _alias_exact_match(self, name_clean: str) -> Optional[int]:
        """Return row index if name_clean exactly matches any canonical name or alias."""
        for canonical, aliases, row_idx in self._index:
            if name_clean == canonical or name_clean in aliases:
                return row_idx
        return None

    def _fuzzy_match(self, name_clean: str) -> Tuple[Optional[Any], int]:
        """Return (best_matching_row, score) using rapidfuzz token_sort_ratio."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning(
                "rapidfuzz not installed; falling back to exact matching only. "
                "Run: pip install rapidfuzz"
            )
            return None, 0

        best_score = 0
        best_row   = None

        for canonical, aliases, row_idx in self._index:
            candidates = [canonical] + aliases
            for candidate in candidates:
                # Use token_sort_ratio so word order doesn't matter
                score = fuzz.token_sort_ratio(name_clean, candidate)
                if score > best_score:
                    best_score = score
                    best_row   = self._df.iloc[row_idx]

        return best_row, best_score

    def _row_to_result(self, row: Any, score: int, original: str) -> Dict[str, Any]:
        """Convert a DataFrame row to a ranking result dict."""
        def _rank(col: str):
            val = str(row.get(col, "") or "").strip()
            if not val or val.lower() in ("", "not ranked", "nan"):
                return NOT_RANKED
            try:
                return int(val)
            except ValueError:
                return val  # e.g. "401-500"

        return {
            "matched_name":   row["institution_name"],
            "match_score":    score,
            "the_rank":       _rank("the_rank"),
            "the_rank_range": _rank("the_rank_range") if _rank("the_rank") == NOT_RANKED else str(row.get("the_rank_range", "") or ""),
            "qs_rank":        _rank("qs_rank"),
            "qs_rank_range":  _rank("qs_rank_range") if _rank("qs_rank") == NOT_RANKED else str(row.get("qs_rank_range", "") or ""),
        }

    @staticmethod
    def _not_ranked_result(original: Optional[str], score: Optional[int]) -> Dict[str, Any]:
        return {
            "matched_name":   None,
            "match_score":    score,
            "the_rank":       NOT_RANKED,
            "the_rank_range": NOT_RANKED,
            "qs_rank":        NOT_RANKED,
            "qs_rank_range":  NOT_RANKED,
        }

    @staticmethod
    def _clean(s: str) -> str:
        """Lowercase, strip punctuation, collapse whitespace for comparison."""
        s = str(s).lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
