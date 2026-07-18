"""
authorship_detector.py  –  Tasks 3.1.5 & 3.2.5
================================================

WHY THIS FILE EXISTS
--------------------
Both journal and conference analysis require detecting the candidate's role
in the author list: First Author, Corresponding Author, or Co-Author.

DESIGN DECISIONS
----------------
- Candidate name is fuzzy-matched against the author list using rapidfuzz.
  This handles minor spelling variations and initials (e.g. "M. Salman" vs
  "Muhammad Salman Qamar").
- Author strings may use comma or semicolon as separators.
- "Corresponding Author" is assigned if:
    a. The string "(corresponding)" / "*" / "†" appears near the name, OR
    b. The candidate is the LAST author in a multi-author paper (heuristic,
       common in CS/Engineering publications).
- If the candidate cannot be located in the author list, role is "Unknown".
- This is pure CPU logic — no LLM calls needed.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Minimum fuzzy-match score to accept a name match (0-100)
_NAME_MATCH_THRESHOLD = 72

# Markers that explicitly indicate corresponding authorship
_CORRESPONDING_MARKERS = re.compile(
    r"\*|†|corresponding\s+author|corr\.\s+author", re.IGNORECASE
)


class AuthorshipDetector:
    """
    Tasks 3.1.5 / 3.2.5: Detect a candidate's role in a publication's author list.

    Usage
    -----
    detector = AuthorshipDetector()
    role = detector.detect_role(
        candidate_name = "Muhammad Salman Qamar",
        authors_str    = "Muhammad Salman Qamar, Ihsan ul Haq, Muhammad Fahad Munir",
    )
    # role == "First Author"
    """

    def detect_role(
        self,
        candidate_name: str,
        authors_str: Optional[str],
    ) -> str:
        """
        Determine the candidate's authorship role.

        Parameters
        ----------
        candidate_name : str
            Full name of the candidate (from candidates.csv).
        authors_str : str | None
            Raw author list string from publications.csv.

        Returns
        -------
        str  –  "First Author" | "Corresponding Author (heuristic)" |
                "Co-Author" | "Unknown"
        """
        if not authors_str or str(authors_str).strip() in ("", "nan", "None"):
            return "Unknown"

        authors_raw = str(authors_str).strip()

        # Check for explicit corresponding-author markers near the whole string
        has_explicit_corr = bool(_CORRESPONDING_MARKERS.search(authors_raw))

        # Split author list — support comma and semicolon separators
        authors = self._split_authors(authors_raw)
        if not authors:
            return "Unknown"

        # Find which position the candidate occupies
        position = self._find_position(candidate_name, authors)

        if position is None:
            return "Unknown"

        n = len(authors)

        if has_explicit_corr and position == 0:
            return "Corresponding Author (explicit)"

        if position == 0:
            return "First Author"

        # Heuristic: last author in multi-author paper is often corresponding
        if n > 2 and position == n - 1 and not has_explicit_corr:
            return "Corresponding Author (heuristic)"

        return "Co-Author"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split_authors(authors_raw: str) -> List[str]:
        """Split author string on commas or semicolons, strip whitespace."""
        # Remove explicit markers first
        cleaned = _CORRESPONDING_MARKERS.sub("", authors_raw)
        # Split on ";" then flatten comma-separated if semicolons used
        if ";" in cleaned:
            parts = [a.strip() for a in cleaned.split(";") if a.strip()]
        else:
            parts = [a.strip() for a in cleaned.split(",") if a.strip()]
        return [p for p in parts if len(p) > 1]

    @staticmethod
    def _find_position(candidate_name: str, authors: List[str]) -> Optional[int]:
        """Return 0-based index of candidate in author list, or None."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning(
                "rapidfuzz not installed; falling back to exact name matching. "
                "Install with: pip install rapidfuzz"
            )
            cand_lower = candidate_name.strip().lower()
            for i, auth in enumerate(authors):
                if cand_lower in auth.lower() or auth.lower() in cand_lower:
                    return i
            return None

        cand_tokens = set(candidate_name.lower().split())
        best_score = 0
        best_idx: Optional[int] = None

        for i, auth in enumerate(authors):
            # Try full token sort ratio (handles word order differences)
            score = fuzz.token_sort_ratio(candidate_name.lower(), auth.lower())

            # Also try partial name token overlap (for abbreviated names like "M. Salman")
            auth_tokens = set(auth.lower().split())
            common = cand_tokens & auth_tokens
            token_overlap = (len(common) / max(len(cand_tokens), 1)) * 100

            effective_score = max(score, token_overlap)

            if effective_score > best_score:
                best_score = effective_score
                best_idx = i

        if best_score >= _NAME_MATCH_THRESHOLD:
            return best_idx

        logger.debug(
            "Candidate '%s' not found in authors (best score: %.1f)",
            candidate_name, best_score,
        )
        return None
