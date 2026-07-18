"""
conference_verifier.py  –  Tasks 3.2.2–3.2.6
=============================================

WHY THIS FILE EXISTS
--------------------
Task 3.2 requires verifying conference publications: CORE ranking, edition
maturity detection, proceedings indexing, and quality labeling.

DESIGN DECISIONS
----------------
- CORE portal (portal.core.edu.au) renders its tables via JavaScript, making
  direct HTTP scraping unreliable. We therefore use LLM batch lookup
  (same Groq model used by Module 2) as the primary CORE rank source.
  The LLM is trained on academic data that includes CORE rankings for major
  CS/Engineering conferences, making it reliable for known conferences.
- Venue reconstruction: truncated conference names (<= 35 chars) are first
  expanded using the paper title as context (same logic as JournalVerifier).
- Edition number detection uses regex to find ordinals (e.g. "13th", "3rd")
  in the venue name. Conferences with edition < 3 are flagged as immature.
- Proceedings indexing is detected by keyword matching (IEEE, Springer LNCS,
  ACM, Scopus, etc.) in the venue name — no external calls needed.
- Results are cached locally at data/research_cache/conference_cache.json.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

_CACHE_DIR  = Path(__file__).resolve().parent.parent / "data" / "research_cache"
_CACHE_FILE = _CACHE_DIR / "conference_cache.json"

# --------------------------------------------------------------------------
# LLM prompts
# --------------------------------------------------------------------------

_CONF_SYSTEM = (
    "You are an academic conference ranking expert with knowledge of the "
    "CORE conference ranking portal (portal.core.edu.au). "
    "Return ONLY a valid JSON object. No markdown, no explanation. "
    "Use null when you are not confident — do NOT guess ranks."
)

_CONF_USER_TEMPLATE = (
    "For each conference below, return a JSON object where keys are the exact "
    "conference names and values are objects with these fields:\n"
    "  core_rank: \"A*\" | \"A\" | \"B\" | \"C\" | \"Unranked\" | \"Cannot Verify\"\n"
    "  known_publisher: \"IEEE\" | \"ACM\" | \"Springer\" | \"Elsevier\" | \"USENIX\" | \"Other\" | null\n\n"
    "Rules:\n"
    "- Use 'Cannot Verify' when unsure. Never guess.\n"
    "- Use official CORE 2023 data where known.\n"
    "- Return ONLY the JSON object. No markdown.\n\n"
    "Conferences:\n{conf_list}"
)

_RECONSTRUCT_SYSTEM = (
    "You are a bibliographic assistant. Given a paper title and a truncated "
    "conference name, reconstruct the full conference name. "
    "Return ONLY a JSON object with key 'full_name'. "
    "If you cannot determine the full name with confidence, set full_name to null."
)

_RECONSTRUCT_USER_TEMPLATE = (
    "Paper title: {title}\n"
    "Truncated conference: {venue}\n"
    "What is the full conference name? Return JSON: {{\"full_name\": \"...\"}}"
)

# --------------------------------------------------------------------------
# Proceedings indexing keyword detection
# --------------------------------------------------------------------------

_INDEXING_PATTERNS: Dict[str, re.Pattern] = {
    "IEEE Xplore":          re.compile(r"\bIEEE\b", re.IGNORECASE),
    "ACM Digital Library":  re.compile(r"\bACM\b", re.IGNORECASE),
    "Springer LNCS":        re.compile(r"\bSpringer\b|\bLNCS\b|\bLecture Notes\b", re.IGNORECASE),
    "Elsevier":             re.compile(r"\bElsevier\b|\bScienceDirect\b", re.IGNORECASE),
    "USENIX":               re.compile(r"\bUSENIX\b", re.IGNORECASE),
    "ACL Anthology":        re.compile(r"\bACL\b|\bEMNLP\b|\bNAACL\b|\bEACL\b|\bCoNLL\b", re.IGNORECASE),
    "Scopus":               re.compile(r"\bScopus\b", re.IGNORECASE),
}

# Ordinal regex (e.g. "1st", "2nd", "13th", "21st")
_ORDINAL_RE = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE
)

# --------------------------------------------------------------------------
# Deterministic quality label
# --------------------------------------------------------------------------

def _classify_conference_quality(
    core_rank: Optional[str],
    indexed_in: List[str],
) -> str:
    """Deterministic quality label from CORE rank and indexing."""
    if core_rank == "A*":
        return "Top-Tier (A*)"
    if core_rank == "A":
        return "High (A)"
    if core_rank in ("B", "C"):
        return f"Moderate ({core_rank})"
    if indexed_in:
        return "Low / Unranked (Indexed)"
    return "Low / Unranked"


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class ConferenceVerifier:
    """
    Tasks 3.2.2–3.2.6: Verify conference publications.

    Usage
    -----
    verifier = ConferenceVerifier(api_key=..., model=..., base_url=...)
    results = verifier.verify_conferences(candidate_id, conf_rows, candidate_name)
    """

    def __init__(
        self,
        api_key:            Optional[str] = None,
        model:              str           = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:           Optional[str] = None,
        temperature:        float         = 0.0,
        reconstruct_venues: bool          = True,
        skip_llm:           bool          = False,
    ):
        self.api_key            = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model              = model
        self.base_url           = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature        = temperature
        self.reconstruct_venues = reconstruct_venues
        self.skip_llm           = skip_llm

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = self._load_cache()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def verify_conferences(
        self,
        candidate_id:   str,
        conf_rows:      List[Dict[str, Any]],
        candidate_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Verify all conference papers for one candidate.

        Returns
        -------
        List of enriched dicts, one per conference paper.
        """
        from .authorship_detector import AuthorshipDetector
        detector = AuthorshipDetector()

        if not conf_rows:
            return []

        # Step 1: Reconstruct truncated venue names
        rows_with_venue = self._reconstruct_all_venues(conf_rows)

        # Step 2: Batch LLM lookup for CORE ranks
        unique_venues = list({
            r["venue_resolved"]
            for r in rows_with_venue
            if r.get("venue_resolved")
        })
        venue_data = self._batch_lookup_conferences(unique_venues)

        # Step 3: Assemble results
        results = []
        for row in rows_with_venue:
            venue     = row.get("venue_resolved") or row.get("venue") or ""
            info      = venue_data.get(venue, {})
            core_rank = info.get("core_rank")

            indexed_in     = self._detect_proceedings_indexing(venue)
            edition_number = self._detect_edition_number(venue)
            maturity_flag  = (edition_number is not None and edition_number < 3)

            quality = _classify_conference_quality(core_rank, indexed_in)
            role    = detector.detect_role(candidate_name, row.get("authors"))

            results.append({
                "candidate_id":        candidate_id,
                "title":               row.get("title"),
                "venue_original":      row.get("venue"),
                "venue_resolved":      venue,
                "venue_reconstructed": row.get("venue_reconstructed", False),
                "year":                row.get("year"),
                "authors":             row.get("authors"),
                "doi":                 row.get("doi"),
                "core_rank":           core_rank,
                "edition_number":      edition_number,
                "maturity_flag":       maturity_flag,
                "indexed_in":          ", ".join(indexed_in) if indexed_in else None,
                "candidate_role":      role,
                "quality_label":       quality,
            })

        self._save_cache()
        return results

    # ------------------------------------------------------------------
    # Venue reconstruction (shared logic with JournalVerifier)
    # ------------------------------------------------------------------

    def _reconstruct_all_venues(
        self, rows: List[Dict]
    ) -> List[Dict]:
        output = []
        for row in rows:
            venue     = str(row.get("venue") or "").strip()
            resolved  = venue
            was_recon = False

            if self.reconstruct_venues and self._is_truncated(venue):
                cache_key = f"reconstruct_conf::{venue}"
                if cache_key in self._cache:
                    resolved  = self._cache[cache_key] or venue
                    was_recon = resolved != venue
                elif not self.skip_llm:
                    full = self._reconstruct_venue(
                        title=str(row.get("title") or ""),
                        venue=venue,
                    )
                    self._cache[cache_key] = full
                    if full and full != venue:
                        resolved  = full
                        was_recon = True

            new_row = dict(row)
            new_row["venue_resolved"]      = resolved
            new_row["venue_reconstructed"] = was_recon
            output.append(new_row)
        return output

    _TRUNCATION_ENDWORDS = frozenset({
        "on", "in", "of", "the", "a", "an", "and", "or", "for", "to",
        "at", "by", "from", "with", "de", "du", "des", "en", "el",
    })

    @classmethod
    def _is_truncated(cls, venue: str) -> bool:
        v = venue.strip()
        if not v or len(v) > 35:
            return False
        if " " not in v:
            return False
        last_word = v.rsplit(" ", 1)[-1].lower()
        if last_word in cls._TRUNCATION_ENDWORDS:
            return True
        if re.fullmatch(r"[a-zA-Z]{1,3}", last_word):
            return True
        return False

    def _reconstruct_venue(self, title: str, venue: str) -> Optional[str]:
        prompt = _RECONSTRUCT_USER_TEMPLATE.format(title=title[:300], venue=venue)
        try:
            raw    = self._call_llm(_RECONSTRUCT_SYSTEM, prompt)
            parsed = self._parse_json(raw, f"reconstruct_conf:{venue}")
            if parsed and parsed.get("full_name"):
                return str(parsed["full_name"]).strip()
        except Exception as e:
            logger.debug("Conference venue reconstruction failed for '%s': %s", venue, e)
        return None

    # ------------------------------------------------------------------
    # Batch conference LLM lookup
    # ------------------------------------------------------------------

    def _batch_lookup_conferences(
        self, venues: List[str], batch_size: int = 8
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        uncached = []

        for v in venues:
            if not v:
                continue
            cache_key = f"conf::{v.lower().strip()}"
            if cache_key in self._cache:
                result[v] = self._cache[cache_key]
            else:
                uncached.append(v)

        if not uncached or self.skip_llm:
            for v in uncached:
                result[v] = {}
            return result

        logger.info(
            "  [3.2] Looking up %d conference venue(s) via LLM (batch size=%d)...",
            len(uncached), batch_size
        )

        for i in range(0, len(uncached), batch_size):
            batch        = uncached[i : i + batch_size]
            batch_result = self._llm_lookup_batch(batch)
            for v in batch:
                info = batch_result.get(v) or {}
                result[v] = info
                self._cache[f"conf::{v.lower().strip()}"] = info
            if i + batch_size < len(uncached):
                time.sleep(0.5)

        return result

    def _llm_lookup_batch(self, venues: List[str]) -> Dict[str, Any]:
        conf_list = "\n".join(f"- {v}" for v in venues)
        prompt = _CONF_USER_TEMPLATE.format(conf_list=conf_list)
        try:
            raw    = self._call_llm(_CONF_SYSTEM, prompt)
            parsed = self._parse_json(raw, "conference_batch")
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.warning("Conference batch LLM lookup failed: %s", e)
        return {}

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_proceedings_indexing(venue: str) -> List[str]:
        """Return list of known indexing sources detected from venue name."""
        found = []
        for source, pattern in _INDEXING_PATTERNS.items():
            if pattern.search(venue):
                found.append(source)
        return found

    @staticmethod
    def _detect_edition_number(venue: str) -> Optional[int]:
        """
        Extract the edition ordinal from a conference name.
        E.g. "13th IEEE International Conference" → 13
        Returns None if no ordinal found.
        """
        m = _ORDINAL_RE.search(venue)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Dict[str, Any]:
        if _CACHE_FILE.exists():
            try:
                return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            _CACHE_FILE.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not save conference cache: %s", e)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str, name: str) -> Optional[Any]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        logger.error(
            "Could not parse LLM JSON for '%s'. Raw:\n%s", name, raw[:400]
        )
        return None
