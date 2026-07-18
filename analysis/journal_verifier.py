"""
journal_verifier.py  –  Tasks 3.1.2–3.1.6
==========================================

WHY THIS FILE EXISTS
--------------------
Task 3.1 requires verifying journal publications against external databases.
Since paid APIs (Clarivate WoS, Elsevier Scopus) require institutional
subscriptions, this module uses:

  1. LLM batch lookup (Groq/OpenAI) – primary source for Scopus indexing,
     quartile (Q1-Q4), WoS indexing, and estimated impact factor.
     Results are stored verbatim with "Cannot Verify" for unknowns.
  2. Local predatory journal list (Beall's list) – pure CSV lookup,
     zero API calls needed.
  3. Local cache (data/research_cache/journal_cache.json) – prevents
     duplicate API calls across runs.

DESIGN DECISIONS
----------------
- Journals are looked up in batches of up to 10 per LLM call to save tokens.
- The LLM is instructed to return "Cannot Verify" (not guess) for obscure
  or recently created journals it has no reliable data on.
- Venue names that appear truncated (<= 35 chars and ends mid-word) are first
  reconstructed by a separate LLM call using the paper title as context.
- Quality labels are assigned by a deterministic rule engine AFTER indexing
  data is retrieved, so no LLM subjectivity enters the final label.
- All original venue names are preserved alongside reconstructed ones.
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
# Predatory journal detection
# --------------------------------------------------------------------------

_BEALL_CSV = (
    Path(__file__).resolve().parent.parent
    / "data" / "rankings" / "beall_list_journals.csv"
)

# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

_CACHE_DIR  = Path(__file__).resolve().parent.parent / "data" / "research_cache"
_CACHE_FILE = _CACHE_DIR / "journal_cache.json"

# --------------------------------------------------------------------------
# LLM prompts
# --------------------------------------------------------------------------

_JOURNAL_SYSTEM = (
    "You are a bibliometric expert with deep knowledge of academic journal "
    "databases including Web of Science (WoS), Scopus, and SCImago. "
    "You MUST return ONLY a valid JSON object. No markdown, no explanation. "
    "If you are not confident about a field, use null — do NOT guess."
)

_JOURNAL_USER_TEMPLATE = (
    "For each journal below, return a JSON object where keys are the exact "
    "journal names and values are objects with these fields:\n"
    "  scopus_indexed: true | false | null (null = cannot verify)\n"
    "  quartile: \"Q1\" | \"Q2\" | \"Q3\" | \"Q4\" | \"Not Ranked\" | \"Cannot Verify\"\n"
    "  wos_indexed: true | false | null\n"
    "  impact_factor: float | null\n"
    "  predatory_suspected: true | false\n\n"
    "Rules:\n"
    "- Use null or 'Cannot Verify' when unsure. NEVER guess.\n"
    "- Predatory = listed in Beall's List or known fake/misleading venues.\n"
    "- Return ONLY the JSON object. No markdown, no explanation.\n\n"
    "Journals:\n{journals_list}"
)

_RECONSTRUCT_SYSTEM = (
    "You are a bibliographic assistant. Given a paper title and a truncated "
    "journal or conference name, reconstruct the full venue name. "
    "Return ONLY a JSON object with key 'full_name'. "
    "If you cannot determine the full name with confidence, set full_name to null."
)

_RECONSTRUCT_USER_TEMPLATE = (
    "Paper title: {title}\n"
    "Truncated venue: {venue}\n"
    "What is the full journal/conference name? Return JSON: {{\"full_name\": \"...\"}}"
)


# --------------------------------------------------------------------------
# Deterministic quality label rules
# --------------------------------------------------------------------------

def _classify_journal_quality(
    scopus: Optional[bool],
    quartile: Optional[str],
    wos: Optional[bool],
    predatory: bool,
) -> str:
    """
    Assign a quality label based on indexing and quartile data.
    Pure deterministic — no LLM involved.
    """
    if predatory:
        return "Potential Predatory"
    if quartile in ("Q1", "Q2") and wos:
        return "High Impact"
    if quartile in ("Q1", "Q2"):
        return "High Impact"
    if quartile in ("Q3", "Q4") or wos:
        return "Moderate Impact"
    if scopus:
        return "Low Impact"
    if scopus is None and wos is None and quartile in (None, "Cannot Verify"):
        return "Unverified Venue"
    return "Low Impact"


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------

class JournalVerifier:
    """
    Tasks 3.1.2–3.1.6: Verify journal publications.

    Usage
    -----
    verifier = JournalVerifier(api_key=..., model=..., base_url=...)
    results = verifier.verify_journals(candidate_id, journal_rows, candidate_name)
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

        # Load predatory list
        self._predatory_names: List[str] = []
        self._predatory_issns: List[str] = []
        self._load_predatory_list()

        # Load / init cache
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Any] = self._load_cache()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def verify_journals(
        self,
        candidate_id:   str,
        journal_rows:   List[Dict[str, Any]],
        candidate_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Verify all journal papers for one candidate.

        Parameters
        ----------
        candidate_id   : str
        journal_rows   : list of dicts (rows from publications.csv filtered to type=journal)
        candidate_name : str (full name from candidates.csv for authorship detection)

        Returns
        -------
        List of enriched dicts, one per journal paper.
        """
        from .authorship_detector import AuthorshipDetector
        detector = AuthorshipDetector()

        if not journal_rows:
            return []

        # Step 1: Reconstruct truncated venue names
        rows_with_venue = self._reconstruct_all_venues(journal_rows)

        # Step 2: Batch LLM lookup for all unique venue names
        unique_venues = list({
            r["venue_resolved"]
            for r in rows_with_venue
            if r.get("venue_resolved")
        })
        venue_data = self._batch_lookup_journals(unique_venues)

        # Step 3: Predatory check + quality label + authorship
        results = []
        for row in rows_with_venue:
            venue = row.get("venue_resolved") or row.get("venue") or ""
            info  = venue_data.get(venue, {})

            scopus    = info.get("scopus_indexed")
            quartile  = info.get("quartile")
            wos       = info.get("wos_indexed")
            impact_f  = info.get("impact_factor")
            predatory = self._is_predatory(venue, row.get("issn"))

            quality = _classify_journal_quality(scopus, quartile, wos, predatory)
            role    = detector.detect_role(candidate_name, row.get("authors"))

            results.append({
                "candidate_id":       candidate_id,
                "title":              row.get("title"),
                "venue_original":     row.get("venue"),
                "venue_resolved":     venue,
                "venue_reconstructed": row.get("venue_reconstructed", False),
                "issn":               row.get("issn"),
                "year":               row.get("year"),
                "authors":            row.get("authors"),
                "doi":                row.get("doi"),
                "scopus_indexed":     scopus,
                "quartile":           quartile,
                "wos_indexed":        wos,
                "impact_factor":      impact_f,
                "predatory_suspected": predatory,
                "candidate_role":     role,
                "quality_label":      quality,
            })

        self._save_cache()
        return results

    # ------------------------------------------------------------------
    # Venue reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_all_venues(
        self, rows: List[Dict]
    ) -> List[Dict]:
        """
        For each row, check if venue is truncated.
        If so (and reconstruct_venues=True), call LLM to recover full name.
        Adds 'venue_resolved' and 'venue_reconstructed' keys to each row.
        """
        output = []
        for row in rows:
            venue = str(row.get("venue") or "").strip()
            resolved = venue
            was_reconstructed = False

            if self.reconstruct_venues and self._is_truncated(venue):
                cache_key = f"reconstruct::{venue}"
                if cache_key in self._cache:
                    resolved = self._cache[cache_key] or venue
                    was_reconstructed = resolved != venue
                elif not self.skip_llm:
                    full = self._reconstruct_venue(
                        title=str(row.get("title") or ""),
                        venue=venue,
                    )
                    self._cache[cache_key] = full
                    if full and full != venue:
                        resolved = full
                        was_reconstructed = True

            new_row = dict(row)
            new_row["venue_resolved"]      = resolved
            new_row["venue_reconstructed"] = was_reconstructed
            output.append(new_row)
        return output

    # Prepositions and articles that signal a phrase was cut mid-stream
    _TRUNCATION_ENDWORDS = frozenset({
        "on", "in", "of", "the", "a", "an", "and", "or", "for", "to",
        "at", "by", "from", "with", "de", "du", "des", "en", "el",
    })

    @classmethod
    def _is_truncated(cls, venue: str) -> bool:
        """
        Heuristic: a venue name is likely truncated if EITHER:
          (a) It is <= 35 chars, multi-word, and its last word is a known
              preposition/article (e.g. 'International Conference on'),
              suggesting the phrase was cut in the middle of a word, OR
          (b) It is <= 35 chars, multi-word, and the final word is a partial
              fragment (mixed-case alpha, length 2-4 chars, not a full English word
              like 'Access', 'Systems', 'Letters').

        This prevents flagging short-but-complete names like:
          'IEEE Access'             -> NOT truncated
          'Electronics'             -> NOT truncated (no space)
          'Applied Sciences'        -> NOT truncated (ends on a complete noun)

        And correctly flags:
          'International Conference on En'  -> truncated
          'International Conference on'     -> truncated (ends on preposition)
          '2nd International Conference o'  -> truncated
        """
        v = venue.strip()
        if not v or len(v) > 35:
            return False
        # Must be multi-word
        if " " not in v:
            return False

        last_word = v.rsplit(" ", 1)[-1].lower()

        # Case (a): last word is a dangling preposition / article
        if last_word in cls._TRUNCATION_ENDWORDS:
            return True

        # Case (b): last word is 1-3 chars of purely alpha (fragment)
        if re.fullmatch(r"[a-zA-Z]{1,3}", last_word):
            return True

        return False

    def _reconstruct_venue(self, title: str, venue: str) -> Optional[str]:
        """Call LLM to reconstruct a truncated venue name."""
        prompt = _RECONSTRUCT_USER_TEMPLATE.format(title=title[:300], venue=venue)
        try:
            raw = self._call_llm(_RECONSTRUCT_SYSTEM, prompt)
            parsed = self._parse_json(raw, f"reconstruct:{venue}")
            if parsed and parsed.get("full_name"):
                return str(parsed["full_name"]).strip()
        except Exception as e:
            logger.debug("Venue reconstruction failed for '%s': %s", venue, e)
        return None

    # ------------------------------------------------------------------
    # Batch journal LLM lookup
    # ------------------------------------------------------------------

    def _batch_lookup_journals(
        self, venues: List[str], batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Look up all unique venues via LLM in batches of batch_size.
        Hits cache first; only sends uncached venues to LLM.
        """
        result: Dict[str, Any] = {}

        uncached = []
        for v in venues:
            if not v:
                continue
            cache_key = f"journal::{v.lower().strip()}"
            if cache_key in self._cache:
                result[v] = self._cache[cache_key]
            else:
                uncached.append(v)

        if not uncached or self.skip_llm:
            # Fill missing with empty dicts
            for v in uncached:
                result[v] = {}
            return result

        logger.info(
            "  [3.1] Looking up %d journal venue(s) via LLM (batch size=%d)...",
            len(uncached), batch_size
        )

        # Process in batches
        for i in range(0, len(uncached), batch_size):
            batch = uncached[i : i + batch_size]
            batch_result = self._llm_lookup_batch(batch)
            for v in batch:
                info = batch_result.get(v) or {}
                result[v] = info
                self._cache[f"journal::{v.lower().strip()}"] = info
            # Be polite to the API
            if i + batch_size < len(uncached):
                time.sleep(0.5)

        return result

    def _llm_lookup_batch(self, venues: List[str]) -> Dict[str, Any]:
        """Send one LLM call to look up a batch of journal names."""
        journals_list = "\n".join(f"- {v}" for v in venues)
        prompt = _JOURNAL_USER_TEMPLATE.format(journals_list=journals_list)
        try:
            raw = self._call_llm(_JOURNAL_SYSTEM, prompt)
            parsed = self._parse_json(raw, "journal_batch")
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.warning("Journal batch LLM lookup failed: %s", e)
        return {}

    # ------------------------------------------------------------------
    # Predatory detection
    # ------------------------------------------------------------------

    def _load_predatory_list(self):
        """Load Beall's List from local CSV if it exists."""
        if not _BEALL_CSV.exists():
            logger.debug("Beall's List CSV not found at %s — predatory check disabled.", _BEALL_CSV)
            return
        try:
            import pandas as pd
            df = pd.read_csv(_BEALL_CSV, dtype=str).fillna("")
            self._predatory_names = [
                n.lower().strip() for n in df.get("journal_name", pd.Series()).tolist()
            ]
            self._predatory_issns = [
                i.lower().strip() for i in df.get("issn", pd.Series()).tolist()
                if i.strip()
            ]
            logger.info(
                "Loaded %d predatory journal names from Beall's List.",
                len(self._predatory_names)
            )
        except Exception as e:
            logger.warning("Could not load Beall's List: %s", e)

    def _is_predatory(self, venue: str, issn: Optional[str]) -> bool:
        """Return True if venue or ISSN matches the predatory list."""
        if not self._predatory_names:
            return False
        try:
            from rapidfuzz import fuzz
            v_lower = venue.lower().strip()
            for pn in self._predatory_names:
                if fuzz.token_sort_ratio(v_lower, pn) >= 90:
                    return True
        except ImportError:
            v_lower = venue.lower().strip()
            for pn in self._predatory_names:
                if pn and pn in v_lower:
                    return True
        if issn:
            issn_clean = str(issn).lower().strip().replace("-", "")
            for pi in self._predatory_issns:
                if pi and pi.replace("-", "") == issn_clean:
                    return True
        return False

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
            logger.warning("Could not save journal cache: %s", e)

    # ------------------------------------------------------------------
    # LLM helpers (mirrors edu_interpreter.py pattern)
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
        """Parse JSON from LLM response with fallback extraction."""
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Strip markdown fences
        m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Find first {...} or [...]
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        logger.error("Could not parse LLM JSON for '%s'. Raw:\n%s", name, raw[:400])
        return None
