"""
research_profile.py  –  Module 3 Orchestrator
===============================================

WHY THIS FILE EXISTS
--------------------
This is the top-level controller for Part 3 (Research Profile Analysis).
It reads publications.csv and candidates.csv (from Module 1), runs tasks
3.1–3.3 for every candidate, and produces:

  data/analysis/journal_profiles.csv     – one row per journal paper
  data/analysis/conference_profiles.csv  – one row per conference paper
  data/analysis/research_aggregates.csv  – one row per candidate (summary stats)
  data/analysis/research_aggregates.xlsx – multi-sheet Excel workbook
  data/analysis/research_report.txt      – human-readable plain-text report

PIPELINE (in order)
-------------------
  JournalVerifier    → tasks 3.1.2-3.1.6  (Scopus/WoS/predatory/quality)
  ConferenceVerifier → tasks 3.2.2-3.2.6  (CORE/indexing/maturity/quality)
  Aggregate stats    → task  3.3           (counts, authorship breakdown)
  LLM summary        → generates research strength label + summary paragraph

DESIGN
------
- JournalVerifier and ConferenceVerifier share a caching layer so repeated
  runs (e.g. after adding a new candidate) only query the LLM for venues
  that haven't been seen before.
- Candidates with zero publications are still included in the aggregate CSV
  with all-zero counts.
- The module is completely independent of Modules 1 and 2 code — it only
  reads CSV files that Module 1 produced.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .journal_verifier    import JournalVerifier
from .conference_verifier import ConferenceVerifier

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# LLM prompts for research strength interpretation (Task 3.3 narrative)
# --------------------------------------------------------------------------

_RESEARCH_SYSTEM = (
    "You are an expert academic evaluator for a university recruitment system. "
    "You will receive structured research statistics for a candidate. "
    "Your task is to assess their research profile. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"research_strength\" and \"research_summary\". "
    "research_strength must be one of: \"Strong\", \"Moderate\", \"Weak\", \"Needs Clarification\". "
    "research_summary must be 2-4 sentences in third person, factual, no embellishment."
)

_RESEARCH_USER_TEMPLATE = (
    "Assess this candidate's research profile and return JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "PUBLICATIONS SUMMARY:\n"
    "  Journal papers         : {total_journals}\n"
    "  Conference papers      : {total_conferences}\n"
    "  Scopus-indexed journals: {scopus_indexed}\n"
    "  WoS-indexed journals   : {wos_indexed}\n"
    "  Q1 papers              : {q1}\n"
    "  Q2 papers              : {q2}\n"
    "  Q3 papers              : {q3}\n"
    "  Q4 papers              : {q4}\n"
    "  High-impact papers     : {high_impact}\n"
    "  Potential predatory    : {predatory}\n"
    "  CORE A* conferences    : {core_astar}\n"
    "  CORE A  conferences    : {core_a}\n"
    "  CORE B/C conferences   : {core_bc}\n"
    "  First-author papers    : {first_author}\n"
    "  Co-author papers       : {coauthor}\n\n"
    "SUGGESTED STRENGTH (rule-based): {rule_label}\n\n"
    "Return JSON:"
)


# --------------------------------------------------------------------------
# Publication-level data quality flag helper
# --------------------------------------------------------------------------

def _flag_publication_quality(row: Dict[str, Any], pub_type: str) -> str:
    """
    Task 3.1.7 / 3.2.7: Generate a pipe-separated string of data quality
    issues found in a single publication row.

    Checks performed:
      - Missing venue / venue is generic placeholder
      - Missing year
      - Missing authors
      - Missing DOI (informational, not blocking)
      - Venue could not be verified externally
      - Potential predatory flag
      - Conference maturity (edition < 3)
    """
    flags = []

    venue = str(row.get("venue_original") or "").strip()
    if not venue or venue.lower() in ("nan", "none", "", "journal", "international journal",
                                       "conference", "international conference"):
        flags.append("MISSING_VENUE")
    elif row.get("quality_label") == "Unverified Venue":
        flags.append("VENUE_UNVERIFIABLE")

    year = str(row.get("year") or "").strip()
    if not year or year.lower() in ("nan", "none", ""):
        flags.append("MISSING_YEAR")

    authors = str(row.get("authors") or "").strip()
    if not authors or authors.lower() in ("nan", "none", ""):
        flags.append("MISSING_AUTHORS")

    doi = str(row.get("doi") or "").strip()
    if not doi or doi.lower() in ("nan", "none", ""):
        flags.append("MISSING_DOI")

    if row.get("candidate_role") == "Unknown":
        flags.append("AUTHORSHIP_UNDETECTABLE")

    if str(row.get("predatory_suspected", "")).lower() == "true":
        flags.append("POTENTIAL_PREDATORY")

    # Conference-specific
    if pub_type == "conference":
        edition = row.get("edition_number")
        if edition is not None and int(edition) < 3:
            flags.append(f"IMMATURE_CONFERENCE_EDITION_{edition}")

    return " | ".join(flags) if flags else "OK"


def _rule_based_research_label(agg: Dict[str, Any]) -> str:
    """Deterministic research strength label — seeds the LLM."""
    total = (agg.get("total_journals", 0) or 0) + (agg.get("total_conferences", 0) or 0)
    high  = agg.get("high_impact", 0) or 0
    q1q2  = (agg.get("q1_papers", 0) or 0) + (agg.get("q2_papers", 0) or 0)
    wos   = agg.get("wos_indexed", 0) or 0
    pred  = agg.get("predatory_suspected", 0) or 0
    first = agg.get("first_author_count", 0) or 0

    if total == 0:
        return "Needs Clarification"
    if pred > 0 and pred >= total / 2:
        return "Weak"
    if q1q2 >= 3 and wos >= 2 and first >= 2:
        return "Strong"
    if q1q2 >= 1 or wos >= 1 or high >= 2:
        return "Moderate"
    if total >= 1:
        return "Weak"
    return "Needs Clarification"


# --------------------------------------------------------------------------
# Aggregate statistics helper
# --------------------------------------------------------------------------

def _compute_aggregate(
    candidate_id:     str,
    journal_results:  List[Dict],
    conf_results:     List[Dict],
) -> Dict[str, Any]:
    """Compute Task 3.3 aggregate statistics for one candidate."""

    # --- Journal statistics ---
    q1 = q2 = q3 = q4 = 0
    scopus_idx = wos_idx = high = moderate = low = unverified = predatory = 0
    first_auth_j = corr_auth_j = coauth_j = unknown_j = 0

    for r in journal_results:
        q = r.get("quartile")
        if q == "Q1":    q1 += 1
        elif q == "Q2":  q2 += 1
        elif q == "Q3":  q3 += 1
        elif q == "Q4":  q4 += 1

        if r.get("scopus_indexed"):  scopus_idx += 1
        if r.get("wos_indexed"):     wos_idx    += 1

        ql = r.get("quality_label", "")
        if ql == "High Impact":         high      += 1
        elif ql == "Moderate Impact":   moderate  += 1
        elif ql == "Low Impact":        low       += 1
        elif ql == "Unverified Venue":  unverified+= 1
        elif ql == "Potential Predatory": predatory += 1

        role = r.get("candidate_role", "")
        if "First" in role:                first_auth_j  += 1
        elif "Corresponding" in role:      corr_auth_j   += 1
        elif role == "Co-Author":          coauth_j      += 1
        else:                              unknown_j     += 1

    # --- Conference statistics ---
    core_astar = core_a = core_b = core_c = core_unranked = 0
    first_auth_c = corr_auth_c = coauth_c = unknown_c = 0

    for r in conf_results:
        cr = r.get("core_rank")
        if cr == "A*":           core_astar   += 1
        elif cr == "A":          core_a       += 1
        elif cr == "B":          core_b       += 1
        elif cr == "C":          core_c       += 1
        else:                    core_unranked+= 1

        role = r.get("candidate_role", "")
        if "First" in role:            first_auth_c += 1
        elif "Corresponding" in role:  corr_auth_c  += 1
        elif role == "Co-Author":      coauth_c     += 1
        else:                          unknown_c    += 1

    return {
        "candidate_id":            candidate_id,
        "total_journals":          len(journal_results),
        "total_conferences":       len(conf_results),
        "scopus_indexed":          scopus_idx,
        "wos_indexed":             wos_idx,
        "q1_papers":               q1,
        "q2_papers":               q2,
        "q3_papers":               q3,
        "q4_papers":               q4,
        "high_impact":             high,
        "moderate_impact":         moderate,
        "low_impact":              low,
        "unverified_venue":        unverified,
        "predatory_suspected":     predatory,
        "core_a_star":             core_astar,
        "core_a":                  core_a,
        "core_b":                  core_b,
        "core_c":                  core_c,
        "core_unranked":           core_unranked,
        "first_author_count":      first_auth_j + first_auth_c,
        "corresponding_author_count": corr_auth_j + corr_auth_c,
        "coauthor_count":          coauth_j + coauth_c,
        "unknown_role_count":      unknown_j + unknown_c,
        "research_strength":       None,   # filled after LLM call
        "research_summary":        None,   # filled after LLM call
    }


# --------------------------------------------------------------------------
# Main orchestrator
# --------------------------------------------------------------------------

class ResearchProfileAnalyser:
    """
    Orchestrates tasks 3.1-3.3 for every candidate in publications.csv.

    Usage
    -----
    analyser = ResearchProfileAnalyser(
        publications_csv = "data/extracted/publications.csv",
        candidates_csv   = "data/extracted/candidates.csv",
        output_dir       = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        publications_csv:   str = "data/extracted/publications.csv",
        candidates_csv:     str = "data/extracted/candidates.csv",
        output_dir:         str = "data/analysis",
        api_key:            Optional[str] = None,
        model:              str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:           Optional[str] = None,
        skip_llm:           bool = False,
        reconstruct_venues: bool = True,
    ):
        self.publications_csv = Path(publications_csv)
        self.candidates_csv   = Path(candidates_csv)
        self.output_dir       = Path(output_dir)
        self.skip_llm         = skip_llm
        self.api_key          = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model            = model
        self.base_url         = base_url or os.environ.get("OPENAI_BASE_URL")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        shared_kwargs = dict(
            api_key            = self.api_key,
            model              = self.model,
            base_url           = self.base_url,
            skip_llm           = skip_llm,
            reconstruct_venues = reconstruct_venues,
        )
        self.journal_verifier  = JournalVerifier(**shared_kwargs)
        self.conf_verifier     = ConferenceVerifier(**shared_kwargs)

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        logger.info("=" * 60)
        logger.info("  TALASH Module 3 - Research Profile Analysis")
        logger.info("  Publications CSV : %s", self.publications_csv)
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        pub_df  = self._load_csv(self.publications_csv, "publications")
        cand_df = self._load_csv(self.candidates_csv,   "candidates")

        if pub_df is None or pub_df.empty:
            logger.error("publications.csv is empty or missing. Run Module 1 first.")
            return {}

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv is empty or missing. Run Module 1 first.")
            return {}

        # Build candidate_id → name lookup
        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )

        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # ------------------------------------------------------------------
        # Checkpointing: load already-completed candidates from disk so a
        # re-run (after cancellation) skips work already done.
        # ------------------------------------------------------------------
        checkpoint_path = self.output_dir / "_research_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))

        all_journals:    List[Dict] = list(checkpoint.get("journals",    []))
        all_conferences: List[Dict] = list(checkpoint.get("conferences", []))
        all_aggregates:  List[Dict] = list(checkpoint.get("aggregates",  []))

        if completed_cids:
            logger.info(
                "Checkpoint found: %d candidate(s) already done, resuming from next.",
                len(completed_cids)
            )

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing: %s", i, len(all_cids), cid)
            cand_name = name_map.get(str(cid), str(cid))

            cand_pubs    = pub_df[pub_df["candidate_id"] == cid]
            journal_rows = (
                cand_pubs[cand_pubs["type"].str.lower().str.strip() == "journal"]
                .to_dict("records")
                if not cand_pubs.empty else []
            )
            conf_rows = (
                cand_pubs[cand_pubs["type"].str.lower().str.strip() == "conference"]
                .to_dict("records")
                if not cand_pubs.empty else []
            )

            j_results = self.journal_verifier.verify_journals(
                candidate_id=cid, journal_rows=journal_rows, candidate_name=cand_name
            )
            c_results = self.conf_verifier.verify_conferences(
                candidate_id=cid, conf_rows=conf_rows, candidate_name=cand_name
            )

            # Attach data quality flags to every paper row
            for r in j_results:
                r["data_quality_flags"] = _flag_publication_quality(r, "journal")
            for r in c_results:
                r["data_quality_flags"] = _flag_publication_quality(r, "conference")

            agg = _compute_aggregate(cid, j_results, c_results)

            if not self.skip_llm:
                interp = self._interpret_research_strength(cid, agg)
                # Small sleep between candidates to reduce Groq 429 rate-limit pressure.
                # When all venues are cached this is effectively free (no LLM call anyway).
                time.sleep(1)
            else:
                rule_lbl = _rule_based_research_label(agg)
                interp   = {
                    "research_strength": rule_lbl,
                    "research_summary":  self._rule_based_summary(cid, agg, rule_lbl),
                }

            agg["research_strength"] = interp.get("research_strength")
            agg["research_summary"]  = interp.get("research_summary")
            agg["rule_based_label"]  = interp.get("rule_based_label",
                                                   _rule_based_research_label(agg))

            all_journals.extend(j_results)
            all_conferences.extend(c_results)
            all_aggregates.append(agg)

            # Save checkpoint after every candidate
            completed_cids.add(cid)
            self._save_checkpoint(checkpoint_path, {
                "completed":   list(completed_cids),
                "journals":    all_journals,
                "conferences": all_conferences,
                "aggregates":  all_aggregates,
            })

        # Delete checkpoint on successful full completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_journals, all_conferences, all_aggregates)

    # ------------------------------------------------------------------
    # LLM research strength interpretation
    # ------------------------------------------------------------------

    def _interpret_research_strength(
        self, candidate_id: str, agg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call LLM to generate research_strength label and summary."""
        rule_lbl = _rule_based_research_label(agg)

        user_prompt = _RESEARCH_USER_TEMPLATE.format(
            candidate_id     = candidate_id,
            total_journals   = agg.get("total_journals", 0),
            total_conferences= agg.get("total_conferences", 0),
            scopus_indexed   = agg.get("scopus_indexed", 0),
            wos_indexed      = agg.get("wos_indexed", 0),
            q1               = agg.get("q1_papers", 0),
            q2               = agg.get("q2_papers", 0),
            q3               = agg.get("q3_papers", 0),
            q4               = agg.get("q4_papers", 0),
            high_impact      = agg.get("high_impact", 0),
            predatory        = agg.get("predatory_suspected", 0),
            core_astar       = agg.get("core_a_star", 0),
            core_a           = agg.get("core_a", 0),
            core_bc          = (agg.get("core_b", 0) or 0) + (agg.get("core_c", 0) or 0),
            first_author     = agg.get("first_author_count", 0),
            coauthor         = agg.get("coauthor_count", 0),
            rule_label       = rule_lbl,
        )
        try:
            raw    = self._call_llm(_RESEARCH_SYSTEM, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "research_strength" in parsed:
                parsed["rule_based_label"] = rule_lbl
                return parsed
        except Exception as e:
            logger.warning("LLM research interpretation failed for '%s': %s", candidate_id, e)

        return {
            "research_strength": rule_lbl,
            "research_summary":  self._rule_based_summary(candidate_id, agg, rule_lbl),
            "rule_based_label":  rule_lbl,
        }

    @staticmethod
    def _rule_based_summary(cid: str, agg: Dict, label: str) -> str:
        total = (agg.get("total_journals") or 0) + (agg.get("total_conferences") or 0)
        q1q2  = (agg.get("q1_papers") or 0) + (agg.get("q2_papers") or 0)
        wos   = agg.get("wos_indexed") or 0
        if total == 0:
            return f"The candidate has no publication records in the dataset."
        parts = [f"The candidate has {total} publication(s) in total "
                 f"({agg.get('total_journals',0)} journal, "
                 f"{agg.get('total_conferences',0)} conference)."]
        if q1q2:
            parts.append(f"{q1q2} paper(s) are in Q1/Q2 Scopus-ranked journals.")
        if wos:
            parts.append(f"{wos} paper(s) are indexed in Web of Science.")
        parts.append(f"Overall research profile is assessed as {label}.")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        journals:    List[Dict],
        conferences: List[Dict],
        aggregates:  List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        # Sort aggregates: Strong → Moderate → Weak → Needs Clarification
        strength_rank = {"Strong": 1, "Moderate": 2, "Weak": 3, "Needs Clarification": 4}
        aggregates = sorted(
            aggregates,
            key=lambda x: (
                strength_rank.get(x.get("research_strength"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Journal profiles CSV
        if journals:
            p = self.output_dir / "journal_profiles.csv"
            pd.DataFrame(journals).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(journals), p.name)
            paths["journals"] = p

        # 2. Conference profiles CSV
        if conferences:
            p = self.output_dir / "conference_profiles.csv"
            pd.DataFrame(conferences).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(conferences), p.name)
            paths["conferences"] = p

        # 3. Aggregates CSV
        if aggregates:
            p = self.output_dir / "research_aggregates.csv"
            pd.DataFrame(aggregates).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(aggregates), p.name)
            paths["aggregates"] = p

        # 4. Excel workbook (3 sheets)
        p = self.output_dir / "research_aggregates.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            if aggregates:
                pd.DataFrame(aggregates).to_excel(
                    writer, sheet_name="Research Aggregates", index=False
                )
            if journals:
                pd.DataFrame(journals).to_excel(
                    writer, sheet_name="Journal Papers", index=False
                )
            if conferences:
                pd.DataFrame(conferences).to_excel(
                    writer, sheet_name="Conference Papers", index=False
                )
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 5. Plain-text report
        self._write_text_report(aggregates, paths)

        return paths

    def _write_text_report(self, aggregates: List[Dict], paths: Dict):
        p = self.output_dir / "research_report.txt"

        strength_counts: Dict[str, int] = {}
        for agg in aggregates:
            s = agg.get("research_strength") or "Unknown"
            strength_counts[s] = strength_counts.get(s, 0) + 1

        lines = [
            "=" * 70,
            "  TALASH Module 3 - Research Profile Report",
            "=" * 70,
            f"  Candidates analysed : {len(aggregates)}",
            "",
            "  Research Strength Distribution:",
        ]
        for lbl, cnt in sorted(strength_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += ["", "-" * 70, "  Per-Candidate Research Summaries", "-" * 70]

        for agg in aggregates:
            cid      = agg["candidate_id"]
            strength = agg.get("research_strength", "Unknown")
            lines.append(f"\n  [{strength}] {cid}")
            lines.append(
                f"  Publications  : {agg.get('total_journals',0)} journal, "
                f"{agg.get('total_conferences',0)} conference"
            )
            lines.append(
                f"  Scopus/WoS    : {agg.get('scopus_indexed',0)} / "
                f"{agg.get('wos_indexed',0)}"
            )
            lines.append(
                f"  Quartiles     : Q1={agg.get('q1_papers',0)} "
                f"Q2={agg.get('q2_papers',0)} "
                f"Q3={agg.get('q3_papers',0)} "
                f"Q4={agg.get('q4_papers',0)}"
            )
            lines.append(
                f"  CORE ranks    : A*={agg.get('core_a_star',0)} "
                f"A={agg.get('core_a',0)} "
                f"B={agg.get('core_b',0)} "
                f"C={agg.get('core_c',0)} "
                f"Unranked={agg.get('core_unranked',0)}"
            )
            lines.append(
                f"  Authorship    : 1st={agg.get('first_author_count',0)} "
                f"Corr={agg.get('corresponding_author_count',0)} "
                f"Co={agg.get('coauthor_count',0)}"
            )
            if agg.get("predatory_suspected"):
                lines.append(
                    f"  *** WARNING: {agg['predatory_suspected']} "
                    f"potentially predatory publication(s) detected ***"
                )
            if agg.get("research_summary"):
                lines.append(f"  Summary: {agg['research_summary']}")

        lines.append("\n" + "=" * 70)
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report written -> %s", p.name)
        paths["report"] = p

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    @staticmethod
    def _save_checkpoint(path: Path, data: Dict):
        try:
            path.write_text(
                json.dumps(data, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Could not save checkpoint: %s", e)

    # ------------------------------------------------------------------
    # Helpers
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
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str, name: str) -> Optional[Dict]:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.error("Could not parse LLM JSON for '%s'. Raw:\n%s", name, raw[:400])
        return None

    @staticmethod
    def _load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
        if not path.exists():
            logger.warning("%s CSV not found: %s", label, path)
            return None
        try:
            df = pd.read_csv(path, dtype=str)
            logger.info("Loaded %d rows from %s", len(df), path.name)
            return df
        except Exception as e:
            logger.error("Failed to read %s: %s", path.name, e)
            return None
