"""
collaboration_analyser.py  –  Module 8: Co-Author Collaboration Analysis Orchestrator
===================================================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates Part 8 (Co-Author Collaboration Analysis):
  - Reads publications.csv, supervision.csv, and candidates.csv (from Module 1)
  - Extracts and normalizes co-author names per publication
  - Detects recurring vs. one-time collaborators and top frequent co-authors
  - Calculates average team size (authors per paper) and profile
  - Cross-references co-authors against supervised students (Module 4 data)
  - Computes collaboration diversity score and network strength label
  - Calls LLM to generate collaboration profile summaries (with rule-based fallback)
  - Exports collaboration_profiles.csv, coauthor_network.csv, collaboration_profiles.xlsx, and collaboration_report.txt
"""

import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .collaboration_verifier import (
    normalize_author_name,
    parse_coauthors,
    classify_team_size_profile,
    match_student_coauthors,
    compute_collaboration_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts for Collaboration Profile Interpretation
# ---------------------------------------------------------------------------

_COLLAB_SYSTEM_PROMPT = (
    "You are an expert academic research evaluator for a university recruitment system. "
    "You will receive structured co-authorship and collaboration statistics for a faculty candidate. "
    "Your task is to write a concise, factual assessment of their research collaboration network. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"collaboration_strength_label\" and \"summary\". "
    "collaboration_strength_label must be one of: \"Broad Network\", \"Balanced Network\", \"Closed Network\", \"Solo Researcher\", \"No Publications\". "
    "summary must be 2-4 sentences in third person, factual, with no embellishment."
)

_COLLAB_USER_TEMPLATE = (
    "Assess this candidate's research collaboration network and return JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "COLLABORATION STATISTICS:\n"
    "  Total publications           : {total_pubs}\n"
    "  Total unique co-authors      : {total_unique_coauthors}\n"
    "  Recurring collaborators      : {recurring_collaborators_count}\n"
    "  One-time collaborators       : {one_time_collaborators}\n"
    "  Top collaborators            : {top_collaborators_json}\n"
    "  Average authors per paper    : {avg_authors_per_paper}\n"
    "  Student co-author matches    : {student_collaborations}\n"
    "  Collaboration diversity index: {collaboration_diversity_score} (ratio of 1-time coauthors)\n"
    "  Team size profile            : {team_size_profile}\n"
    "  International collaboration  : {international_collaboration}\n\n"
    "SUGGESTED NETWORK LABEL (rule-based): {rule_label}\n\n"
    "Return JSON:"
)


class CollaborationAnalyser:
    """
    Orchestrates tasks 8.1–8.7 for every candidate in publications.csv, supervision.csv, and candidates.csv.

    Usage
    -----
    analyser = CollaborationAnalyser(
        publications_csv = "data/extracted/publications.csv",
        supervision_csv  = "data/extracted/supervision.csv",
        candidates_csv   = "data/extracted/candidates.csv",
        output_dir       = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        publications_csv: str = "data/extracted/publications.csv",
        supervision_csv:  str = "data/extracted/supervision.csv",
        candidates_csv:   str = "data/extracted/candidates.csv",
        output_dir:       str = "data/analysis",
        api_key:          Optional[str] = None,
        model:            str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:         Optional[str] = None,
        skip_llm:         bool = False,
    ):
        self.publications_csv = Path(publications_csv)
        self.supervision_csv  = Path(supervision_csv)
        self.candidates_csv   = Path(candidates_csv)
        self.output_dir       = Path(output_dir)
        self.skip_llm         = skip_llm
        self.api_key          = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model            = model
        self.base_url         = base_url or os.environ.get("OPENAI_BASE_URL")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        logger.info("=" * 60)
        logger.info("  TALASH Module 8 – Co-Author Collaboration Analysis")
        logger.info("  Publications CSV : %s", self.publications_csv)
        logger.info("  Supervision CSV  : %s", self.supervision_csv)
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        pub_df  = self._load_csv(self.publications_csv, "publications")
        sup_df  = self._load_csv(self.supervision_csv,  "supervision")
        cand_df = self._load_csv(self.candidates_csv,   "candidates")

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv missing or empty. Run Module 1 first.")
            return {}

        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )
        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # Group publications by candidate_id
        pub_by_cid: Dict[str, List[Dict]] = {}
        if pub_df is not None and not pub_df.empty:
            for _, row in pub_df.iterrows():
                cid = str(row.get("candidate_id", "")).strip()
                if cid:
                    pub_by_cid.setdefault(cid, []).append(row.to_dict())

        # Group student names by candidate_id
        students_by_cid: Dict[str, List[str]] = {}
        if sup_df is not None and not sup_df.empty:
            for _, row in sup_df.iterrows():
                cid  = str(row.get("candidate_id", "")).strip()
                sname = str(row.get("student_name", "")).strip()
                if cid and sname and sname.lower() not in ("nan", "none", ""):
                    students_by_cid.setdefault(cid, []).append(sname)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        checkpoint_path = self.output_dir / "_collaboration_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))

        all_collab_profiles: List[Dict] = list(checkpoint.get("collab_profiles", []))
        all_network_rows:    List[Dict] = list(checkpoint.get("network_rows",    []))

        if completed_cids:
            logger.info(
                "Checkpoint found: %d candidate(s) already done, resuming.",
                len(completed_cids)
            )

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing collaboration network for: %s", i, len(all_cids), cid)
            cand_name   = name_map.get(str(cid), str(cid))
            cand_pubs   = pub_by_cid.get(str(cid), [])
            cand_students = students_by_cid.get(str(cid), [])

            profile, network_rows = self._analyse_candidate(
                cid, cand_name, cand_pubs, cand_students
            )

            all_collab_profiles.append(profile)
            all_network_rows.extend(network_rows)

            completed_cids.add(str(cid))
            self._save_checkpoint(checkpoint_path, {
                "completed":       list(completed_cids),
                "collab_profiles": all_collab_profiles,
                "network_rows":    all_network_rows,
            })

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_collab_profiles, all_network_rows)

    # ------------------------------------------------------------------
    # Per-Candidate Analysis
    # ------------------------------------------------------------------

    def _analyse_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        cand_pubs: List[Dict[str, Any]],
        cand_students: List[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

        valid_pubs = [
            p for p in cand_pubs
            if str(p.get("title") or "").strip().lower() not in ("nan", "none", "")
        ]
        total_pubs = len(valid_pubs)

        if total_pubs == 0:
            empty_profile = {
                "candidate_id":                   candidate_id,
                "candidate_name":                 candidate_name,
                "total_publications":             0,
                "total_unique_coauthors":         0,
                "recurring_collaborators_count":  0,
                "one_time_collaborators":         0,
                "avg_authors_per_paper":          0.0,
                "student_collaborations":         0,
                "collaboration_diversity_score":  0.0,
                "team_size_profile":              "No Publications",
                "international_collaboration":    "Affiliation data unavailable",
                "collaboration_strength_label":   "No Publications",
                "top_collaborators_json":         "[]",
                "summary":                        f"Candidate {candidate_id} has no publications listed in their CV.",
                "rule_based_label":               "No Publications",
            }
            return empty_profile, []

        coauthor_counts: Dict[str, int] = {}
        total_authors_sum = 0
        all_coauthors_list: List[str] = []

        # Tasks 8.1 - 8.3
        for p in valid_pubs:
            authors_str = str(p.get("authors") or "").strip()
            coauthors, paper_author_cnt = parse_coauthors(authors_str, candidate_name)
            total_authors_sum += paper_author_cnt
            all_coauthors_list.extend(coauthors)

            for ca in set(coauthors):
                coauthor_counts[ca] = coauthor_counts.get(ca, 0) + 1

        # Task 8.4: Match student co-authors
        matched_students = match_student_coauthors(all_coauthors_list, cand_students)
        student_matches_count = len(matched_students)

        # Task 8.5: Compute collaboration metrics
        metrics = compute_collaboration_metrics(
            coauthor_counts, total_pubs, total_authors_sum, student_matches_count
        )

        top_collabs = metrics["top_collaborators"]
        top_collabs_json = json.dumps(top_collabs, ensure_ascii=False)
        intl_collab = "Affiliation data unavailable"
        rule_label  = metrics["collaboration_strength_label"]

        # Build network rows
        network_rows = []
        for name, count in sorted(coauthor_counts.items(), key=lambda x: x[1], reverse=True):
            is_student = any(fuzz.token_sort_ratio(name.lower(), s.lower()) >= 80 for s in matched_students)
            network_rows.append({
                "candidate_id":     candidate_id,
                "coauthor_name":    name,
                "paper_count":      count,
                "is_recurring":     count >= 2,
                "is_student_match": is_student,
            })

        # Task 8.7: LLM Interpretation
        if self.skip_llm:
            strength_label = rule_label
            summary = self._rule_based_summary(
                candidate_name, total_pubs, metrics["total_unique_coauthors"],
                metrics["recurring_collaborators_count"], top_collabs,
                metrics["avg_authors_per_paper"], metrics["team_size_profile"],
                rule_label
            )
        else:
            interp = self._interpret_collaboration_profile(
                candidate_id, total_pubs, metrics["total_unique_coauthors"],
                metrics["recurring_collaborators_count"], metrics["one_time_collaborators"],
                top_collabs_json, metrics["avg_authors_per_paper"], student_matches_count,
                metrics["collaboration_diversity_score"], metrics["team_size_profile"],
                intl_collab, rule_label
            )
            strength_label = interp.get("collaboration_strength_label", rule_label)
            summary = interp.get("summary", "")
            time.sleep(0.5)

        collab_profile = {
            "candidate_id":                   candidate_id,
            "candidate_name":                 candidate_name,
            "total_publications":             total_pubs,
            "total_unique_coauthors":         metrics["total_unique_coauthors"],
            "recurring_collaborators_count":  metrics["recurring_collaborators_count"],
            "one_time_collaborators":         metrics["one_time_collaborators"],
            "avg_authors_per_paper":          metrics["avg_authors_per_paper"],
            "student_collaborations":         student_matches_count,
            "collaboration_diversity_score":  metrics["collaboration_diversity_score"],
            "team_size_profile":              metrics["team_size_profile"],
            "international_collaboration":    intl_collab,
            "collaboration_strength_label":   strength_label,
            "top_collaborators_json":         top_collabs_json,
            "summary":                        summary,
            "rule_based_label":               rule_label,
        }

        return collab_profile, network_rows

    # ------------------------------------------------------------------
    # LLM Interpretation
    # ------------------------------------------------------------------

    def _interpret_collaboration_profile(
        self,
        candidate_id: str,
        total_pubs: int,
        total_unique_coauthors: int,
        recurring_collaborators_count: int,
        one_time_collaborators: int,
        top_collaborators_json: str,
        avg_authors_per_paper: float,
        student_collaborations: int,
        diversity_score: float,
        team_size_profile: str,
        intl_collab: str,
        rule_label: str,
    ) -> Dict[str, Any]:
        user_prompt = _COLLAB_USER_TEMPLATE.format(
            candidate_id                  = candidate_id,
            total_pubs                    = total_pubs,
            total_unique_coauthors        = total_unique_coauthors,
            recurring_collaborators_count = recurring_collaborators_count,
            one_time_collaborators        = one_time_collaborators,
            top_collaborators_json        = top_collaborators_json,
            avg_authors_per_paper         = avg_authors_per_paper,
            student_collaborations        = student_collaborations,
            collaboration_diversity_score = diversity_score,
            team_size_profile             = team_size_profile,
            international_collaboration   = intl_collab,
            rule_label                    = rule_label,
        )

        try:
            raw = self._call_llm(_COLLAB_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "collaboration_strength_label" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM collaboration assessment failed for '%s': %s", candidate_id, e)

        return {
            "collaboration_strength_label": rule_label,
            "summary": self._rule_based_summary(
                candidate_id, total_pubs, total_unique_coauthors,
                recurring_collaborators_count, json.loads(top_collaborators_json),
                avg_authors_per_paper, team_size_profile, rule_label
            ),
        }

    @staticmethod
    def _rule_based_summary(
        name: str,
        total: int,
        unique_coauthors: int,
        recurring_cnt: int,
        top_collabs: List[Dict],
        avg_authors: float,
        team_profile: str,
        label: str,
    ) -> str:
        if total == 0:
            return f"Candidate {name} has no publications listed in their CV."

        top_names = [f"{c['name']} ({c['paper_count']} papers)" for c in top_collabs[:3]]
        top_str = ", ".join(top_names) if top_names else "none"

        parts = [
            f"Candidate {name} has collaborated with {unique_coauthors} unique co-author(s) across {total} publication(s).",
            f"They operate as a {team_profile} with an average of {avg_authors:.1f} authors per paper.",
            f"The candidate has {recurring_cnt} recurring collaborator(s) (top co-authors: {top_str}).",
            f"Overall research collaboration network is categorized as {label}."
        ]
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        collab_profiles: List[Dict],
        network_rows:    List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        label_rank = {"Broad Network": 1, "Balanced Network": 2, "Closed Network": 3, "Solo Researcher": 4, "No Publications": 5}
        collab_profiles = sorted(
            collab_profiles,
            key=lambda x: (
                label_rank.get(x.get("collaboration_strength_label"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Co-Author Network CSV
        p = self.output_dir / "coauthor_network.csv"
        pd.DataFrame(network_rows).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(network_rows), p.name)
        paths["coauthor_network"] = p

        # 2. Collaboration Profiles CSV
        p = self.output_dir / "collaboration_profiles.csv"
        pd.DataFrame(collab_profiles).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(collab_profiles), p.name)
        paths["collab_profiles"] = p

        # 3. Excel Workbook
        p = self.output_dir / "collaboration_profiles.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            pd.DataFrame(collab_profiles).to_excel(
                writer, sheet_name="Collaboration Profiles", index=False
            )
            pd.DataFrame(network_rows).to_excel(
                writer, sheet_name="Coauthor Network", index=False
            )
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 4. Text Report
        self._write_text_report(collab_profiles, network_rows, paths)

        return paths

    def _write_text_report(self, collab_profiles: List[Dict], network_rows: List[Dict], paths: Dict):
        p = self.output_dir / "collaboration_report.txt"

        label_counts: Dict[str, int] = {}
        for cp in collab_profiles:
            lbl = cp.get("collaboration_strength_label") or "Unknown"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        total_network_nodes = len(network_rows)

        lines = [
            "=" * 70,
            "  TALASH Module 8 – Co-Author Collaboration Analysis Report",
            "=" * 70,
            f"  Candidates analysed               : {len(collab_profiles)}",
            f"  Total unique coauthor interactions: {total_network_nodes}",
            "",
            "  Collaboration Network Label Distribution:",
        ]
        for lbl, cnt in sorted(label_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += [
            "",
            "-" * 70,
            "  Per-Candidate Collaboration Summaries",
            "-" * 70,
        ]

        for cp in collab_profiles:
            cid  = cp["candidate_id"]
            lbl  = cp.get("collaboration_strength_label", "Unknown")
            tot  = cp.get("total_publications", 0)
            uniq = cp.get("total_unique_coauthors", 0)
            avg  = cp.get("avg_authors_per_paper", 0.0)
            prof = cp.get("team_size_profile", "N/A")
            div  = cp.get("collaboration_diversity_score", 0.0)

            lines.append(f"\n  [{lbl}] {cid}")
            lines.append(f"  Total Publications: {tot} | Unique Co-Authors: {uniq} | Recurring: {cp.get('recurring_collaborators_count',0)} | 1-Time: {cp.get('one_time_collaborators',0)}")
            lines.append(f"  Team Size Profile: {prof} (avg {avg:.1f} authors/paper) | Diversity Score: {div:.2f}")
            lines.append(f"  Top Collaborators: {cp.get('top_collaborators_json','[]')}")
            if cp.get("summary"):
                lines.append(f"  Summary: {cp['summary']}")

        lines.append("\n" + "=" * 70)
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report written -> %s", p.name)
        paths["report"] = p

    # ------------------------------------------------------------------
    # Checkpoint Helpers
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
    # LLM + JSON Helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system: str, user: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed.")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature = 0.1,
            max_tokens  = 512,
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
        for pat in [
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"(\{.*\})",
        ]:
            m = re.search(pat, raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
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
