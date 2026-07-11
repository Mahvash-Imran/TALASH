"""
educational_profile.py  –  Module 2 Orchestrator
==================================================

WHY THIS FILE EXISTS
--------------------
This is the top-level controller for Part 2 (Educational Profile Analysis).
It reads the CSV outputs of Module 1 (education.csv, experience.csv), runs
tasks 2.1-2.9 in order, and produces:

  data/analysis/educational_profiles.csv   – one row per candidate
  data/analysis/edu_degrees.csv            – one row per degree
  data/analysis/edu_gaps.csv               – one row per gap
  data/analysis/edu_report.txt             – human-readable summary

PIPELINE (in order)
-------------------
  EduNormalizer      → tasks 2.1-2.4  (classify levels, normalize marks)
  RankingLookup      → task  2.5      (THE / QS ranking per institution)
  ProgressionChecker → tasks 2.6-2.7  (progression consistency + gap detection)
  GapJustifier       → task  2.8      (cross-reference gaps vs experience.csv)
  EduInterpreter     → task  2.9      (LLM summary + strength label)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .edu_normalizer     import EduNormalizer
from .ranking_lookup     import RankingLookup
from .progression_checker import ProgressionChecker
from .gap_justifier      import GapJustifier
from .edu_interpreter    import EduInterpreter

logger = logging.getLogger(__name__)


class EducationalProfileAnalyser:
    """
    Orchestrates tasks 2.1-2.9 for every candidate in education.csv.

    Usage
    -----
    analyser = EducationalProfileAnalyser(
        education_csv  = "data/extracted/education.csv",
        experience_csv = "data/extracted/experience.csv",
        output_dir     = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        education_csv:  str = "data/extracted/education.csv",
        experience_csv: str = "data/extracted/experience.csv",
        output_dir:     str = "data/analysis",
        rankings_csv:   Optional[str] = None,
        api_key:        Optional[str] = None,
        model:          str = "llama-3.3-70b-versatile",
        base_url:       Optional[str] = None,
        skip_llm:       bool = False,
    ):
        self.education_csv  = Path(education_csv)
        self.experience_csv = Path(experience_csv)
        self.output_dir     = Path(output_dir)
        self.skip_llm       = skip_llm

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate pipeline components
        self.normalizer  = EduNormalizer()
        self.ranking     = RankingLookup(csv_path=rankings_csv)
        self.progression = ProgressionChecker()
        self.justifier   = GapJustifier()
        self.interpreter = EduInterpreter(
            api_key=api_key, model=model, base_url=base_url
        )

    # ------------------------------------------------------------------
    # Public: run full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Path]:
        """
        Run the full Module 2 pipeline for all candidates.

        Returns a dict of output file paths.
        """
        logger.info("=" * 60)
        logger.info("  TALASH Module 2 - Educational Profile Analysis")
        logger.info("  Education CSV  : %s", self.education_csv)
        logger.info("  Experience CSV : %s", self.experience_csv)
        logger.info("  Output dir     : %s", self.output_dir)
        logger.info("=" * 60)

        # Load source CSVs
        edu_df  = self._load_csv(self.education_csv,  "education")
        exp_df  = self._load_csv(self.experience_csv, "experience")

        if edu_df is None or edu_df.empty:
            logger.error("education.csv is empty or missing. Run Module 1 first.")
            return {}

        # Get all candidate IDs from education.csv
        candidate_ids = edu_df["candidate_id"].dropna().unique().tolist()
        logger.info("Found %d candidate(s) in education.csv", len(candidate_ids))

        all_profiles  = []   # one dict per candidate
        all_degrees   = []   # one dict per degree row
        all_gaps      = []   # one dict per gap

        for i, cid in enumerate(candidate_ids, 1):
            logger.info("[%d/%d] Analysing: %s", i, len(candidate_ids), cid)

            # Get this candidate's rows
            edu_rows = edu_df[edu_df["candidate_id"] == cid].to_dict("records")
            exp_rows = (
                exp_df[exp_df["candidate_id"] == cid].to_dict("records")
                if exp_df is not None and not exp_df.empty else []
            )

            # Run per-candidate pipeline
            profile, degrees, gaps = self.analyse_candidate(cid, edu_rows, exp_rows)

            all_profiles.append(profile)
            all_degrees.extend(degrees)
            all_gaps.extend(gaps)

        # Export
        output_paths = self._export(all_profiles, all_degrees, all_gaps)
        logger.info("[DONE] Module 2 complete. Outputs in: %s", self.output_dir)
        return output_paths

    # ------------------------------------------------------------------
    # Per-candidate pipeline
    # ------------------------------------------------------------------

    def analyse_candidate(
        self,
        candidate_id: str,
        edu_rows:     List[Dict],
        exp_rows:     List[Dict],
    ) -> tuple:
        """
        Run tasks 2.1-2.9 for one candidate.

        Returns
        -------
        (profile_dict, enriched_degree_rows, gap_rows)
        """

        # Tasks 2.1-2.4: level classification + marks normalization
        enriched = self.normalizer.process(edu_rows)

        # Task 2.5: institutional ranking lookup
        enriched = self.ranking.enrich_education_rows(enriched)

        # Tasks 2.6-2.7: progression + gap detection
        prog = self.progression.analyse(enriched)

        # Task 2.8: gap justification
        justified_gaps = self.justifier.justify(
            prog["educational_gaps"], exp_rows
        )
        prog["educational_gaps"] = justified_gaps

        # Task 2.9: LLM interpretation (can be skipped for pure data runs)
        if self.skip_llm:
            interp = {
                "educational_strength": EduInterpreter._rule_based_label(prog),
                "summary":              EduInterpreter._rule_based_summary(candidate_id, prog,
                                            EduInterpreter._rule_based_label(prog)),
                "rule_based_label":     EduInterpreter._rule_based_label(prog),
            }
        else:
            interp = self.interpreter.interpret(candidate_id, prog)

        # Combine into a profile row
        profile = {
            "candidate_id":           candidate_id,
            "highest_degree":         prog["highest_degree"],
            "progression_consistent": prog["progression_consistent"],
            "performance_trend":      prog["performance_trend"],
            "specialization_drift":   len(prog["specialization_drift"]) > 0,
            "drift_details":          str(prog["specialization_drift"]) if prog["specialization_drift"] else None,
            "total_gaps":             len(justified_gaps),
            "significant_gaps":       sum(1 for g in justified_gaps if g.get("significant")),
            "unexplained_gaps":       sum(1 for g in justified_gaps if g.get("justification_type") == "Unexplained"),
            "educational_strength":   interp.get("educational_strength"),
            "rule_based_label":       interp.get("rule_based_label"),
            "summary":                interp.get("summary"),
        }

        # Tag degree rows with candidate_id
        degree_rows = []
        for d in prog.get("degrees_sorted", []):
            row = dict(d)
            row["candidate_id"] = candidate_id
            degree_rows.append(row)

        # Tag gap rows
        gap_rows = []
        for g in justified_gaps:
            row = dict(g)
            row["candidate_id"] = candidate_id
            gap_rows.append(row)

        return profile, degree_rows, gap_rows

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        profiles: List[Dict],
        degrees:  List[Dict],
        gaps:     List[Dict],
    ) -> Dict[str, Path]:
        paths = {}

        # 1. Educational profiles summary
        if profiles:
            p = self.output_dir / "educational_profiles.csv"
            pd.DataFrame(profiles).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(profiles), p.name)
            paths["profiles"] = p

        # 2. Per-degree details
        if degrees:
            p = self.output_dir / "edu_degrees.csv"
            pd.DataFrame(degrees).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(degrees), p.name)
            paths["degrees"] = p

        # 3. Per-gap details
        if gaps:
            p = self.output_dir / "edu_gaps.csv"
            pd.DataFrame(gaps).to_csv(p, index=False)
            logger.info("Wrote %d rows -> %s", len(gaps), p.name)
            paths["gaps"] = p

        # 4. Excel workbook (all three sheets)
        p = self.output_dir / "educational_profiles.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            if profiles:
                pd.DataFrame(profiles).to_excel(writer, sheet_name="Profiles", index=False)
            if degrees:
                pd.DataFrame(degrees).to_excel(writer, sheet_name="Degrees", index=False)
            if gaps:
                pd.DataFrame(gaps).to_excel(writer, sheet_name="Gaps", index=False)
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 5. Plain-text report
        self._write_text_report(profiles, paths)

        return paths

    def _write_text_report(self, profiles: List[Dict], paths: Dict):
        p = self.output_dir / "edu_report.txt"
        lines = [
            "=" * 70,
            "  TALASH Module 2 - Educational Profile Report",
            "=" * 70,
            f"  Candidates analysed : {len(profiles)}",
            "",
        ]

        strength_counts: Dict[str, int] = {}
        for prof in profiles:
            s = prof.get("educational_strength") or "Unknown"
            strength_counts[s] = strength_counts.get(s, 0) + 1

        lines.append("  Strength Distribution:")
        for label, count in sorted(strength_counts.items()):
            lines.append(f"    {label:<22}: {count}")

        lines += ["", "-" * 70, "  Per-Candidate Summaries", "-" * 70]
        for prof in profiles:
            lines.append(f"\n  [{prof.get('educational_strength')}] {prof['candidate_id']}")
            lines.append(f"  Highest Degree : {prof.get('highest_degree')}")
            lines.append(f"  Progression    : {'Consistent' if prof.get('progression_consistent') else 'Inconsistent'}")
            lines.append(f"  Perf. Trend    : {prof.get('performance_trend')}")
            lines.append(f"  Gaps (total/significant/unexplained): "
                         f"{prof.get('total_gaps')}/{prof.get('significant_gaps')}/{prof.get('unexplained_gaps')}")
            if prof.get("summary"):
                lines.append(f"  Summary: {prof['summary']}")

        lines.append("\n" + "=" * 70)

        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report written -> %s", p.name)
        paths["report"] = p

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
