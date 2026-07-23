"""
experience_analyser.py  –  Module 9: Professional Experience & Skill Alignment Orchestrator
==========================================================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates Part 9 (Professional Experience & Skill Alignment Analysis):
  - Combines education and experience into unified YYYY-MM timelines
  - Detects multi-job and education-employment overlaps (Acceptable vs Suspicious)
  - Detects career gaps > 3 months and checks active degree/publication justification
  - Tracks career progression trajectories and total experience years
  - Extracts skills and classifies evidence levels (Strong, Moderate, Unverified)
  - Computes JD alignment scores
  - Calls LLM to generate narrative summaries (with rule-based fallback)
  - Exports unified_timelines.csv, timeline_overlaps.csv, experience_gaps.csv,
    skill_evidence.csv, experience_profiles.csv, experience_profiles.xlsx, and experience_report.txt
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .timeline_builder import (
    standardize_date,
    build_candidate_timeline,
    detect_overlaps,
    detect_experience_gaps,
    assess_career_progression,
)
from .skill_aligner import (
    extract_and_align_skills,
    compute_jd_alignment_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

_EXP_SYSTEM_PROMPT = (
    "You are an expert academic and industry talent evaluation officer for a university recruitment system. "
    "You will receive structured experience timeline, overlap, gap, career progression, and skill evidence metrics for a faculty candidate. "
    "Your task is to write a concise, factual assessment of their professional experience and skill alignment. "
    "Return ONLY a valid JSON object with exactly two keys: "
    "\"experience_label\" and \"summary\". "
    "experience_label must be one of: \"Strong Profile\", \"Solid Profile\", \"Developing Profile\", \"Needs Clarification\", \"No Experience Listed\". "
    "summary must be 2-4 sentences in third person, factual, with no embellishment."
)

_EXP_USER_TEMPLATE = (
    "Assess this candidate's professional experience and skill alignment, returning JSON.\n\n"
    "CANDIDATE ID: {candidate_id}\n\n"
    "EXPERIENCE STATISTICS:\n"
    "  Total experience years    : {total_experience_years}\n"
    "  Career progression        : {career_progression}\n"
    "  Total timeline events     : {total_events}\n"
    "  Overlaps detected         : {overlap_count} ({suspicious_overlaps} suspicious)\n"
    "  Gaps detected             : {gap_count} ({unjustified_gaps} unjustified)\n"
    "  Strong evidence skills    : {strong_skills_count} ({strong_skills_list})\n"
    "  Unverified skills         : {unverified_skills_count}\n"
    "  JD alignment score        : {jd_score}% ({jd_alignment_label})\n\n"
    "SUGGESTED LABEL (rule-based): {rule_label}\n\n"
    "Return JSON:"
)


class ExperienceProfileAnalyser:
    """
    Orchestrates tasks 9.1–9.5 for every candidate in experience.csv, education.csv, skills.csv, and publications.csv.

    Usage
    -----
    analyser = ExperienceProfileAnalyser(
        candidates_csv   = "data/extracted/candidates.csv",
        education_csv    = "data/extracted/education.csv",
        experience_csv   = "data/extracted/experience.csv",
        skills_csv       = "data/extracted/skills.csv",
        publications_csv = "data/extracted/publications.csv",
        output_dir       = "data/analysis",
    )
    analyser.run()
    """

    def __init__(
        self,
        candidates_csv:   str = "data/extracted/candidates.csv",
        education_csv:    str = "data/extracted/education.csv",
        experience_csv:   str = "data/extracted/experience.csv",
        skills_csv:       str = "data/extracted/skills.csv",
        publications_csv: str = "data/extracted/publications.csv",
        output_dir:       str = "data/analysis",
        jd_requirements:  Optional[List[str]] = None,
        api_key:          Optional[str] = None,
        model:            str = "meta-llama/llama-4-scout-17b-16e-instruct",
        base_url:         Optional[str] = None,
        skip_llm:         bool = False,
    ):
        self.candidates_csv   = Path(candidates_csv)
        self.education_csv    = Path(education_csv)
        self.experience_csv   = Path(experience_csv)
        self.skills_csv       = Path(skills_csv)
        self.publications_csv = Path(publications_csv)
        self.output_dir       = Path(output_dir)
        self.jd_requirements  = jd_requirements
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
        logger.info("  TALASH Module 9 – Experience & Skill Alignment Analysis")
        logger.info("  Candidates CSV   : %s", self.candidates_csv)
        logger.info("  Experience CSV   : %s", self.experience_csv)
        logger.info("  Education CSV    : %s", self.education_csv)
        logger.info("  Skills CSV       : %s", self.skills_csv)
        logger.info("  Publications CSV : %s", self.publications_csv)
        logger.info("  Output dir       : %s", self.output_dir)
        logger.info("=" * 60)

        cand_df = self._load_csv(self.candidates_csv, "candidates")
        edu_df  = self._load_csv(self.education_csv,  "education")
        exp_df  = self._load_csv(self.experience_csv, "experience")
        sk_df   = self._load_csv(self.skills_csv,     "skills")
        pub_df  = self._load_csv(self.publications_csv, "publications")

        if cand_df is None or cand_df.empty:
            logger.error("candidates.csv missing or empty. Run Module 1 first.")
            return {}

        name_map: Dict[str, str] = dict(
            zip(cand_df["candidate_id"].astype(str), cand_df["name"].astype(str))
        )
        all_cids = cand_df["candidate_id"].dropna().unique().tolist()

        # Group data by candidate_id
        edu_by_cid:  Dict[str, List[Dict]] = self._group_by_cid(edu_df)
        exp_by_cid:  Dict[str, List[Dict]] = self._group_by_cid(exp_df)
        sk_by_cid:   Dict[str, List[Dict]] = self._group_by_cid(sk_df)
        pub_by_cid:  Dict[str, List[Dict]] = self._group_by_cid(pub_df)

        # Module 7 publication themes (if generated)
        pub_themes_path = self.output_dir / "publication_themes.csv"
        pub_themes_df = self._load_csv(pub_themes_path, "publication_themes")
        themes_by_cid: Dict[str, List[Dict]] = self._group_by_cid(pub_themes_df) if pub_themes_df is not None else {}

        # Checkpointing
        checkpoint_path = self.output_dir / "_experience_checkpoint.json"
        checkpoint      = self._load_checkpoint(checkpoint_path)
        completed_cids  = set(checkpoint.get("completed", []))

        all_profiles:  List[Dict] = list(checkpoint.get("profiles",  []))
        all_timelines: List[Dict] = list(checkpoint.get("timelines", []))
        all_overlaps:  List[Dict] = list(checkpoint.get("overlaps",  []))
        all_gaps:      List[Dict] = list(checkpoint.get("gaps",      []))
        all_skills:    List[Dict] = list(checkpoint.get("skills",    []))

        if completed_cids:
            logger.info("Checkpoint found: %d candidate(s) already done, resuming.", len(completed_cids))

        for i, cid in enumerate(all_cids, 1):
            if cid in completed_cids:
                logger.info("[%d/%d] SKIP (checkpoint): %s", i, len(all_cids), cid)
                continue

            logger.info("[%d/%d] Analysing experience & skills for: %s", i, len(all_cids), cid)
            cand_name = name_map.get(str(cid), str(cid))
            cand_edu  = edu_by_cid.get(str(cid), [])
            cand_exp  = exp_by_cid.get(str(cid), [])
            cand_sk   = sk_by_cid.get(str(cid), [])
            cand_pub  = pub_by_cid.get(str(cid), [])
            cand_thm  = themes_by_cid.get(str(cid), [])

            prof, timelines, overlaps, gaps, skills = self._analyse_candidate(
                cid, cand_name, cand_edu, cand_exp, cand_sk, cand_pub, cand_thm
            )

            all_profiles.append(prof)
            all_timelines.extend(timelines)
            all_overlaps.extend(overlaps)
            all_gaps.extend(gaps)
            all_skills.extend(skills)

            completed_cids.add(str(cid))
            self._save_checkpoint(checkpoint_path, {
                "completed": list(completed_cids),
                "profiles":  all_profiles,
                "timelines": all_timelines,
                "overlaps":  all_overlaps,
                "gaps":      all_gaps,
                "skills":    all_skills,
            })

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Checkpoint cleared (run complete).")

        return self._export(all_profiles, all_timelines, all_overlaps, all_gaps, all_skills)

    # ------------------------------------------------------------------
    # Per-Candidate Analysis
    # ------------------------------------------------------------------

    def _analyse_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        cand_edu: List[Dict[str, Any]],
        cand_exp: List[Dict[str, Any]],
        cand_sk:  List[Dict[str, Any]],
        cand_pub: List[Dict[str, Any]],
        cand_thm: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

        # Task 9.1: Build Unified Timeline
        timeline_events = build_candidate_timeline(candidate_id, cand_edu, cand_exp)

        # Task 9.2: Overlap Detection
        overlaps = detect_overlaps(timeline_events)
        suspicious_overlaps = sum(1 for o in overlaps if o["classification"] == "Suspicious")

        # Task 9.3: Gap & Career Progression
        gaps = detect_experience_gaps(timeline_events, cand_pub)
        unjustified_gaps = sum(1 for g in gaps if g["justification_status"] == "Unjustified Gap")

        exp_events_only = [e for e in timeline_events if e["event_type"] == "experience"]
        career_progression, total_years = assess_career_progression(exp_events_only)

        # Task 9.4: Skill Alignment & Evidence Classification
        skills_evidence = extract_and_align_skills(
            candidate_id, cand_sk, cand_exp, cand_pub, cand_thm
        )
        strong_skills = [s["skill_name"] for s in skills_evidence if s["evidence_level"] == "Strong Evidence"]
        unverified_skills = [s["skill_name"] for s in skills_evidence if s["evidence_level"] == "Unverified / Low Evidence"]

        # Task 9.5: JD Alignment Scoring
        jd_metrics = compute_jd_alignment_score(skills_evidence, self.jd_requirements)

        # Rule-based Label
        if total_years >= 5 and suspicious_overlaps == 0 and unjustified_gaps == 0:
            rule_label = "Strong Profile"
        elif total_years >= 2 and suspicious_overlaps <= 1:
            rule_label = "Solid Profile"
        elif len(exp_events_only) > 0:
            rule_label = "Developing Profile"
        elif suspicious_overlaps > 1 or unjustified_gaps > 1:
            rule_label = "Needs Clarification"
        else:
            rule_label = "No Experience Listed"

        # Task 9.5: LLM Narrative Interpretation
        if self.skip_llm or total_years == 0:
            exp_label = rule_label
            summary = self._rule_based_summary(
                candidate_name, total_years, career_progression, len(timeline_events),
                len(overlaps), len(gaps), strong_skills, jd_metrics["jd_alignment_score"], rule_label
            )
        else:
            interp = self._interpret_experience_profile(
                candidate_id, total_years, career_progression, len(timeline_events),
                len(overlaps), suspicious_overlaps, len(gaps), unjustified_gaps,
                len(strong_skills), ", ".join(strong_skills[:3]), len(unverified_skills),
                jd_metrics["jd_alignment_score"], jd_metrics["alignment_label"], rule_label
            )
            exp_label = interp.get("experience_label", rule_label)
            summary   = interp.get("summary", "")
            time.sleep(0.5)

        profile = {
            "candidate_id":            candidate_id,
            "candidate_name":          candidate_name,
            "total_experience_years":  total_years,
            "career_progression":      career_progression,
            "total_timeline_events":   len(timeline_events),
            "overlap_count":           len(overlaps),
            "suspicious_overlaps":     suspicious_overlaps,
            "gap_count":               len(gaps),
            "unjustified_gaps":        unjustified_gaps,
            "strong_skills_count":     len(strong_skills),
            "top_strong_skills":       ", ".join(strong_skills[:4]),
            "unverified_skills_count": len(unverified_skills),
            "jd_alignment_score":      jd_metrics["jd_alignment_score"],
            "jd_alignment_label":      jd_metrics["alignment_label"],
            "experience_label":        exp_label,
            "summary":                 summary,
            "rule_based_label":        rule_label,
        }

        # Build clean export rows for timelines
        export_timelines = []
        for te in timeline_events:
            export_timelines.append({
                "candidate_id": candidate_id,
                "event_type":   te["event_type"],
                "title":        te["title"],
                "organization": te["organization"],
                "start_date":   te["start_str"],
                "end_date":     te["end_str"],
                "is_current":   te["is_current"],
            })

        return profile, export_timelines, overlaps, gaps, skills_evidence

    # ------------------------------------------------------------------
    # LLM Interpretation
    # ------------------------------------------------------------------

    def _interpret_experience_profile(
        self,
        candidate_id: str,
        total_years: float,
        career_progression: str,
        total_events: int,
        overlap_count: int,
        suspicious_overlaps: int,
        gap_count: int,
        unjustified_gaps: int,
        strong_skills_count: int,
        strong_skills_list: str,
        unverified_skills_count: int,
        jd_score: float,
        jd_alignment_label: str,
        rule_label: str,
    ) -> Dict[str, Any]:
        user_prompt = _EXP_USER_TEMPLATE.format(
            candidate_id            = candidate_id,
            total_experience_years  = total_years,
            career_progression      = career_progression,
            total_events            = total_events,
            overlap_count           = overlap_count,
            suspicious_overlaps     = suspicious_overlaps,
            gap_count               = gap_count,
            unjustified_gaps        = unjustified_gaps,
            strong_skills_count     = strong_skills_count,
            strong_skills_list      = strong_skills_list or "None",
            unverified_skills_count = unverified_skills_count,
            jd_score                = jd_score,
            jd_alignment_label      = jd_alignment_label,
            rule_label              = rule_label,
        )

        try:
            raw = self._call_llm(_EXP_SYSTEM_PROMPT, user_prompt)
            parsed = self._parse_json(raw, candidate_id)
            if parsed and "experience_label" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM experience assessment failed for '%s': %s", candidate_id, e)

        return {
            "experience_label": rule_label,
            "summary": self._rule_based_summary(
                candidate_id, total_years, career_progression, total_events,
                overlap_count, gap_count, [s.strip() for s in strong_skills_list.split(",") if s.strip()],
                jd_score, rule_label
            ),
        }

    @staticmethod
    def _rule_based_summary(
        name: str,
        total_years: float,
        progression: str,
        total_events: int,
        overlap_cnt: int,
        gap_cnt: int,
        strong_skills: List[str],
        jd_score: float,
        label: str,
    ) -> str:
        if total_years == 0 and total_events == 0:
            return f"Candidate {name} has no professional experience listed in their CV."

        skills_str = ", ".join(strong_skills[:3]) if strong_skills else "general domain competencies"
        parts = [
            f"Candidate {name} possesses {total_years:.1f} years of total professional experience demonstrating a {progression.lower()} trajectory.",
            f"Their unified timeline contains {total_events} recorded activity event(s), with {overlap_cnt} overlap(s) and {gap_cnt} gap(s) identified.",
            f"Proven technical competencies include {skills_str}, achieving a {jd_score:.1f}% position alignment score.",
            f"Overall professional experience profile is classified as {label}."
        ]
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(
        self,
        profiles:  List[Dict],
        timelines: List[Dict],
        overlaps:  List[Dict],
        gaps:      List[Dict],
        skills:    List[Dict],
    ) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}

        label_rank = {"Strong Profile": 1, "Solid Profile": 2, "Developing Profile": 3, "Needs Clarification": 4, "No Experience Listed": 5}
        profiles = sorted(
            profiles,
            key=lambda x: (
                label_rank.get(x.get("experience_label"), 99),
                x.get("candidate_id", ""),
            )
        )

        # 1. Unified Timelines CSV
        p = self.output_dir / "unified_timelines.csv"
        pd.DataFrame(timelines).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(timelines), p.name)
        paths["unified_timelines"] = p

        # 2. Timeline Overlaps CSV
        p = self.output_dir / "timeline_overlaps.csv"
        pd.DataFrame(overlaps).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(overlaps), p.name)
        paths["timeline_overlaps"] = p

        # 3. Experience Gaps CSV
        p = self.output_dir / "experience_gaps.csv"
        pd.DataFrame(gaps).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(gaps), p.name)
        paths["experience_gaps"] = p

        # 4. Skill Evidence CSV
        p = self.output_dir / "skill_evidence.csv"
        pd.DataFrame(skills).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(skills), p.name)
        paths["skill_evidence"] = p

        # 5. Experience Profiles CSV
        p = self.output_dir / "experience_profiles.csv"
        pd.DataFrame(profiles).to_csv(p, index=False)
        logger.info("Wrote %d rows -> %s", len(profiles), p.name)
        paths["experience_profiles"] = p

        # 6. Excel Workbook
        p = self.output_dir / "experience_profiles.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as writer:
            pd.DataFrame(profiles).to_excel(writer, sheet_name="Experience Profiles", index=False)
            pd.DataFrame(timelines).to_excel(writer, sheet_name="Unified Timelines", index=False)
            pd.DataFrame(overlaps).to_excel(writer, sheet_name="Timeline Overlaps", index=False)
            pd.DataFrame(gaps).to_excel(writer, sheet_name="Experience Gaps", index=False)
            pd.DataFrame(skills).to_excel(writer, sheet_name="Skill Evidence Alignment", index=False)
        logger.info("Excel workbook written -> %s", p.name)
        paths["excel"] = p

        # 7. Text Report
        self._write_text_report(profiles, overlaps, gaps, paths)

        return paths

    def _write_text_report(self, profiles: List[Dict], overlaps: List[Dict], gaps: List[Dict], paths: Dict):
        p = self.output_dir / "experience_report.txt"

        label_counts: Dict[str, int] = {}
        for cp in profiles:
            lbl = cp.get("experience_label") or "Unknown"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        lines = [
            "=" * 70,
            "  TALASH Module 9 – Experience & Skill Alignment Report",
            "=" * 70,
            f"  Candidates analysed          : {len(profiles)}",
            f"  Total timeline overlaps      : {len(overlaps)} ({sum(1 for o in overlaps if o['classification']=='Suspicious')} suspicious)",
            f"  Total career gaps > 3 months : {len(gaps)} ({sum(1 for g in gaps if g['justification_status']=='Unjustified Gap')} unjustified)",
            "",
            "  Experience Profile Type Distribution:",
        ]
        for lbl, cnt in sorted(label_counts.items()):
            lines.append(f"    {lbl:<25}: {cnt}")

        lines += [
            "",
            "-" * 70,
            "  Per-Candidate Experience & Skill Summaries",
            "-" * 70,
        ]

        for cp in profiles:
            cid  = cp["candidate_id"]
            lbl  = cp.get("experience_label", "Unknown")
            yrs  = cp.get("total_experience_years", 0.0)
            prog = cp.get("career_progression", "N/A")
            ovl  = cp.get("overlap_count", 0)
            gap  = cp.get("gap_count", 0)
            strg = cp.get("top_strong_skills", "None")
            jd   = cp.get("jd_alignment_score", 0.0)

            lines.append(f"\n  [{lbl}] {cid}")
            lines.append(f"  Total Experience: {yrs:.1f} yrs | Trajectory: {prog} | Overlaps: {ovl} | Gaps: {gap}")
            lines.append(f"  Proven Skills: {strg} | JD Score: {jd:.1f}% ({cp.get('jd_alignment_label','N/A')})")
            if cp.get("summary"):
                lines.append(f"  Summary: {cp['summary']}")

        lines.append("\n" + "=" * 70)
        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Text report written -> %s", p.name)
        paths["report"] = p

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_cid(df: Optional[pd.DataFrame]) -> Dict[str, List[Dict]]:
        res: Dict[str, List[Dict]] = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                cid = str(row.get("candidate_id", "")).strip()
                if cid:
                    res.setdefault(cid, []).append(row.to_dict())
        return res

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
            path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Could not save checkpoint: %s", e)

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
        for pat in [r"```(?:json)?\s*(\{.*?\})\s*```", r"(\{.*\})"]:
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
