"""
composite_evaluator.py  –  Module 10 Engine: Candidate Composite Scoring & Tier Evaluation
========================================================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates Part 10 (Composite Candidate Evaluation & Overall Scoring):
  - Integrates results from Modules 2 through 9 into a single weighted score out of 100
  - Weights:
      1. Educational Record & Quality        : 20%
      2. Research Quality & Impact (Pubs)     : 25%
      3. Student Supervision & Guidance       : 10%
      4. Books & Patents Output               : 10%
      5. Topic Breadth & Diversity Score      : 10%
      6. Co-Author Collaboration Network      : 10%
      7. Experience & Skill Alignment         : 15%
  - Classifies candidates into Tiers:
      Tier 1: Exceptional Fit          (Score >= 80)
      Tier 2: Strong Candidate         (Score 65 - 79)
      Tier 3: Moderate Candidate       (Score 50 - 64)
      Tier 4: Needs Clarification/Weak (Score < 50)
  - Exports data/analysis/composite_evaluations.csv and composite_evaluations.xlsx
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def compute_candidate_composite_score(
    candidate_id: str,
    edu_profile: Optional[Dict[str, Any]] = None,
    research_profile: Optional[Dict[str, Any]] = None,
    supervision_profile: Optional[Dict[str, Any]] = None,
    book_profile: Optional[Dict[str, Any]] = None,
    patent_profile: Optional[Dict[str, Any]] = None,
    breadth_profile: Optional[Dict[str, Any]] = None,
    collab_profile: Optional[Dict[str, Any]] = None,
    exp_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Computes a weighted overall composite score out of 100 for a candidate.
    """
    # 1. Educational Score (Max 20 pts)
    edu_score = 10.0  # Base
    if edu_profile:
        lbl = str(edu_profile.get("educational_strength") or edu_profile.get("educational_strength_label") or edu_profile.get("rule_based_label") or "").lower()
        if "strong" in lbl:
            edu_score = 18.0
        elif "moderate" in lbl:
            edu_score = 14.0
        elif "weak" in lbl or "needs" in lbl:
            edu_score = 9.0

        deg = str(edu_profile.get("highest_degree") or "").lower()
        phd_cnt = float(edu_profile.get("phd_count") or 0)
        if "phd" in deg or "ph.d" in deg or "doctor" in deg or phd_cnt > 0:
            edu_score = min(edu_score + 2.0, 20.0)

    # 2. Research Quality Score (Max 25 pts)
    research_score = 5.0
    if research_profile:
        lbl = str(research_profile.get("research_strength") or research_profile.get("rule_based_label") or research_profile.get("scholarly_strength_label") or "").lower()
        tj = float(research_profile.get("total_journals") or 0)
        tc = float(research_profile.get("total_conferences") or 0)
        pub_cnt = float(research_profile.get("total_publications") or (tj + tc))
        q1_q2 = float(research_profile.get("q1_papers") or 0) + float(research_profile.get("q2_papers") or 0)

        if "strong" in lbl or q1_q2 >= 3 or pub_cnt >= 10:
            research_score = 22.0
        elif "moderate" in lbl or q1_q2 >= 1 or pub_cnt >= 4:
            research_score = 16.0
        elif pub_cnt >= 1:
            research_score = 10.0
        else:
            research_score = 5.0

        if pub_cnt >= 15 or q1_q2 >= 5:
            research_score = min(research_score + 3.0, 25.0)

    # 3. Student Supervision Score (Max 10 pts)
    # Default: neutral 5.0 when data is absent from the CV (data_missing=True).
    # Penalise with 2.0 only when we have a confirmed record with zero students.
    sup_score = 5.0
    if supervision_profile:
        data_missing = str(supervision_profile.get("data_missing") or "").strip().lower()
        ms_cnt   = float(supervision_profile.get("total_ms_supervised")  or supervision_profile.get("ms_main_supervisor")  or 0)
        phd_cnt  = float(supervision_profile.get("total_phd_supervised") or supervision_profile.get("phd_main_supervisor") or 0)
        joint_cnt = float(supervision_profile.get("total_joint_papers")  or supervision_profile.get("joint_publications_count") or 0)

        total_sup = ms_cnt + (phd_cnt * 2.0)
        if total_sup >= 5 or phd_cnt >= 2:
            sup_score = 10.0
        elif total_sup >= 2 or phd_cnt >= 1:
            sup_score = 7.0
        elif total_sup >= 1 or joint_cnt >= 1:
            sup_score = 4.0
        elif data_missing in ("true", "1", "yes"):
            # Data not found in CV — cannot penalise; award neutral score
            sup_score = 5.0
        else:
            # Confirmed zero supervision records
            sup_score = 2.0

    # 4. Books & Patents Score (Max 10 pts)
    innovation_score = 2.0
    b_cnt = float(book_profile.get("total_books") or 0) if book_profile else 0
    p_cnt = float(patent_profile.get("total_patents") or 0) if patent_profile else 0

    if b_cnt >= 1 and p_cnt >= 1:
        innovation_score = 10.0
    elif p_cnt >= 1:
        innovation_score = 8.0
    elif b_cnt >= 1:
        innovation_score = 6.0

    # 5. Topic Breadth Score (Max 10 pts)
    breadth_score = 5.0
    if breadth_profile:
        div_score = float(breadth_profile.get("diversity_score") or breadth_profile.get("raw_shannon_entropy") or breadth_profile.get("shannon_entropy_diversity_score") or 0)
        themes = float(breadth_profile.get("distinct_themes_count") or 1)
        breadth_score = round(min(div_score * 8.0 + (themes * 0.5) + 2.0, 10.0), 1)

    # 6. Collaboration Network Score (Max 10 pts)
    collab_score = 5.0
    if collab_profile:
        co_cnt = float(collab_profile.get("total_unique_coauthors") or 0)
        lbl = str(collab_profile.get("collaboration_strength_label") or "").lower()
        if "broad" in lbl or co_cnt >= 15:
            collab_score = 10.0
        elif "balanced" in lbl or co_cnt >= 5:
            collab_score = 7.5
        elif co_cnt >= 1:
            collab_score = 5.0
        else:
            collab_score = 2.5

    # 7. Experience & Skill Alignment Score (Max 15 pts)
    exp_score = 5.0
    if exp_profile:
        yrs = float(exp_profile.get("total_experience_years") or 0)
        if yrs >= 10:
            exp_score = 15.0
        elif yrs >= 5:
            exp_score = 12.0
        elif yrs >= 2:
            exp_score = 8.0
        elif yrs > 0:
            exp_score = 5.0
        else:
            exp_score = 3.0

    overall_score = round(edu_score + research_score + sup_score + innovation_score + breadth_score + collab_score + exp_score, 1)
    overall_score = min(overall_score, 100.0)

    # Tier Classification
    if overall_score >= 80.0:
        candidate_tier = "Tier 1: Exceptional Fit"
    elif overall_score >= 65.0:
        candidate_tier = "Tier 2: Strong Candidate"
    elif overall_score >= 50.0:
        candidate_tier = "Tier 3: Moderate Candidate"
    else:
        candidate_tier = "Tier 4: Needs Clarification / Weak"

    return {
        "candidate_id": candidate_id,
        "overall_composite_score": overall_score,
        "candidate_tier": candidate_tier,
        "education_score": edu_score,
        "research_score": research_score,
        "supervision_score": sup_score,
        "innovation_score": innovation_score,
        "breadth_score": breadth_score,
        "collaboration_score": collab_score,
        "experience_score": exp_score,
    }


class CompositeEvaluator:
    """
    Orchestrates Part 10 for all candidates.
    Reads previous module outputs in data/analysis and computes candidate overall scores.
    """

    def __init__(self, output_dir: str = "data/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Path]:
        logger.info("Executing Module 10 Candidate Composite Evaluation...")

        # Load CSVs from analysis directory
        edu_df  = self._load_csv("educational_profiles.csv")
        res_df  = self._load_csv("research_aggregates.csv")
        sup_df  = self._load_csv("supervision_profiles.csv")
        book_df = self._load_csv("book_aggregates.csv")
        pat_df  = self._load_csv("patent_aggregates.csv")
        brd_df  = self._load_csv("research_breadth_profiles.csv")
        col_df  = self._load_csv("collaboration_profiles.csv")
        exp_df  = self._load_csv("experience_profiles.csv")

        # Map by candidate_id
        edu_map  = self._to_map(edu_df)
        res_map  = self._to_map(res_df)
        sup_map  = self._to_map(sup_df)
        book_map = self._to_map(book_df)
        pat_map  = self._to_map(pat_df)
        brd_map  = self._to_map(brd_df)
        col_map  = self._to_map(col_df)
        exp_map  = self._to_map(exp_df)

        all_candidates = sorted(list(set(
            list(edu_map.keys()) + list(res_map.keys()) + list(exp_map.keys())
        )))

        composite_results = []

        cand_map = {}
        cand_csv = Path("data/extracted/candidates.csv")
        if cand_csv.exists():
            try:
                cand_df = pd.read_csv(cand_csv, dtype=str).fillna("")
                cand_map = {str(r.get("candidate_id")): str(r.get("name")) for _, r in cand_df.iterrows() if r.get("candidate_id")}
            except Exception:
                pass

        for cid in all_candidates:
            c_score = compute_candidate_composite_score(
                candidate_id        = cid,
                edu_profile         = edu_map.get(cid),
                research_profile    = res_map.get(cid),
                supervision_profile = sup_map.get(cid),
                book_profile        = book_map.get(cid),
                patent_profile      = pat_map.get(cid),
                breadth_profile     = brd_map.get(cid),
                collab_profile      = col_map.get(cid),
                exp_profile         = exp_map.get(cid),
            )

            # Include candidate name if present
            cname = cand_map.get(cid) or (edu_map.get(cid) or {}).get("candidate_name") or (res_map.get(cid) or {}).get("candidate_name") or (exp_map.get(cid) or {}).get("candidate_name")
            if not cname or str(cname).lower() in ("nan", "none", "null", ""):
                cname = cid.replace("_", " ").title()
            c_score["candidate_name"] = str(cname)
            composite_results.append(c_score)

        # Sort results by overall_composite_score descending
        composite_results.sort(key=lambda x: x["overall_composite_score"], reverse=True)

        # Export CSV & Excel
        csv_path = self.output_dir / "composite_evaluations.csv"
        pd.DataFrame(composite_results).to_csv(csv_path, index=False)

        xlsx_path = self.output_dir / "composite_evaluations.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            pd.DataFrame(composite_results).to_excel(writer, sheet_name="Composite Evaluations", index=False)

        logger.info("Wrote %d rows -> composite_evaluations.csv", len(composite_results))

        return {
            "composite_csv": csv_path,
            "composite_xlsx": xlsx_path,
        }

    def _load_csv(self, filename: str) -> Optional[pd.DataFrame]:
        p = self.output_dir / filename
        if p.exists():
            try:
                return pd.read_csv(p, dtype=str)
            except Exception:
                pass
        return None

    @staticmethod
    def _to_map(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        if df is None or df.empty:
            return {}
        res = {}
        for _, r in df.iterrows():
            cid = str(r.get("candidate_id", "")).strip()
            if cid:
                res[cid] = r.to_dict()
        return res
