"""
run_jd_matching.py  –  CLI & Runner for Module 11 Job Description Matching
==========================================================================

WHY THIS FILE EXISTS
--------------------
Orchestrates JD parsing, candidate profile loading, match scoring, and persistent storage
in data/analysis/jd_matches/{jd_id}/.
"""

import argparse
import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from analysis.jd_parser import parse_job_description
from analysis.jd_matcher import compute_match_score

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/analysis")
EXTRACTED_DIR = Path("data/extracted")
JD_MATCHES_DIR = DATA_DIR / "jd_matches"


def generate_jd_id(jd_title: str) -> str:
    """Generates a unique, URL-safe slugified ID for a JD."""
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', jd_title.strip()).strip('_').lower()
    if not slug:
        slug = "jd_position"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{slug[:30]}_{timestamp}"


def load_candidate_profiles(candidate_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Loads candidate data from analysis & extracted CSVs (composite evaluations, experience, education, skills).
    Aggregates rich candidate-specific parameters for precise JD matching.
    """
    comp_file = DATA_DIR / "composite_evaluations.csv"
    if not comp_file.exists():
        logger.warning("composite_evaluations.csv not found.")
        return []

    comp_df = pd.read_csv(comp_file, dtype=str).fillna("")
    if candidate_ids:
        comp_df = comp_df[comp_df["candidate_id"].isin(candidate_ids)]

    # Load supplementary tables for skills & detailed info
    exp_df = _load_csv(DATA_DIR / "experience_profiles.csv")
    edu_df = _load_csv(DATA_DIR / "educational_profiles.csv")
    res_df = _load_csv(DATA_DIR / "research_aggregates.csv")
    skills_df = _load_csv(EXTRACTED_DIR / "skills.csv")
    sk_ev_df = _load_csv(DATA_DIR / "skill_evidence.csv")
    edu_ext_df = _load_csv(EXTRACTED_DIR / "education.csv")

    exp_map = _to_map(exp_df)
    edu_map = _to_map(edu_df)
    res_map = _to_map(res_df)

    candidates = []
    for _, r in comp_df.iterrows():
        cid = r.get("candidate_id", "").strip()
        if not cid:
            continue

        exp_info = exp_map.get(cid, {})
        edu_info = edu_map.get(cid, {})
        res_info = res_map.get(cid, {})

        # Actual total experience years
        try:
            exp_years = float(exp_info.get("total_experience_years") or r.get("total_experience_years") or 0.0)
        except (ValueError, TypeError):
            exp_years = 0.0

        # Degree & Specializations
        h_deg = edu_info.get("highest_degree") or r.get("highest_degree") or "BS"
        specs = []
        if edu_ext_df is not None and not edu_ext_df.empty:
            c_edus = edu_ext_df[edu_ext_df["candidate_id"] == cid]
            specs = [str(s).strip() for s in c_edus["specialization"].dropna().unique() if str(s).strip() and str(s).lower() != "nan"]

        # Aggregate skills
        skills_set = set()
        raw_top = exp_info.get("top_strong_skills", "")
        if raw_top and str(raw_top).lower() != "nan":
            for s in str(raw_top).split(","):
                if s.strip():
                    skills_set.add(s.strip())

        if skills_df is not None and not skills_df.empty:
            c_sks = skills_df[skills_df["candidate_id"] == cid]
            for s in c_sks["skill_name"].dropna():
                if str(s).strip() and str(s).lower() != "nan":
                    skills_set.add(str(s).strip())

        if sk_ev_df is not None and not sk_ev_df.empty:
            c_evs = sk_ev_df[sk_ev_df["candidate_id"] == cid]
            for s in c_evs["skill_name"].dropna():
                if str(s).strip() and str(s).lower() != "nan":
                    skills_set.add(str(s).strip())

        combined_skills = list(skills_set) + specs

        candidates.append({
            "candidate_id": cid,
            "candidate_name": r.get("candidate_name", cid),
            "highest_degree": h_deg,
            "total_experience_years": exp_years,
            "skills": combined_skills,
            "specialization": " ".join(specs),
            "overall_composite_score": float(r.get("overall_composite_score") or 0.0),
            "candidate_tier": r.get("candidate_tier", "Unclassified"),
            "summary": f"Degree: {h_deg}. Experience: {exp_years} years. Specializations: {', '.join(specs)}. Skills: {', '.join(list(skills_set))}. {res_info.get('research_summary', '')}",
        })

    return candidates


def evaluate_jd_against_candidates(
    jd_text: str,
    jd_id: Optional[str] = None,
    candidate_ids: Optional[List[str]] = None,
    custom_candidates: Optional[List[Dict[str, Any]]] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Main evaluation entry point. Parses JD, runs matching against candidates, and saves results.
    """
    parsed_jd = parse_job_description(jd_text, skip_llm=skip_llm)
    if not jd_id:
        jd_id = generate_jd_id(parsed_jd.get("title", "Job Position"))

    jd_dir = JD_MATCHES_DIR / jd_id
    jd_dir.mkdir(parents=True, exist_ok=True)

    # Save original JD text and parsed JSON
    (jd_dir / "jd_original.txt").write_text(jd_text, encoding="utf-8")
    (jd_dir / "jd_parsed.json").write_text(json.dumps(parsed_jd, indent=2), encoding="utf-8")

    # Load candidates
    candidates = custom_candidates if custom_candidates is not None else load_candidate_profiles(candidate_ids)

    results = []
    for cand in candidates:
        m_res = compute_match_score(cand, parsed_jd, jd_text=jd_text, skip_llm=skip_llm)
        m_res["evaluated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results.append(m_res)

    # Sort results by match_score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)

    # Save to CSV
    if results:
        res_df = pd.DataFrame(results)
        csv_df = res_df.copy()
        if "matched_skills" in csv_df.columns:
            csv_df["matched_skills"] = csv_df["matched_skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        if "missing_skills" in csv_df.columns:
            csv_df["missing_skills"] = csv_df["missing_skills"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        csv_df.to_csv(jd_dir / "results.csv", index=False)

    return {
        "jd_id": jd_id,
        "title": parsed_jd.get("title", "Position"),
        "parsed_jd": parsed_jd,
        "candidate_count": len(results),
        "results": results,
        "saved_dir": str(jd_dir),
    }


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            pass
    return None


def _to_map(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if df is None or df.empty:
        return {}
    res = {}
    for _, r in df.iterrows():
        cid = str(r.get("candidate_id", "")).strip()
        if cid:
            res[cid] = r.to_dict()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TALASH Module 11 — Job Description Matching CLI")
    parser.add_argument("--jd-file", type=str, help="Path to JD text or markdown file")
    parser.add_argument("--candidate-id", type=str, help="Single candidate ID to evaluate")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM API calls and use heuristic evaluation")
    args = parser.parse_args()

    if args.jd_file:
        jd_path = Path(args.jd_file)
        if jd_path.exists():
            text = jd_path.read_text(encoding="utf-8")
            cids = [args.candidate_id] if args.candidate_id else None
            res = evaluate_jd_against_candidates(text, candidate_ids=cids, skip_llm=args.skip_llm)
            print(f"Evaluated {res['candidate_count']} candidates against JD '{res['title']}' (ID: {res['jd_id']}).")
            print(f"Results saved to {res['saved_dir']}")
        else:
            print(f"File not found: {args.jd_file}")
    else:
        print("Please provide --jd-file path/to/jd.txt")
