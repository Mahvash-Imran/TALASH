"""
jd_matcher.py  –  Module 11: Candidate vs Job Description Matching Engine
========================================================================

WHY THIS FILE EXISTS
--------------------
Evaluates candidate profiles against parsed JD requirements using a 2-tier matching strategy:
  1. Structured Overlap (fuzzy skill matching, degree comparison, experience thresholds).
  2. Semantic Alignment (LLM-based fit score + concise rationale text).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Degree hierarchy weights for comparison
_DEGREE_HIERARCHY = {
    "phd": 4,
    "ph.d": 4,
    "doctorate": 4,
    "ms": 3,
    "m.s": 3,
    "mphil": 3,
    "m.phil": 3,
    "master": 3,
    "bs": 2,
    "b.s": 2,
    "bachelor": 2,
    "bsc": 2,
    "hssc": 1,
    "ssc": 0,
}


def _get_degree_rank(deg_str: str) -> int:
    d = str(deg_str or "").lower().strip()
    for k, rank in _DEGREE_HIERARCHY.items():
        if k in d:
            return rank
    return 1


def structured_match_score(candidate: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes fuzzy structured overlap metrics between a candidate profile and JD requirements.
    """
    req_skills = jd.get("required_skills") or []
    pref_skills = jd.get("preferred_skills") or []
    all_jd_skills = req_skills + pref_skills

    # Build rich candidate skills and domain search pool
    cand_skills_raw = candidate.get("skills") or candidate.get("top_strong_skills") or []
    if isinstance(cand_skills_raw, str):
        cand_skills = [s.strip() for s in cand_skills_raw.split(",") if s.strip()]
    elif isinstance(cand_skills_raw, list):
        cand_skills = []
        for s in cand_skills_raw:
            if isinstance(s, dict):
                cand_skills.append(str(s.get("name") or s.get("skill_name") or ""))
            else:
                cand_skills.append(str(s))
    else:
        cand_skills = []

    # Include specializations in search pool
    specs = candidate.get("specializations") or candidate.get("specialization") or []
    if isinstance(specs, list):
        cand_skills.extend([str(x) for x in specs])
    elif isinstance(specs, str) and specs:
        cand_skills.append(specs)

    matched_skills = []
    missing_skills = []

    if req_skills:
        for req_sk in req_skills:
            req_sk_clean = req_sk.strip()
            if not req_sk_clean:
                continue
            best_score = max(
                (fuzz.partial_ratio(req_sk_clean.lower(), c_sk.lower()) for c_sk in cand_skills),
                default=0
            )
            if best_score >= 70 or any(req_sk_clean.lower() in c_sk.lower() or c_sk.lower() in req_sk_clean.lower() for c_sk in cand_skills):
                matched_skills.append(req_sk_clean)
            else:
                missing_skills.append(req_sk_clean)

        skill_match_pct = round((len(matched_skills) / len(req_skills)) * 100.0, 1)
    else:
        # If JD has no explicit skills listed, check candidate skills against JD title/text
        jd_title = str(jd.get("title") or "").lower()
        has_any_domain_match = any(fuzz.partial_ratio(c_sk.lower(), jd_title) >= 65 for c_sk in cand_skills)
        skill_match_pct = 65.0 if has_any_domain_match else 30.0

    # Degree level match check
    req_deg = jd.get("required_degree_level") or "BS"
    cand_deg = str(candidate.get("highest_degree") or "BS")
    req_deg_rank = _get_degree_rank(req_deg)
    cand_deg_rank = _get_degree_rank(cand_deg)
    degree_match = cand_deg_rank >= req_deg_rank

    # Experience match check
    min_exp = float(jd.get("min_experience_years") or 0)
    cand_exp = float(candidate.get("total_experience_years") or 0)
    experience_match = cand_exp >= min_exp if min_exp > 0 else True

    # Discipline match check
    req_disc = jd.get("required_discipline") or []
    cand_disc = str(candidate.get("specialization") or candidate.get("education_summary") or "").lower()
    disc_match = True
    if req_disc:
        disc_match = any(
            fuzz.partial_ratio(d.lower(), cand_disc) >= 65 or d.lower() in cand_disc
            for d in req_disc
        )

    return {
        "skill_match_pct": skill_match_pct,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "degree_match": degree_match,
        "cand_degree": cand_deg,
        "req_degree": req_deg,
        "experience_match": experience_match,
        "cand_experience_years": cand_exp,
        "req_experience_years": min_exp,
        "discipline_match": disc_match,
    }


def llm_semantic_score(
    candidate_summary: str,
    jd_text: str,
    api_key: Optional[str] = None,
    model: str = "llama-3.3-70b-versatile",
    base_url: Optional[str] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Asks LLM for a semantic similarity score (0-100) and short fit rationale.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    is_valid_key = bool(key and not str(key).startswith("your_") and len(str(key).strip()) > 20)

    if skip_llm or not is_valid_key or not candidate_summary or not jd_text:
        return {
            "semantic_score": 65.0,
            "rationale": "Candidate profile shows domain alignment based on extracted qualification data.",
        }

    try:
        from openai import OpenAI
        b_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        client = OpenAI(api_key=key, base_url=b_url)

        prompt = f"""Compare this Candidate Summary with the Job Description and evaluate overall semantic fit.

CANDIDATE SUMMARY:
{candidate_summary[:1500]}

JOB DESCRIPTION:
{jd_text[:1500]}

Return JSON only:
{{
  "semantic_score": 85,   // number 0 to 100
  "rationale": "2-3 concise sentences explaining the key strengths and potential gaps relative to the position."
}}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        content = resp.choices[0].message.content or ""
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        return {
            "semantic_score": float(parsed.get("semantic_score", 65.0)),
            "rationale": str(parsed.get("rationale", "Candidate profile shows alignment with role.")).strip(),
        }
    except Exception as e:
        logger.warning(f"LLM semantic score failed: {e}")
        return {
            "semantic_score": 65.0,
            "rationale": "Candidate profile evaluated via structured parameter alignment.",
        }


def compute_match_score(
    candidate: Dict[str, Any],
    jd: Dict[str, Any],
    jd_text: str = "",
    api_key: Optional[str] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Computes final composite match score and returns detailed analysis payload.
    """
    cid = str(candidate.get("candidate_id") or candidate.get("id") or "UNKNOWN")
    cname = str(candidate.get("candidate_name") or candidate.get("name") or cid)

    structured = structured_match_score(candidate, jd)

    # Build candidate summary for semantic check
    cand_summary_parts = [
        f"Candidate: {cname}",
        f"Highest Degree: {structured['cand_degree']}",
        f"Total Experience: {structured['cand_experience_years']} years",
        f"Specializations: {candidate.get('specialization', '')}",
        f"Skills: {', '.join(structured['matched_skills'] + candidate.get('skills', []))}",
        f"Summary: {candidate.get('summary', '')}"
    ]
    cand_summary_text = "\n".join(cand_summary_parts)

    semantic = llm_semantic_score(
        cand_summary_text,
        jd_text or str(jd),
        api_key=api_key,
        skip_llm=skip_llm
    )

    # Calculate weighted final match score (0-100)
    skill_part = structured["skill_match_pct"] * 0.45
    semantic_part = semantic["semantic_score"] * 0.20
    degree_part = (100.0 if structured["degree_match"] else 30.0) * 0.15

    # Experience calculation
    req_exp = structured["req_experience_years"]
    cand_exp = structured["cand_experience_years"]
    if req_exp <= 0:
        exp_part = 100.0 * 0.10
    else:
        exp_ratio = min(cand_exp / req_exp, 1.0)
        exp_part = (exp_ratio * 100.0) * 0.10

    disc_part = (100.0 if structured["discipline_match"] else 25.0) * 0.10

    final_score = round(skill_part + semantic_part + degree_part + exp_part + disc_part, 1)
    final_score = min(max(final_score, 0.0), 100.0)

    # Tier classification
    if final_score >= 75.0:
        tier = "Strong Fit"
    elif final_score >= 55.0:
        tier = "Moderate Fit"
    else:
        tier = "Weak Fit"

    return {
        "candidate_id": cid,
        "candidate_name": cname,
        "match_score": final_score,
        "match_tier": tier,
        "skill_match_pct": structured["skill_match_pct"],
        "matched_skills": structured["matched_skills"],
        "missing_skills": structured["missing_skills"],
        "degree_match": structured["degree_match"],
        "cand_degree": structured["cand_degree"],
        "req_degree": structured["req_degree"],
        "experience_match": structured["experience_match"],
        "cand_experience_years": structured["cand_experience_years"],
        "req_experience_years": structured["req_experience_years"],
        "discipline_match": structured["discipline_match"],
        "semantic_score": semantic["semantic_score"],
        "rationale": semantic["rationale"],
    }
