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


# Bidirectional & Sub-branch Skill Equivalence Map
SKILL_EQUIVALENCE_MAP = {
    "artificial intelligence": ["machine learning", "deep learning", "ai", "ml", "neural networks", "data science", "computer vision", "nlp", "reinforcement learning", "pattern recognition"],
    "machine learning": ["artificial intelligence", "deep learning", "ai", "ml", "neural networks", "data science", "pattern recognition", "predictive modeling"],
    "deep learning": ["machine learning", "artificial intelligence", "ai", "ml", "neural networks", "computer vision", "nlp", "cnn", "rnn", "transformers"],
    "computer vision": ["image processing", "pattern recognition", "cv", "ai", "machine learning", "deep learning", "object detection"],
    "natural language processing": ["nlp", "text mining", "large language models", "llm", "ai", "machine learning", "computational linguistics"],
    "data science": ["machine learning", "data analytics", "data analysis", "big data", "statistics", "ai", "data mining"],
    "software engineering": ["software development", "programming", "system design", "coding", "software architecture", "python", "c++", "java", "oop"],
    "wireless networks": ["networking", "telecommunication", "wireless communication", "sensor networks", "iot", "5g", "mobile computing"],
    "signal processing": ["digital signal processing", "dsp", "audio processing", "image processing", "communications"],
    "cloud computing": ["aws", "azure", "gcp", "distributed systems", "devops", "docker", "kubernetes"],
    "cybersecurity": ["network security", "information security", "cryptography", "security", "ethical hacking"],
}


def _check_skill_equivalence(req_skill: str, candidate_skills: List[str]) -> bool:
    req_clean = req_skill.strip().lower()
    if not req_clean:
        return False

    # Direct fuzzy / substring check
    for c_sk in candidate_skills:
        c_clean = c_sk.strip().lower()
        if not c_clean:
            continue
        if req_clean in c_clean or c_clean in req_clean:
            return True
        if fuzz.partial_ratio(req_clean, c_clean) >= 75 or fuzz.token_sort_ratio(req_clean, c_clean) >= 80:
            return True

    # Synonyms / Sub-branches (e.g. AI <-> Machine Learning)
    equiv_terms = set(SKILL_EQUIVALENCE_MAP.get(req_clean, []))
    for main_term, syns in SKILL_EQUIVALENCE_MAP.items():
        if req_clean == main_term or req_clean in syns:
            equiv_terms.add(main_term)
            equiv_terms.update(syns)

    for eq_term in equiv_terms:
        for c_sk in candidate_skills:
            c_clean = c_sk.strip().lower()
            if not c_clean:
                continue
            if eq_term in c_clean or c_clean in eq_term:
                return True
            if fuzz.partial_ratio(eq_term, c_clean) >= 80:
                return True

    return False


def structured_match_score(candidate: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes fuzzy structured overlap metrics between a candidate profile and JD requirements.
    Incorporates skill equivalence (e.g. AI == Machine Learning) and PhD qualification boosts.
    """
    req_skills = jd.get("required_skills") or []
    pref_skills = jd.get("preferred_skills") or []

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
            if _check_skill_equivalence(req_sk_clean, cand_skills):
                matched_skills.append(req_sk_clean)
            else:
                missing_skills.append(req_sk_clean)

        skill_match_pct = round((len(matched_skills) / len(req_skills)) * 100.0, 1)
    else:
        # If JD has no explicit skills listed, check candidate skills against JD title/text
        jd_title = str(jd.get("title") or "").lower()
        has_any_domain_match = any(fuzz.partial_ratio(c_sk.lower(), jd_title) >= 65 for c_sk in cand_skills)
        skill_match_pct = 65.0 if has_any_domain_match else 30.0

    # Degree level match check & PhD detection
    req_deg = jd.get("required_degree_level") or "BS"
    cand_deg = str(candidate.get("highest_degree") or candidate.get("education") or "BS")
    req_deg_rank = _get_degree_rank(req_deg)
    cand_deg_rank = _get_degree_rank(cand_deg)
    degree_match = cand_deg_rank >= req_deg_rank
    is_phd = cand_deg_rank >= 4 or any(p in cand_deg.lower() for p in ["phd", "ph.d", "doctorate"])

    has_no_skills = (len(cand_skills) == 0) or (req_skills and len(matched_skills) == 0)

    # PhD Boost: ONLY if candidate has at least 1 matched / relevant skill!
    # If candidate has 0 skills, DO NOT grant skill floor or PhD boost.
    if is_phd and not has_no_skills:
        skill_match_pct = max(skill_match_pct, 75.0)

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
        "cand_deg_rank": cand_deg_rank,
        "is_phd": is_phd,
        "has_no_skills": has_no_skills,
        "req_degree": req_deg,
        "experience_match": experience_match,
        "cand_experience_years": cand_exp,
        "req_experience_years": min_exp,
        "discipline_match": disc_match,
    }


def generate_candidate_rationale(
    candidate: Dict[str, Any],
    structured: Dict[str, Any],
    jd: Dict[str, Any]
) -> str:
    """
    Generates a personalized, candidate-specific reasoning narrative.
    """
    cname = candidate.get("candidate_name") or candidate.get("candidate_id") or "The candidate"
    deg = structured.get("cand_degree") or "BS"
    exp = structured.get("cand_experience_years") or 0.0
    matched = structured.get("matched_skills") or []
    missing = structured.get("missing_skills") or []
    req_exp = structured.get("req_experience_years") or 0.0

    parts = []
    parts.append(f"{cname} holds a {deg} degree with {exp} years of relevant experience.")

    if matched:
        parts.append(f"Demonstrates verified skill alignment in {', '.join(matched)}.")
    else:
        parts.append("Lacks direct overlap with the specific technical skills required for this vacancy.")

    if missing:
        parts.append(f"Unmatched required skills: {', '.join(missing[:3])}.")
    else:
        parts.append("Fully satisfies all required technical skill requirements.")

    if req_exp > 0 and exp < req_exp:
        parts.append(f"Total experience ({exp} yrs) is below the required threshold of {req_exp} years.")

    return " ".join(parts)


def llm_semantic_score(
    candidate: Dict[str, Any],
    candidate_summary: str,
    jd_text: str,
    structured: Dict[str, Any],
    jd: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = "groq/compound-mini",
    base_url: Optional[str] = None,
    skip_llm: bool = False,
) -> Dict[str, Any]:
    """
    Asks LLM for a semantic similarity score (0-100) and short fit rationale.
    Falls back to dynamic candidate-specific rationale generator if LLM is skipped or unavailable.
    """
    heuristic_rat = generate_candidate_rationale(candidate, structured, jd)
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    is_valid_key = bool(key and not str(key).startswith("your_") and len(str(key).strip()) > 20)

    if skip_llm or not is_valid_key or not candidate_summary or not jd_text:
        return {
            "semantic_score": 65.0 if structured["matched_skills"] else 40.0,
            "rationale": heuristic_rat,
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
  "rationale": "2-3 concise sentences specifically explaining {candidate.get('candidate_name', 'this candidate')}'s key strengths and potential gaps for this exact position."
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
            "rationale": str(parsed.get("rationale", heuristic_rat)).strip(),
        }
    except Exception as e:
        logger.warning(f"LLM semantic score failed: {e}")
        return {
            "semantic_score": 65.0 if structured["matched_skills"] else 40.0,
            "rationale": heuristic_rat,
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
        candidate=candidate,
        candidate_summary=cand_summary_text,
        jd_text=jd_text or str(jd),
        structured=structured,
        jd=jd,
        api_key=api_key,
        skip_llm=skip_llm
    )

    # Calculate weighted final match score (0-100)
    is_phd = structured.get("is_phd", False) or structured.get("cand_deg_rank", 0) >= 4
    has_no_skills = structured.get("has_no_skills", False)

    # ZERO-SKILL PENALTY: If candidate has 0 skills or 0 matched skills, place them LAST at the bottom of the list!
    if has_no_skills:
        final_score = 10.0 if is_phd else 5.0
        tier = "Weak Fit"
        return {
            "candidate_id": cid,
            "candidate_name": cname,
            "match_score": final_score,
            "match_tier": tier,
            "skill_match_pct": 0.0,
            "matched_skills": [],
            "missing_skills": structured["missing_skills"],
            "degree_match": structured["degree_match"],
            "cand_degree": structured["cand_degree"],
            "req_degree": structured["req_degree"],
            "experience_match": structured["experience_match"],
            "cand_experience_years": structured["cand_experience_years"],
            "req_experience_years": structured["req_experience_years"],
            "discipline_match": structured["discipline_match"],
            "semantic_score": 0.0,
            "rationale": f"{cname} holds a {structured['cand_degree']} degree but has 0 relevant technical skills listed or matched for this position. Placed last due to complete lack of required technical skills.",
        }

    req_exp = structured["req_experience_years"]
    cand_exp = structured["cand_experience_years"]
    exp_score = 100.0 if req_exp <= 0 else (min(cand_exp / req_exp, 1.0) * 100.0)
    disc_score = 100.0 if structured["discipline_match"] else (50.0 if is_phd else 25.0)

    if is_phd:
        # PhD Weighting: Degree & Academic Qualifications (35%), Skills (30%), Semantic (15%), Exp (10%), Disc (10%)
        # Includes +10.0 PhD bonus points ONLY when candidate has matching skills
        skill_part = structured["skill_match_pct"] * 0.30
        degree_part = 100.0 * 0.35
        semantic_part = semantic["semantic_score"] * 0.15
        exp_part = exp_score * 0.10
        disc_part = disc_score * 0.10
        final_score = round(skill_part + degree_part + semantic_part + exp_part + disc_part + 10.0, 1)
    else:
        skill_part = structured["skill_match_pct"] * 0.40
        degree_part = (100.0 if structured["degree_match"] else 30.0) * 0.20
        semantic_part = semantic["semantic_score"] * 0.20
        exp_part = exp_score * 0.10
        disc_part = disc_score * 0.10
        final_score = round(skill_part + degree_part + semantic_part + exp_part + disc_part, 1)

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
