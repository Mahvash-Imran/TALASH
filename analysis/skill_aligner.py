"""
skill_aligner.py  –  Module 9 Helper: Skill Extraction, Semantic Evidence Level & JD Scoring
============================================================================================

WHY THIS FILE EXISTS
--------------------
Provides semantic skill alignment logic for Module 9:
  1. Skill Extractor: Merges explicit skills (from skills.csv) with implicit technical skills extracted from experience descriptions and publication titles.
  2. Evidence Level Classifier:
     - Strong Evidence: Demonstrated in job responsibilities OR primary publication research theme.
     - Moderate Evidence: Mentioned in job title or secondary research area.
     - Unverified / Low Evidence: Self-claimed in CV skills.csv without backing experience or publication outputs.
  3. Job Description (JD) Alignment Scorer: Calculates candidate alignment score (0-100%) against a provided or target JD requirement.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from rapidfuzz import fuzz


def normalize_skill_name(skill: str) -> str:
    """Normalizes skill text (lowercasing, cleaning punctuation)."""
    if not skill:
        return ""
    s = str(skill).strip()
    s = re.sub(r"[,.\-–_]", " ", s)
    tokens = [t for t in s.split() if t]
    return " ".join(tokens).title()


def extract_and_align_skills(
    candidate_id: str,
    explicit_skills: List[Dict[str, Any]],
    experience_records: List[Dict[str, Any]],
    publication_records: List[Dict[str, Any]],
    publication_themes: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts all candidate skills and classifies their evidence level into:
      - Strong Evidence
      - Moderate Evidence
      - Unverified / Low Evidence
    """
    skill_sources: Dict[str, Dict[str, Any]] = {}

    # 1. Register explicit CV skills
    for sk in explicit_skills:
        name = str(sk.get("skill_name") or "").strip()
        cat = str(sk.get("category") or "Technical").strip()
        if name and name.lower() not in ("nan", "none", ""):
            norm_name = normalize_skill_name(name)
            if norm_name:
                skill_sources[norm_name.lower()] = {
                    "skill_name": norm_name,
                    "category": cat.title(),
                    "in_cv_skills": True,
                    "exp_matches": [],
                    "pub_matches": [],
                }

    # Helper text pools for experience and publications
    exp_titles_text = " ".join([str(e.get("job_title") or "") for e in experience_records]).lower()
    exp_desc_text = " ".join([f"{e.get('job_title') or ''} {e.get('organization') or ''} {e.get('description') or ''}" for e in experience_records]).lower()
    pub_titles_text = " ".join([str(p.get("title") or "") for p in publication_records]).lower()

    # Pre-defined domain skill keywords to mine implicitly from pubs/exp if skills.csv is sparse
    implicit_domain_keywords = [
        "Machine Learning", "Deep Learning", "Artificial Intelligence", "Computer Vision",
        "Natural Language Processing", "Wireless Networks", "IoT", "Cybersecurity",
        "Signal Processing", "Data Science", "Python", "MATLAB", "C++", "Java",
        "Image Processing", "Embedded Systems", "Power Systems", "Cloud Computing",
        "Robotics", "Control Systems", "Software Engineering", "Web Development"
    ]

    for kw in implicit_domain_keywords:
        kw_low = kw.lower()
        if kw_low in exp_desc_text or kw_low in pub_titles_text:
            if kw_low not in skill_sources:
                skill_sources[kw_low] = {
                    "skill_name": kw,
                    "category": "Domain Expertise",
                    "in_cv_skills": False,
                    "exp_matches": [],
                    "pub_matches": [],
                }

    # Evaluate evidence for each skill
    aligned_results = []

    for key, info in skill_sources.items():
        sname = info["skill_name"]
        s_low = key

        # Check matching experience records
        exp_matches = []
        for exp in experience_records:
            jtitle = str(exp.get("job_title") or "").strip()
            org = str(exp.get("organization") or "").strip()
            desc = str(exp.get("description") or "").strip()
            full_text = f"{jtitle} {org} {desc}".lower()

            if s_low in full_text or fuzz.partial_ratio(s_low, full_text) >= 85:
                exp_matches.append(f"{jtitle} @ {org}")

        # Check matching publication records
        pub_matches = []
        for pub in publication_records:
            title = str(pub.get("title") or "").strip()
            if s_low in title.lower() or fuzz.partial_ratio(s_low, title.lower()) >= 85:
                pub_matches.append(title[:60] + "...")

        # Add publication theme matches if available
        if publication_themes:
            for pt in publication_themes:
                theme = str(pt.get("primary_theme") or "").lower()
                if s_low in theme:
                    pub_matches.append(f"Primary Theme: {pt.get('primary_theme')}")

        # Classify Evidence Level
        if len(exp_matches) >= 1 and len(pub_matches) >= 1:
            evidence_level = "Strong Evidence"
            rationale = f"Validated by {len(exp_matches)} employment role(s) AND {len(pub_matches)} publication(s)."
        elif len(exp_matches) >= 2 or len(pub_matches) >= 2:
            evidence_level = "Strong Evidence"
            rationale = f"Demonstrated across multiple records ({len(exp_matches)} job(s), {len(pub_matches)} paper(s))."
        elif len(exp_matches) == 1 or len(pub_matches) == 1:
            evidence_level = "Moderate Evidence"
            rationale = f"Supported by {len(exp_matches)} job role(s) or {len(pub_matches)} publication(s)."
        else:
            evidence_level = "Unverified / Low Evidence"
            rationale = "Listed in CV skills but missing direct backing in experience descriptions or publications."

        aligned_results.append({
            "candidate_id": candidate_id,
            "skill_name": sname,
            "category": info["category"],
            "in_cv_skills": info["in_cv_skills"],
            "evidence_level": evidence_level,
            "exp_evidence_count": len(exp_matches),
            "pub_evidence_count": len(pub_matches),
            "evidence_sources": f"Jobs: [{'; '.join(exp_matches[:2])}] | Pubs: [{'; '.join(pub_matches[:2])}]",
            "rationale": rationale,
        })

    # Sort results by evidence strength
    level_rank = {"Strong Evidence": 1, "Moderate Evidence": 2, "Unverified / Low Evidence": 3}
    aligned_results.sort(key=lambda x: (level_rank.get(x["evidence_level"], 9), x["skill_name"]))

    return aligned_results


SKILL_EQUIVALENCE_MAP = {
    "artificial intelligence": ["machine learning", "deep learning", "ai", "ml", "neural networks", "data science", "computer vision", "nlp"],
    "machine learning": ["artificial intelligence", "deep learning", "ai", "ml", "neural networks", "data science", "pattern recognition"],
    "deep learning": ["machine learning", "artificial intelligence", "ai", "ml", "neural networks", "computer vision", "nlp"],
    "computer vision": ["image processing", "pattern recognition", "cv", "ai", "machine learning", "deep learning"],
    "natural language processing": ["nlp", "text mining", "large language models", "llm", "ai", "machine learning"],
    "data science": ["machine learning", "data analytics", "data analysis", "big data", "statistics", "ai"],
    "software engineering": ["software development", "programming", "system design", "coding", "software architecture", "python", "c++", "java"],
    "wireless networks": ["networking", "telecommunication", "wireless communication", "sensor networks", "iot"],
    "signal processing": ["digital signal processing", "dsp", "audio processing", "image processing"],
}


def compute_jd_alignment_score(
    skills_evidence: List[Dict[str, Any]],
    jd_requirements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes Job Description (JD) alignment score (0 to 100%).
    Default benchmark criteria: Computer Science & Academic Faculty Requirements.
    Supports skill equivalences (e.g. AI == Machine Learning).
    """
    if not jd_requirements:
        jd_requirements = [
            "Machine Learning", "Artificial Intelligence", "Deep Learning",
            "Python", "Computer Vision", "Wireless Networks", "Signal Processing",
            "Data Science", "Software Engineering", "Research", "Teaching"
        ]

    matched_strong = []
    matched_moderate = []
    missing_requirements = []

    cand_skills_map = {s["skill_name"].lower(): s for s in skills_evidence}

    for req in jd_requirements:
        req_low = req.lower().strip()
        matched = False

        # Equivalence candidates
        equiv_terms = set([req_low] + SKILL_EQUIVALENCE_MAP.get(req_low, []))
        for main_term, syns in SKILL_EQUIVALENCE_MAP.items():
            if req_low == main_term or req_low in syns:
                equiv_terms.add(main_term)
                equiv_terms.update(syns)

        for eq_term in equiv_terms:
            for sk_low, info in cand_skills_map.items():
                if eq_term in sk_low or sk_low in eq_term or fuzz.token_sort_ratio(eq_term, sk_low) >= 80:
                    matched = True
                    if info["evidence_level"] == "Strong Evidence":
                        matched_strong.append(req)
                    else:
                        matched_moderate.append(req)
                    break
            if matched:
                break

        if not matched:
            missing_requirements.append(req)

    total_reqs = len(jd_requirements)
    score = round(((len(matched_strong) * 1.0 + len(matched_moderate) * 0.5) / total_reqs) * 100, 1)

    if score >= 75:
        alignment_label = "High Alignment"
    elif score >= 45:
        alignment_label = "Moderate Alignment"
    else:
        alignment_label = "Low Alignment"

    return {
        "jd_alignment_score": score,
        "alignment_label": alignment_label,
        "matched_strong_skills": matched_strong,
        "matched_moderate_skills": matched_moderate,
        "missing_skills": missing_requirements,
    }
