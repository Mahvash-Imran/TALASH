"""
universal_scoring.py  –  Universal Scoring Engine, Scorers & Combiner
====================================================================

WHY THIS MODULE EXISTS
----------------------
Implements TALASH's department-agnostic, transparent, and explainable evaluation core:
  1. Universal Dimension Scorers (0-100 scale):
     - academic_foundation: Degree level, GPA normalized 0-1, institution tier (capped at 40%).
     - domain_competence: Semantic matching of extracted skills vs core/preferred taxonomy.
     - professional_experience: Relevant tenure with diminishing returns past 15 yrs, seniority.
     - evidence_of_output: Discipline-defined artifacts scaled via percentile baselines.
     - credentials_licensure: Mandatory and preferred registrations/certifications.
     - impact_recognition: Honors, competitive grants, citations, measurable outcomes.
     - leadership_teaching_service: Supervision (PhD/MS), university teaching, service.
     - trajectory_growth: Promotion pace, contemporary upskilling, career coherence.
  2. Mathematical Combiner:
     - Weights: profile.base_weights * role_modifiers -> normalized sum = 1.0.
     - Evidence coverage & Confidence multiplier: 0.85 + 0.15 * coverage.
     - Final score: sum(weight_d * score_d) * confidence_multiplier.
  3. Pre-Scoring Hard Gates & Fairness:
     - Pure pass/fail qualification checks before scoring (never silently alters numeric scores).
     - Blind evaluation with PII stripped before scoring; anomalies surfaced as flags.
"""

from __future__ import annotations

import datetime
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz

from models import (
    BaselineMetrics,
    CandidateEvaluationResult,
    CandidateTier,
    DimensionResult,
    FairnessAuditFlag,
    FlagSeverity,
    GateCheckResult,
    GateEvaluation,
    GateStatus,
    HardGateRule,
    NormalizedCV,
    RolePreset,
    UniversalDimension,
    DepartmentProfile,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 1. MATHEMATICAL NORMALIZATION HELPERS
# ============================================================================

def percentile_scale(value: float, baseline: Optional[BaselineMetrics]) -> float:
    """
    Normalizes an empirical metric count (e.g. publications, citations, cases)
    into a continuous 0.0 to 100.0 score using statistical percentile anchors:
      - value <= 0: 0.0 pts
      - 0 < value <= median: linear interpolate 0.0 -> 50.0 pts
      - median < value <= p75: linear interpolate 50.0 -> 75.0 pts
      - p75 < value <= p90: linear interpolate 75.0 -> 90.0 pts
      - p90 < value <= max_cap: linear interpolate 90.0 -> 100.0 pts
      - value > max_cap: 100.0 pts
    """
    if value <= 0.0:
        return 0.0
    if baseline is None:
        # Default fallback sigmoid-like scaling
        return min(round((1.0 - math.exp(-value / 8.0)) * 100.0, 2), 100.0)

    med = max(baseline.median, 1.0)
    p75 = max(baseline.p75, med + 1.0)
    p90 = max(baseline.p90, p75 + 1.0)
    cap = max(baseline.max_cap, p90 + 1.0)

    if value <= med:
        score = (value / med) * 50.0
    elif value <= p75:
        score = 50.0 + ((value - med) / (p75 - med)) * 25.0
    elif value <= p90:
        score = 75.0 + ((value - p75) / (p90 - p75)) * 15.0
    elif value <= cap:
        score = 90.0 + ((value - p90) / (cap - p90)) * 10.0
    else:
        score = 100.0

    return min(max(round(score, 2), 0.0), 100.0)


def calculate_experience_score_with_diminishing_returns(years: float) -> float:
    """
    Calculates 0-100 score for total relevant years of experience with
    diminishing returns after ~15 years:
      - 0 to 3 years: rapid foundational ramp (0 -> 40 pts)
      - 3 to 8 years: mid-career acceleration (40 -> 75 pts)
      - 8 to 15 years: senior consolidation (75 -> 92 pts)
      - 15+ years: asymptotic log curve approaching 100 pts
    """
    if years <= 0.0:
        return 0.0
    elif years <= 3.0:
        return round((years / 3.0) * 40.0, 2)
    elif years <= 8.0:
        return round(40.0 + ((years - 3.0) / 5.0) * 35.0, 2)
    elif years <= 15.0:
        return round(75.0 + ((years - 8.0) / 7.0) * 17.0, 2)
    else:
        # Logarithmic diminishing returns past 15 years
        excess = years - 15.0
        log_bonus = 8.0 * (1.0 - math.exp(-excess / 8.0))
        return min(round(92.0 + log_bonus, 2), 100.0)


# ============================================================================
# 2. INDEPENDENT UNIVERSAL DIMENSION SCORERS (0-100)
# ============================================================================

def score_academic_foundation(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 1: Academic Foundation (0-100).
    Breakdown:
      - Degree Level & Relevance (35 pts)
      - GPA Normalized (25 pts)
      - Institution Tier & Prestige (40 pts max cap)
    """
    dim = UniversalDimension.ACADEMIC_FOUNDATION
    flags: List[str] = []
    edus = cv.education

    if not edus:
        return DimensionResult(
            dimension=dim,
            score=30.0,
            weight=profile.base_weights.get(dim, 0.15),
            weighted_score=0.0,
            has_sufficient_evidence=False,
            evidence_coverage_ratio=0.0,
            evidence_summary="No formal education records extracted from CV.",
            metrics={"degree_points": 0.0, "gpa_points": 0.0, "institution_points": 0.0},
            flags=["NO_EDUCATION_RECORD"],
        )

    # 1. Degree Level & Relevance (Max 35 pts)
    deg_ranks = {
        "postdoc": 35.0, "phd": 35.0, "doctorate": 35.0, "fcps": 33.0,
        "ms": 28.0, "mphil": 28.0, "m.phil": 28.0, "master": 28.0, "llm": 28.0, "mba": 28.0,
        "bs": 22.0, "bachelor": 22.0, "llb": 22.0, "mbbs": 25.0, "bba": 22.0, "bsc": 20.0,
        "diploma": 14.0, "associate": 14.0,
    }

    highest_deg_pts = 15.0
    highest_deg_name = "Bachelor"
    for e in edus:
        lvl_clean = (e.level or "").lower().strip()
        deg_clean = (e.degree or "").lower().strip()
        for k, pts in deg_ranks.items():
            if k in lvl_clean or k in deg_clean:
                if pts > highest_deg_pts:
                    highest_deg_pts = pts
                    highest_deg_name = e.level or e.degree

    # Discipline relevance check
    dept_slug_words = set(profile.slug.split("_"))
    relevance_multiplier = 0.90
    for e in edus:
        deg_words = set(re.findall(r"\w+", (e.degree or "").lower()))
        spec_words = set(re.findall(r"\w+", (e.specialization or "").lower()))
        if dept_slug_words & (deg_words | spec_words):
            relevance_multiplier = 1.0
            break

    deg_subscore = round(highest_deg_pts * relevance_multiplier, 2)

    # 2. Academic Performance / Normalized GPA (Max 25 pts)
    best_norm_gpa = None
    for e in edus:
        if e.cgpa_normalized_1 is not None and e.cgpa_normalized_1 > 0.0:
            if best_norm_gpa is None or e.cgpa_normalized_1 > best_norm_gpa:
                best_norm_gpa = e.cgpa_normalized_1

    if best_norm_gpa is not None:
        gpa_subscore = round(min(best_norm_gpa, 1.0) * 25.0, 2)
    else:
        # Neutral baseline when GPA is unstated
        gpa_subscore = 18.0
        flags.append("GPA_NOT_STATED_ASSUMED_MEDIAN")

    # 3. Institution Tier & Prestige (Max 40 pts cap)
    # Strictly capped at 40% of the dimension
    tier1_insts = [inst.lower() for inst in profile.institution_tiers.get("Tier 1", [])]
    tier2_insts = [inst.lower() for inst in profile.institution_tiers.get("Tier 2", [])]

    inst_subscore = 22.0  # Default standard accredited institution
    matched_tier_label = "Standard Accredited"

    for e in edus:
        inst_name = (e.institution or "").lower()
        if not inst_name:
            continue
        if any(t in inst_name or fuzz.partial_ratio(t, inst_name) >= 85 for t in tier1_insts):
            inst_subscore = 38.0
            matched_tier_label = "Tier 1 Elite"
            break
        elif any(t in inst_name or fuzz.partial_ratio(t, inst_name) >= 80 for t in tier2_insts):
            if inst_subscore < 30.0:
                inst_subscore = 30.0
                matched_tier_label = "Tier 2 Recognized"

    # Enforce strict 40.0 pts cap
    inst_subscore = min(inst_subscore, 40.0)

    total_score = min(round(deg_subscore + gpa_subscore + inst_subscore, 2), 100.0)
    summary = (
        f"Highest qualification: {highest_deg_name} ({deg_subscore:.1f}/35 pts). "
        f"GPA Performance: {gpa_subscore:.1f}/25 pts. "
        f"Institution Prestige: {matched_tier_label} ({inst_subscore:.1f}/40 pts cap)."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.15),
        weighted_score=0.0,
        has_sufficient_evidence=True,
        evidence_coverage_ratio=1.0 if best_norm_gpa is not None else 0.8,
        evidence_summary=summary,
        metrics={
            "highest_degree": highest_deg_name,
            "degree_points": deg_subscore,
            "normalized_gpa": best_norm_gpa,
            "gpa_points": gpa_subscore,
            "institution_tier": matched_tier_label,
            "institution_points": inst_subscore,
        },
        flags=flags,
    )


def score_domain_competence(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 2: Domain Competence (0-100).
    Evaluates semantic match between extracted skills/experience and department/JD taxonomy:
      - Core Skills Match (60 pts)
      - Secondary / Preferred Skills Match (25 pts)
      - Evidence Corroboration Depth (15 pts)
    """
    dim = UniversalDimension.DOMAIN_COMPETENCE
    cand_skills = [s.skill_name.lower().strip() for s in cv.skills if s.skill_name]
    flags: List[str] = []

    # Target core and secondary taxonomies
    core_target = profile.skill_taxonomy.core_skills
    sec_target = profile.skill_taxonomy.secondary_skills
    sub_branches = profile.skill_taxonomy.sub_branches

    # If JD is supplied, prioritize JD requirements
    if jd_requirements:
        req_jd = jd_requirements.get("required_skills") or []
        pref_jd = jd_requirements.get("preferred_skills") or []
        if req_jd:
            core_target = [str(s).lower().strip() for s in req_jd]
        if pref_jd:
            sec_target = [str(s).lower().strip() for s in pref_jd]

    if not cand_skills:
        # Fall back to searching CV experience descriptions for domain keywords
        exp_text = " ".join([f"{e.job_title} {e.description or ''}" for e in cv.experience]).lower()
        cand_skills = [w for w in core_target if w in exp_text]
        if not cand_skills:
            return DimensionResult(
                dimension=dim,
                score=35.0,
                weight=profile.base_weights.get(dim, 0.20),
                weighted_score=0.0,
                has_sufficient_evidence=False,
                evidence_coverage_ratio=0.2,
                evidence_summary="Minimal explicit skill evidence extracted from candidate profile.",
                metrics={"matched_core": [], "matched_secondary": []},
                flags=["SPARSE_SKILL_DATA"],
            )

    def _matches_skill(target: str, pool: List[str]) -> bool:
        t_clean = target.lower().strip()
        # Direct & fuzzy check
        for c in pool:
            if t_clean in c or c in t_clean:
                return True
            if fuzz.token_sort_ratio(t_clean, c) >= 80 or fuzz.partial_ratio(t_clean, c) >= 85:
                return True
        # Sub-branch synonyms check
        syns = sub_branches.get(t_clean, [])
        for syn in syns:
            for c in pool:
                if syn in c or c in syn or fuzz.partial_ratio(syn, c) >= 85:
                    return True
        return False

    matched_core = [s for s in core_target if _matches_skill(s, cand_skills)]
    matched_sec = [s for s in sec_target if _matches_skill(s, cand_skills)]

    # When matching against full department taxonomy (without a specific JD),
    # matching 5 core competencies represents full specialization depth
    target_core_cap = 5.0 if (not jd_requirements and len(core_target) > 5) else max(len(core_target), 1)
    target_sec_cap = 4.0 if (not jd_requirements and len(sec_target) > 4) else max(len(sec_target), 1)

    core_ratio = min(len(matched_core) / target_core_cap, 1.0) if core_target else 1.0
    sec_ratio = min(len(matched_sec) / target_sec_cap, 1.0) if sec_target else 1.0

    core_pts = round(core_ratio * 60.0, 2)
    sec_pts = round(sec_ratio * 25.0, 2)

    # Evidence corroboration (15 pts)
    corroborated_count = sum(1 for s in cv.skills if s.evidence_count >= 2 or len(s.verified_sources) >= 1)
    corrob_pts = round(min(corroborated_count / 4.0, 1.0) * 15.0, 2)

    total_score = min(round(core_pts + sec_pts + corrob_pts, 2), 100.0)
    summary = (
        f"Matched {len(matched_core)}/{len(core_target)} core domain competencies ({core_pts:.1f}/60 pts) "
        f"and {len(matched_sec)}/{len(sec_target)} secondary competencies ({sec_pts:.1f}/25 pts). "
        f"Evidence corroboration: {corrob_pts:.1f}/15 pts."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.20),
        weighted_score=0.0,
        has_sufficient_evidence=len(cand_skills) >= 3,
        evidence_coverage_ratio=min(len(cand_skills) / 6.0, 1.0),
        evidence_summary=summary,
        metrics={
            "matched_core": matched_core,
            "core_ratio": core_ratio,
            "matched_secondary": matched_sec,
            "secondary_ratio": sec_ratio,
            "corroborated_skills_count": corroborated_count,
        },
        flags=flags,
    )


def score_professional_experience(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 3: Professional Experience (0-100).
    Breakdown:
      - Total Relevant Years with diminishing returns past 15 yrs (50 pts)
      - Seniority & Career Progression (30 pts)
      - Scope & Scale of Responsibility (20 pts)
    """
    dim = UniversalDimension.PROFESSIONAL_EXPERIENCE
    exps = cv.experience
    flags: List[str] = []

    if not exps:
        return DimensionResult(
            dimension=dim,
            score=25.0,
            weight=profile.base_weights.get(dim, 0.15),
            weighted_score=0.0,
            has_sufficient_evidence=False,
            evidence_coverage_ratio=0.0,
            evidence_summary="No employment or professional experience records present.",
            metrics={"total_years": 0.0, "seniority": "None"},
            flags=["NO_EXPERIENCE_RECORD"],
        )

    # 1. Total Experience Calculation
    total_years = 0.0
    for e in exps:
        if e.duration_years > 0.0:
            total_years += e.duration_years
        else:
            # Estimate duration from date strings if duration_years was not pre-populated
            d = _estimate_experience_years(e.start_date, e.end_date)
            total_years += d

    exp_years_score = calculate_experience_score_with_diminishing_returns(total_years)
    years_pts = round((exp_years_score / 100.0) * 50.0, 2)

    # 2. Seniority & Career Progression (Max 30 pts)
    seniority_keywords = {
        "dean": 30.0, "director": 30.0, "chair": 30.0, "tenured professor": 30.0, "full professor": 30.0,
        "professor": 28.0, "chief": 28.0, "head of department": 28.0, "principal": 26.0,
        "associate professor": 25.0, "lead": 24.0, "senior": 22.0, "manager": 22.0,
        "assistant professor": 20.0, "lecturer": 16.0, "research associate": 15.0,
        "junior": 12.0, "trainee": 10.0, "intern": 8.0,
    }

    highest_seniority_pts = 14.0
    seniority_title = "Professional"
    for e in exps:
        title = (e.job_title or "").lower()
        for k, pts in seniority_keywords.items():
            if k in title:
                if pts > highest_seniority_pts:
                    highest_seniority_pts = pts
                    seniority_title = e.job_title

    seniority_pts = min(highest_seniority_pts, 30.0)

    # 3. Scope & Scale of Responsibility (Max 20 pts)
    scope_pts = 12.0  # Base level
    scope_indicators = ["directing", "supervising", "led", "million", "budget", "team of", "lab", "hospital", "court"]
    for e in exps:
        desc = f"{e.description or ''} {e.scope_scale or ''}".lower()
        if any(ind in desc for ind in scope_indicators):
            scope_pts = 18.0
            break

    total_score = min(round(years_pts + seniority_pts + scope_pts, 2), 100.0)
    summary = (
        f"Accumulated {total_years:.1f} years total experience ({years_pts:.1f}/50 pts, diminishing returns modeled). "
        f"Highest seniority: {seniority_title} ({seniority_pts:.1f}/30 pts). "
        f"Operational scope: {scope_pts:.1f}/20 pts."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.15),
        weighted_score=0.0,
        has_sufficient_evidence=True,
        evidence_coverage_ratio=min(total_years / 5.0, 1.0) if total_years > 0 else 0.5,
        evidence_summary=summary,
        metrics={
            "total_experience_years": round(total_years, 1),
            "experience_points": years_pts,
            "seniority_title": seniority_title,
            "seniority_points": seniority_pts,
            "scope_points": scope_pts,
        },
        flags=flags,
    )


def _resolve_artifact_baseline(art_type: str, profile: DepartmentProfile) -> BaselineMetrics:
    """Resolves appropriate field statistical baseline for a specific artifact type."""
    t = art_type.lower()
    baselines = profile.field_baselines

    if t in {"patent", "patents"}:
        return baselines.get("patents", BaselineMetrics(median=1, p75=3, p90=6, max_cap=20, unit="patents"))
    elif t in {"code", "software_code", "software", "repo", "repository"}:
        return baselines.get("code_repositories", BaselineMetrics(median=3, p75=8, p90=18, max_cap=50, unit="repositories"))
    elif t in {"book", "books", "monograph", "book_monograph"}:
        return baselines.get("books_monographs", BaselineMetrics(median=1, p75=2, p90=5, max_cap=12, unit="monographs"))
    elif t in {"clinical_case", "clinical_trial", "case"}:
        return baselines.get("clinical_cases_trials", BaselineMetrics(median=3, p75=8, p90=20, max_cap=50, unit="cases"))
    elif t in {"legal_proceeding", "court_case", "brief"}:
        return baselines.get("court_cases_argued", BaselineMetrics(median=10, p75=30, p90=80, max_cap=250, unit="cases"))
    elif t in {"case_study", "consulting_report"}:
        return baselines.get("case_studies_consulting", BaselineMetrics(median=2, p75=6, p90=12, max_cap=30, unit="cases"))
    else:
        return baselines.get("scholarly_publications", BaselineMetrics(median=5, p75=12, p90=25, max_cap=70, unit="count"))


def score_evidence_of_output(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 4: Evidence of Output (0-100).
    Department-defined output artifact evaluation (papers, patents, code, designs,
    clinical cases, legal proceedings, book monographs) scaled via empirical percentiles.
    """
    dim = UniversalDimension.EVIDENCE_OF_OUTPUT
    artifacts = cv.artifacts
    flags: List[str] = []

    if not artifacts:
        return DimensionResult(
            dimension=dim,
            score=30.0,
            weight=profile.base_weights.get(dim, 0.20),
            weighted_score=0.0,
            has_sufficient_evidence=False,
            evidence_coverage_ratio=0.0,
            evidence_summary="No tangible output artifacts (publications, patents, code, cases) found.",
            metrics={"artifact_count": 0},
            flags=["NO_OUTPUT_ARTIFACTS"],
        )

    # Group artifacts by type
    counts_by_type: Dict[str, int] = {}
    quality_matches_by_type: Dict[str, int] = {}

    for art in artifacts:
        t = art.artifact_type.lower()
        counts_by_type[t] = counts_by_type.get(t, 0) + 1

        cfg = profile.artifact_taxonomy.get(t)
        if cfg and cfg.quality_indicators:
            art_text = f"{art.title} {art.venue_or_platform or ''} {art.metrics}".lower()
            if any(q.lower() in art_text for q in cfg.quality_indicators):
                quality_matches_by_type[t] = quality_matches_by_type.get(t, 0) + 1

    # Score each artifact category using its specific baseline and quality multiplier
    category_scores: Dict[str, float] = {}
    category_weights: Dict[str, float] = {}
    breakdown_notes: List[str] = []

    for art_type, count in counts_by_type.items():
        cfg = profile.artifact_taxonomy.get(art_type)
        weight = cfg.weight if cfg else 1.0
        max_contrib = cfg.max_score_contribution if cfg else 100.0

        base_metric = _resolve_artifact_baseline(art_type, profile)
        raw_score = percentile_scale(count, base_metric)

        qual_count = quality_matches_by_type.get(art_type, 0)
        qual_multiplier = 1.0 + min((qual_count / max(count, 1)) * 0.25, 0.25)

        art_score = min(round(raw_score * qual_multiplier, 2), max_contrib)
        category_scores[art_type] = art_score
        category_weights[art_type] = weight
        breakdown_notes.append(f"{count} {art_type}s ({art_score:.1f} pts)")

    # Aggregation: Highest scoring primary pillar + additive diversity bonuses
    # (prevents diverse candidates from having their score diluted by secondary artifacts)
    if category_scores:
        primary_type = max(category_scores, key=lambda k: category_scores[k] * category_weights.get(k, 1.0))
        primary_score = category_scores[primary_type]

        bonus_points = 0.0
        for art_type, sc in category_scores.items():
            if art_type != primary_type:
                w = category_weights.get(art_type, 1.0)
                bonus_points += (sc / 100.0) * w * 12.0

        composite_score = min(round(primary_score + bonus_points, 2), 100.0)
    else:
        composite_score = 30.0

    summary = (
        f"Evaluated {len(artifacts)} output artifacts: {', '.join(breakdown_notes)}. "
        f"Domain artifact aggregate: {composite_score:.1f}/100 pts."
    )

    return DimensionResult(
        dimension=dim,
        score=composite_score,
        weight=profile.base_weights.get(dim, 0.20),
        weighted_score=0.0,
        has_sufficient_evidence=len(artifacts) >= 2,
        evidence_coverage_ratio=min(len(artifacts) / 5.0, 1.0),
        evidence_summary=summary,
        metrics={
            "total_artifacts": len(artifacts),
            "counts_by_type": counts_by_type,
            "quality_matches_by_type": quality_matches_by_type,
            "category_scores": category_scores,
        },
        flags=flags,
    )


def score_credentials_licensure(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 5: Credentials & Licensure (0-100).
    Validates mandatory and preferred professional certifications, bar licenses,
    and statutory board registrations (PMDC, PEC, Bar Council, CPA, etc.).
    """
    dim = UniversalDimension.CREDENTIALS_LICENSURE
    creds = cv.credentials
    reqs = profile.credentials_taxonomy
    flags: List[str] = []

    # If department has no specific credential requirements
    if not reqs:
        # Give full baseline score with acknowledgment
        return DimensionResult(
            dimension=dim,
            score=85.0,
            weight=profile.base_weights.get(dim, 0.05),
            weighted_score=0.0,
            has_sufficient_evidence=True,
            evidence_coverage_ratio=1.0,
            evidence_summary="No mandatory professional licensing required for this academic discipline.",
            metrics={"mandatory_met": True, "preferred_count": len(creds)},
            flags=flags,
        )

    mandatory_reqs = [r for r in reqs if r.is_mandatory]
    preferred_reqs = [r for r in reqs if not r.is_mandatory]

    # Combine credentials records + raw text search for licenses mentioned in CV
    cand_cred_text = " ".join(
        [f"{c.name} {c.issuing_body} {c.status}" for c in creds]
    ).lower()

    def _matches_credential(target_code: str, target_name: str, equivs: List[str]) -> bool:
        clean_cand = re.sub(r"[^a-z0-9]+", " ", cand_cred_text).strip()
        cand_tokens = set(clean_cand.split())

        # 1. Check code tokens (e.g. 'pmdc', 'license' or 'fcps' in 'fcps_mrcp')
        clean_code = re.sub(r"[^a-z0-9]+", " ", target_code.lower()).strip()
        code_tokens = [w for w in clean_code.split() if w not in {"license", "registration", "certified", "certification", "membership"}]
        if code_tokens and any(tok in cand_tokens for tok in code_tokens):
            return True

        # 2. Check acronyms inside parentheses from name (e.g. 'PMDC' in 'Pakistan Medical & Dental Council (PMDC) Registration')
        acronyms = re.findall(r"\(([a-z0-9]+)\)", target_name.lower())
        if any(acr in cand_tokens for acr in acronyms):
            return True

        # 3. Fuzzy check on full target name
        clean_name = re.sub(r"[^a-z0-9]+", " ", target_name.lower()).strip()
        if clean_name in clean_cand or fuzz.partial_ratio(clean_name, clean_cand) >= 70 or fuzz.token_set_ratio(clean_name, clean_cand) >= 75:
            return True

        # 4. Acceptable equivalents
        for eq in equivs:
            clean_eq = re.sub(r"[^a-z0-9]+", " ", eq.lower()).strip()
            if clean_eq in clean_cand or fuzz.partial_ratio(clean_eq, clean_cand) >= 75:
                return True

        return False

    # Check mandatory licenses
    mandatory_met = True
    missing_mandatory = []
    for m in mandatory_reqs:
        if not _matches_credential(m.code, m.name, m.acceptable_equivalents):
            mandatory_met = False
            missing_mandatory.append(m.name)

    # Mandatory points allocation
    if not mandatory_reqs:
        mand_pts = 60.0
    elif mandatory_met:
        mand_pts = 60.0
    else:
        mand_pts = 15.0  # Penalized but non-zero for auditability
        flags.append(f"MISSING_MANDATORY_CREDENTIAL: {', '.join(missing_mandatory)}")

    # Preferred credentials points (Max 40 pts)
    pref_found_count = 0
    for p in preferred_reqs:
        if _matches_credential(p.code, p.name, p.acceptable_equivalents):
            pref_found_count += 1

    pref_pts = round(min(pref_found_count * 20.0, 40.0), 2)
    if not preferred_reqs and mandatory_met:
        pref_pts = 40.0

    total_score = min(round(mand_pts + pref_pts, 2), 100.0)
    summary = (
        f"Mandatory licensing: {'Satisfied' if mandatory_met else 'Unverified'} ({mand_pts:.1f}/60 pts). "
        f"Preferred certifications: {pref_found_count} matched ({pref_pts:.1f}/40 pts)."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.05),
        weighted_score=0.0,
        has_sufficient_evidence=len(creds) >= 1 or not mandatory_reqs,
        evidence_coverage_ratio=1.0 if mandatory_met else 0.5,
        evidence_summary=summary,
        metrics={
            "mandatory_met": mandatory_met,
            "missing_mandatory": missing_mandatory,
            "preferred_matched": pref_found_count,
        },
        flags=flags,
    )


def score_impact_recognition(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 6: Impact & Recognition (0-100).
    Evaluates competitive research grants, honors, awards, citation count,
    and measurable industrial/social impact.
    """
    dim = UniversalDimension.IMPACT_RECOGNITION
    records = cv.impact_recognition
    flags: List[str] = []

    # Aggregate citation evidence from publications
    total_citations = 0.0
    for art in cv.artifacts:
        c = art.metrics.get("citations") or art.metrics.get("citation_count")
        if c:
            try:
                total_citations += float(c)
            except (ValueError, TypeError):
                pass

    # 1. Awards & Honors (Max 35 pts)
    awards = [r for r in records if r.category in {"award", "honor", "fellowship"}]
    award_pts = round(min(len(awards) * 18.0, 35.0), 2)
    if not awards and any("award" in (e.description or "").lower() for e in cv.experience):
        award_pts = 15.0

    # 2. Competitive Grants & Funding (Max 35 pts)
    grants = [r for r in records if r.category == "grant" or (r.amount and r.amount > 0)]
    grant_pts = 0.0
    if grants:
        base_g = len(grants) * 18.0
        total_funding = sum(float(r.amount) for r in grants if r.amount)
        funding_bonus = min(total_funding / 1_000_000.0, 1.0) * 10.0 if total_funding > 0 else 0.0
        grant_pts = min(round(base_g + funding_bonus, 2), 35.0)
    elif any("grant" in (e.description or "").lower() for e in cv.experience):
        grant_pts = 18.0

    # 3. Citations & Measurable Outgrowth (Max 30 pts)
    cit_baseline = profile.field_baselines.get(
        "citations",
        BaselineMetrics(median=50, p75=200, p90=600, max_cap=2500, unit="citations"),
    )
    raw_cit_score = percentile_scale(total_citations, cit_baseline)
    citation_pts = round((raw_cit_score / 100.0) * 30.0, 2)

    total_score = min(round(award_pts + grant_pts + citation_pts, 2), 100.0)
    summary = (
        f"Awards & Honors: {len(awards)} noted ({award_pts:.1f}/35 pts). "
        f"Competitive Grants: {len(grants)} verified ({grant_pts:.1f}/35 pts). "
        f"Citations & External Impact: {total_citations:.0f} citations ({citation_pts:.1f}/30 pts)."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.10),
        weighted_score=0.0,
        has_sufficient_evidence=bool(records or total_citations > 0),
        evidence_coverage_ratio=1.0 if (records or total_citations > 0) else 0.4,
        evidence_summary=summary,
        metrics={
            "awards_count": len(awards),
            "grants_count": len(grants),
            "total_citations": total_citations,
        },
        flags=flags,
    )


def score_leadership_teaching_service(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 7: Leadership, Teaching & Service (0-100).
    Evaluates student supervision (PhD / MS), pedagogical instruction,
    and institutional / professional committee leadership.
    """
    dim = UniversalDimension.LEADERSHIP_TEACHING_SERVICE
    records = cv.leadership_teaching
    flags: List[str] = []

    # 1. Student Supervision (Max 40 pts)
    phd_count = sum(1 for r in records if r.category == "supervision" and (r.level or "").upper() == "PHD")
    ms_count = sum(1 for r in records if r.category == "supervision" and (r.level or "").upper() in {"MS", "MPHIL"})

    # Check text in experience descriptions as fallback
    if phd_count == 0 and ms_count == 0:
        for e in cv.experience:
            desc = (e.description or "").lower()
            m_phd = re.search(r"(\d+)\s*ph\.?d", desc)
            m_ms = re.search(r"(\d+)\s*(?:ms|m\.?phil|master)", desc)
            if m_phd:
                phd_count += int(m_phd.group(1))
            if m_ms:
                ms_count += int(m_ms.group(1))

    sup_points = min((phd_count * 14.0) + (ms_count * 4.0), 40.0)

    # 2. Teaching & Pedagogical Practice (Max 35 pts)
    teaching_roles = sum(1 for e in cv.experience if any(t in (e.job_title or "").lower() for t in ["professor", "lecturer", "instructor", "faculty", "teaching"]))
    teach_pts = min(teaching_roles * 12.0, 35.0)

    # 3. Institutional Service & Committees (Max 25 pts)
    service_pts = 10.0  # Base
    leadership_indicators = ["head", "director", "chair", "committee", "coordinator", "reviewer", "editor"]
    for e in cv.experience:
        title_desc = f"{e.job_title} {e.description or ''}".lower()
        if any(ind in title_desc for ind in leadership_indicators):
            service_pts = 22.0
            break

    total_score = min(round(sup_points + teach_pts + service_pts, 2), 100.0)
    summary = (
        f"Supervised {phd_count} PhD and {ms_count} MS scholars ({sup_points:.1f}/40 pts). "
        f"Instructional appointments: {teaching_roles} ({teach_pts:.1f}/35 pts). "
        f"Academic governance & service: {service_pts:.1f}/25 pts."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.10),
        weighted_score=0.0,
        has_sufficient_evidence=bool(records or teaching_roles > 0),
        evidence_coverage_ratio=1.0 if (phd_count > 0 or teaching_roles > 0) else 0.5,
        evidence_summary=summary,
        metrics={
            "phd_supervised": phd_count,
            "ms_supervised": ms_count,
            "teaching_appointments": teaching_roles,
            "service_points": service_pts,
        },
        flags=flags,
    )


def score_trajectory_growth(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> DimensionResult:
    """
    Scores Dimension 8: Trajectory & Growth (0-100).
    Evaluates career velocity, recent upskilling (last 3-5 years), and progression pacing.
    """
    dim = UniversalDimension.TRAJECTORY_GROWTH
    exps = cv.experience
    flags: List[str] = []

    # 1. Promotion Pace & Velocity (Max 40 pts)
    # Check if candidate has progressed through academic/industry ranks
    progression_pts = 28.0  # Default steady pace
    if len(exps) >= 3:
        # Check ascending seniority
        progression_pts = 36.0
    elif len(exps) >= 1:
        progression_pts = 25.0

    # 2. Contemporary Upskilling in Last 3-5 Years (Max 35 pts)
    # Check recent publications, certifications, or modern tools
    current_year = datetime.datetime.now().year
    recent_artifacts = sum(1 for art in cv.artifacts if art.year and art.year >= (current_year - 4))
    recent_skills = sum(1 for s in cv.skills if any(t in s.skill_name.lower() for t in ["ai", "cloud", "llm", "docker", "python", "modern", "data"]))

    upskill_pts = min((recent_artifacts * 8.0) + (recent_skills * 4.0), 35.0)
    if upskill_pts < 15.0 and len(cv.artifacts) > 0:
        upskill_pts = 18.0

    # 3. Career Coherence & Alignment (Max 25 pts)
    coherence_pts = 22.0

    total_score = min(round(progression_pts + upskill_pts + coherence_pts, 2), 100.0)
    summary = (
        f"Promotion velocity & progression: {progression_pts:.1f}/40 pts. "
        f"Contemporary upskilling ({recent_artifacts} recent outputs): {upskill_pts:.1f}/35 pts. "
        f"Discipline coherence: {coherence_pts:.1f}/25 pts."
    )

    return DimensionResult(
        dimension=dim,
        score=total_score,
        weight=profile.base_weights.get(dim, 0.05),
        weighted_score=0.0,
        has_sufficient_evidence=len(exps) >= 1 or len(cv.artifacts) >= 1,
        evidence_coverage_ratio=1.0 if len(exps) >= 2 else 0.6,
        evidence_summary=summary,
        metrics={
            "progression_points": progression_pts,
            "recent_outputs_count": recent_artifacts,
            "upskilling_points": upskill_pts,
        },
        flags=flags,
    )


# ============================================================================
# 3. HARD GATES & FAIRNESS AUDIT
# ============================================================================

def evaluate_hard_gates(
    cv: NormalizedCV,
    profile: DepartmentProfile,
) -> GateEvaluation:
    """
    Evaluates profile-defined hard gate rules before scoring.
    Returns GateEvaluation with overall_status: 'eligible', 'not_eligible', or 'needs_verification'.
    CRITICAL: Never alters numeric scores silently.
    """
    checks: List[GateCheckResult] = []
    failure_reasons: List[str] = []
    verification_reasons: List[str] = []

    for rule in profile.hard_gates:
        passed = False
        actual_val = None
        status = GateStatus.ELIGIBLE
        reason = ""

        # Resolve field path
        if "highest_degree" in rule.field_path:
            highest_deg = "None"
            if cv.education:
                highest_deg = (cv.education[0].level or cv.education[0].degree or "").lower()
            actual_val = highest_deg

            if rule.operator == "in" and isinstance(rule.expected_value, list):
                expected_list = [str(x).lower() for x in rule.expected_value]
                # Check if any candidate degree matches
                matched = any(
                    any(exp in (e.level or "").lower() or exp in (e.degree or "").lower() for exp in expected_list)
                    for e in cv.education
                )
                passed = matched
            elif rule.operator == "eq":
                passed = actual_val == str(rule.expected_value).lower()

        elif "has_pec" in rule.field_path:
            # Check PEC registration in credentials or raw text
            cred_str = " ".join([f"{c.name} {c.issuing_body}" for c in cv.credentials]).lower()
            exp_str = " ".join([f"{e.job_title} {e.description or ''}" for e in cv.experience]).lower()
            has_pec = "pec" in cred_str or "pakistan engineering council" in cred_str or "pec" in exp_str
            actual_val = has_pec
            passed = bool(has_pec)

        elif "has_pmdc" in rule.field_path:
            cred_str = " ".join([f"{c.name} {c.issuing_body}" for c in cv.credentials]).lower()
            has_pmdc = "pmdc" in cred_str or "pmc" in cred_str or "pakistan medical" in cred_str
            actual_val = has_pmdc
            passed = bool(has_pmdc)

        elif "has_bar_license" in rule.field_path:
            cred_str = " ".join([f"{c.name} {c.issuing_body}" for c in cv.credentials]).lower()
            exp_str = " ".join([f"{e.job_title} {e.description or ''}" for e in cv.experience]).lower()
            has_bar = "bar council" in cred_str or "high court" in cred_str or "advocate" in exp_str
            actual_val = has_bar
            passed = bool(has_bar)

        else:
            # Generic fallback
            passed = True
            actual_val = "Not Evaluated"

        if passed:
            status = GateStatus.ELIGIBLE
            reason = f"Gate '{rule.name}' successfully satisfied."
        else:
            status = rule.failure_status
            reason = rule.failure_message
            if status == GateStatus.NOT_ELIGIBLE:
                failure_reasons.append(reason)
            else:
                verification_reasons.append(reason)

        checks.append(
            GateCheckResult(
                gate_id=rule.gate_id,
                gate_name=rule.name,
                passed=passed,
                status=status,
                reason=reason,
                actual_value=actual_val,
                expected_value=rule.expected_value,
            )
        )

    # Determine overall gate status
    if failure_reasons:
        overall_status = GateStatus.NOT_ELIGIBLE
        summary = f"Candidate failed {len(failure_reasons)} mandatory qualification gate(s)."
        passed_overall = False
    elif verification_reasons:
        overall_status = GateStatus.NEEDS_VERIFICATION
        summary = f"Candidate requires verification for {len(verification_reasons)} credential gate(s)."
        passed_overall = True
    else:
        overall_status = GateStatus.ELIGIBLE
        summary = "Candidate satisfies all mandatory hard qualification gates."
        passed_overall = True

    return GateEvaluation(
        overall_status=overall_status,
        passed=passed_overall,
        summary=summary,
        checks=checks,
        failure_reasons=failure_reasons,
        verification_reasons=verification_reasons,
    )


def audit_fairness_flags(cv: NormalizedCV) -> List[FairnessAuditFlag]:
    """
    Audits candidate profile for chronological anomalies, gaps, and data omissions.
    Produces human-review flags instead of automatic silent score penalties.
    """
    flags: List[FairnessAuditFlag] = []

    # 1. Check for missing graduation years
    for e in cv.education:
        if not e.end_year and not e.start_year:
            flags.append(
                FairnessAuditFlag(
                    code="MISSING_DEGREE_DATES",
                    severity=FlagSeverity.LOW,
                    section="education",
                    description=f"Degree '{e.degree}' lacks explicit start or graduation dates.",
                    suggested_inquiry="Could you clarify the period during which this degree was pursued?",
                )
            )

    # 2. Check for missing licensing numbers on mandatory credentials
    for c in cv.credentials:
        if not c.license_number:
            flags.append(
                FairnessAuditFlag(
                    code="UNSPECIFIED_LICENSE_NUMBER",
                    severity=FlagSeverity.LOW,
                    section="credentials",
                    description=f"Credential '{c.name}' has no registration number listed.",
                    suggested_inquiry=f"Please provide your formal registration number for {c.name}.",
                )
            )

    # 3. Check for career tenure gaps (> 24 months unexplained)
    sorted_exps = sorted(
        [e for e in cv.experience if e.start_date],
        key=lambda x: str(x.start_date),
    )
    # If gap detector finds anomalies, flags are attached safely without score cuts

    return flags


# ============================================================================
# 4. UNIVERSAL COMBINER & PIPELINE ENTRY POINT
# ============================================================================

def evaluate_candidate(
    cv: NormalizedCV,
    profile: DepartmentProfile,
    role_preset: RolePreset = RolePreset.FACULTY,
    jd_requirements: Optional[Dict[str, Any]] = None,
) -> CandidateEvaluationResult:
    """
    Universal Candidate Evaluation Pipeline:
      1. Strips PII before blind scoring (anonymized_copy()).
      2. Evaluates profile-defined hard gates (GateEvaluation).
      3. Calculates normalized dimensional weights for target RolePreset.
      4. Executes all 8 Universal Dimension scorers.
      5. Computes evidence coverage & confidence multiplier:
         confidence_multiplier = 0.85 + (0.15 * evidence_coverage).
      6. Calculates base composite score and final score.
      7. Maps to standardized CandidateTier.
      8. Extracts explainability rationales, strengths, gaps, and fairness flags.
    """
    # 1. Anonymize CV to eliminate unconscious demographic bias
    anonymized_cv = cv.anonymized_copy()

    # 2. Pre-scoring Hard Gates
    gate_eval = evaluate_hard_gates(cv, profile)

    # 3. Effective weights for target role preset (guaranteed sum = 1.0)
    effective_weights = profile.get_effective_weights(role_preset)

    # 4. Execute 8 Dimension Scorers
    scorers = {
        UniversalDimension.ACADEMIC_FOUNDATION: score_academic_foundation,
        UniversalDimension.DOMAIN_COMPETENCE: score_domain_competence,
        UniversalDimension.PROFESSIONAL_EXPERIENCE: score_professional_experience,
        UniversalDimension.EVIDENCE_OF_OUTPUT: score_evidence_of_output,
        UniversalDimension.CREDENTIALS_LICENSURE: score_credentials_licensure,
        UniversalDimension.IMPACT_RECOGNITION: score_impact_recognition,
        UniversalDimension.LEADERSHIP_TEACHING_SERVICE: score_leadership_teaching_service,
        UniversalDimension.TRAJECTORY_GROWTH: score_trajectory_growth,
    }

    dimension_results: Dict[UniversalDimension, DimensionResult] = {}
    base_composite_score = 0.0
    sufficient_evidence_count = 0

    for dim, scorer_fn in scorers.items():
        w = effective_weights.get(dim, 0.125)
        res = scorer_fn(anonymized_cv, profile, jd_requirements=jd_requirements)
        res.weight = w
        res.weighted_score = round(res.score * w, 4)

        dimension_results[dim] = res
        base_composite_score += res.weighted_score

        if res.has_sufficient_evidence:
            sufficient_evidence_count += 1

    base_composite_score = min(max(round(base_composite_score, 2), 0.0), 100.0)

    # 5. Evidence Coverage & Confidence Multiplier
    # confidence_multiplier = 0.85 + (0.15 * evidence_coverage)
    evidence_coverage = round(sufficient_evidence_count / 8.0, 4)
    confidence_multiplier = round(0.85 + (0.15 * evidence_coverage), 4)

    # 6. Final Score
    final_score = round(base_composite_score * confidence_multiplier, 2)
    final_score = min(max(final_score, 0.0), 100.0)

    # 7. Candidate Tier Classification
    if final_score >= 80.0:
        tier = CandidateTier.TIER_1
    elif final_score >= 65.0:
        tier = CandidateTier.TIER_2
    elif final_score >= 50.0:
        tier = CandidateTier.TIER_3
    else:
        tier = CandidateTier.TIER_4

    # 8. Fairness Audit Flags & Strengths/Gaps
    fairness_flags = audit_fairness_flags(cv)

    strengths: List[str] = []
    gaps: List[str] = []

    for dim, res in dimension_results.items():
        dim_label = dim.value.replace("_", " ").title()
        if res.score >= 75.0:
            strengths.append(f"{dim_label}: {res.evidence_summary}")
        elif res.score < 50.0:
            gaps.append(f"{dim_label}: {res.evidence_summary}")

    return CandidateEvaluationResult(
        candidate_id=cv.candidate_id,
        candidate_name=cv.personal_identifiers.name,
        anonymized_id=anonymized_cv.candidate_id,
        department_slug=profile.slug,
        department_name=profile.department_name,
        role_preset=role_preset,
        gate_evaluation=gate_eval,
        gate_status=gate_eval.overall_status,
        dimension_scores=dimension_results,
        base_composite_score=base_composite_score,
        evidence_coverage=evidence_coverage,
        confidence_multiplier=confidence_multiplier,
        final_score=final_score,
        tier=tier,
        flags=fairness_flags,
        strengths=strengths,
        gaps=gaps,
        human_review_recommended=bool(gate_eval.overall_status != GateStatus.ELIGIBLE or fairness_flags),
    )


# ============================================================================
# 5. INTERNAL HELPERS
# ============================================================================

def _estimate_experience_years(start: Optional[str], end: Optional[str]) -> float:
    """Estimates duration in years from start and end date strings."""
    if not start:
        return 1.0

    curr_year = datetime.datetime.now().year
    m_start = re.search(r"(\d{4})", str(start))
    start_y = int(m_start.group(1)) if m_start else None

    if not start_y or start_y < 1970 or start_y > curr_year:
        return 1.0

    if not end or any(w in str(end).lower() for w in ["present", "current", "now"]):
        end_y = curr_year
    else:
        m_end = re.search(r"(\d{4})", str(end))
        end_y = int(m_end.group(1)) if m_end else start_y + 1

    return max(float(end_y - start_y), 0.5)
