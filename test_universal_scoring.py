"""
test_universal_scoring.py  –  Unit Tests for Universal Scoring Engine (Phase 3)
==============================================================================
"""

import sys
from pathlib import Path

# Add talash root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import models
import profile_loader
import universal_scoring


def test_percentile_scale_math():
    """Verify statistical percentile interpolation and max_cap enforcement."""
    base = models.BaselineMetrics(median=6, p75=15, p90=30, max_cap=80, unit="count")
    
    # 0 should yield 0
    assert universal_scoring.percentile_scale(0, base) == 0.0
    
    # Median should yield exactly 50
    assert universal_scoring.percentile_scale(6, base) == 50.0
    
    # p75 should yield exactly 75
    assert universal_scoring.percentile_scale(15, base) == 75.0
    
    # p90 should yield exactly 90
    assert universal_scoring.percentile_scale(30, base) == 90.0
    
    # max_cap should yield 100
    assert universal_scoring.percentile_scale(80, base) == 100.0
    
    # Beyond max_cap should cap strictly at 100.0 without overflow
    assert universal_scoring.percentile_scale(500, base) == 100.0
    
    # Monotonicity test
    val1 = universal_scoring.percentile_scale(10, base)
    val2 = universal_scoring.percentile_scale(20, base)
    assert 50.0 < val1 < 75.0
    assert 75.0 < val2 < 90.0
    assert val1 < val2
    print("[PASS] test_percentile_scale_math passed.")


def test_experience_diminishing_returns():
    """Verify diminishing returns model for career years."""
    # 0 years -> 0 pts
    assert universal_scoring.calculate_experience_score_with_diminishing_returns(0) == 0.0
    
    # 3 years -> 40 pts
    assert universal_scoring.calculate_experience_score_with_diminishing_returns(3.0) == 40.0
    
    # 8 years -> 75 pts
    assert universal_scoring.calculate_experience_score_with_diminishing_returns(8.0) == 75.0
    
    # 15 years -> 92 pts
    assert universal_scoring.calculate_experience_score_with_diminishing_returns(15.0) == 92.0
    
    # 25 years -> diminishing returns past 15 years (between 92 and 100)
    score_25 = universal_scoring.calculate_experience_score_with_diminishing_returns(25.0)
    score_40 = universal_scoring.calculate_experience_score_with_diminishing_returns(40.0)
    assert 92.0 < score_25 < score_40 <= 100.0
    print("[PASS] test_experience_diminishing_returns passed.")


def test_academic_foundation_scorer():
    """Test Dimension 1: Degree level, GPA normalization, and 40% institution tier cap."""
    cs_profile = profile_loader.get_profile("computer_science")
    assert cs_profile is not None

    cv = models.NormalizedCV(
        candidate_id="cand_test_01",
        personal_identifiers=models.PersonalIdentifiers(name="Dr. Candidate"),
        education=[
            models.EducationRecord(
                level="PhD",
                degree="Computer Science",
                institution="FAST NUCES",
                cgpa=3.9,
                cgpa_scale=4.0,
            )
        ]
    )

    res = universal_scoring.score_academic_foundation(cv, cs_profile)
    assert res.dimension == models.UniversalDimension.ACADEMIC_FOUNDATION
    assert res.score >= 85.0
    assert res.metrics["institution_points"] <= 40.0  # Must be strictly capped at 40 pts
    assert res.has_sufficient_evidence is True
    print(f"[PASS] test_academic_foundation_scorer passed (score={res.score}).")


def test_domain_competence_semantic_match():
    """Test Dimension 2: Core vs secondary matching with sub-branch equivalence."""
    cs_profile = profile_loader.get_profile("computer_science")
    assert cs_profile is not None

    # Candidate lists deep learning, neural networks, python, algorithms, docker
    cv = models.NormalizedCV(
        candidate_id="cand_test_02",
        skills=[
            models.SkillRecord(skill_name="deep learning", category="core", evidence_count=3),
            models.SkillRecord(skill_name="neural networks", category="core", evidence_count=2),
            models.SkillRecord(skill_name="python", category="tool", evidence_count=4),
            models.SkillRecord(skill_name="algorithms", category="core", evidence_count=2),
            models.SkillRecord(skill_name="docker", category="secondary", evidence_count=1),
        ]
    )

    res = universal_scoring.score_domain_competence(cv, cs_profile)
    assert res.dimension == models.UniversalDimension.DOMAIN_COMPETENCE
    assert res.score >= 50.0
    assert len(res.metrics["matched_core"]) >= 1
    print(f"[PASS] test_domain_competence_semantic_match passed (score={res.score}).")


def test_evidence_of_output_artifacts():
    """Test Dimension 4: Percentile scaling of department-defined artifacts."""
    cs_profile = profile_loader.get_profile("computer_science")
    assert cs_profile is not None

    cv = models.NormalizedCV(
        candidate_id="cand_test_03",
        artifacts=[
            models.EvidenceArtifactRecord(
                artifact_type="journal",
                title="Deep Vision Paper 1",
                venue_or_platform="IEEE TPAMI",
                year=2023,
                metrics={"citations": 45}
            ),
            models.EvidenceArtifactRecord(
                artifact_type="journal",
                title="Transformer Paper 2",
                venue_or_platform="ACM Computing Surveys",
                year=2022,
                metrics={"citations": 30}
            ),
            models.EvidenceArtifactRecord(
                artifact_type="software_code",
                title="Open Source Library",
                venue_or_platform="GitHub",
                metrics={"stars": 120}
            ),
            models.EvidenceArtifactRecord(
                artifact_type="patent",
                title="Novel Sensor Architecture",
                venue_or_platform="US Patent Office",
                metrics={"patent_status": "Granted"}
            )
        ]
    )

    res = universal_scoring.score_evidence_of_output(cv, cs_profile)
    assert res.dimension == models.UniversalDimension.EVIDENCE_OF_OUTPUT
    assert res.score >= 60.0
    assert res.has_sufficient_evidence is True
    print(f"[PASS] test_evidence_of_output_artifacts passed (score={res.score}).")


def test_hard_gates_evaluation():
    """Test pre-scoring hard gate pass, fail, and verification statuses."""
    ee_profile = profile_loader.get_profile("electrical_engineering")
    assert ee_profile is not None

    # Candidate with PEC registration
    cand_with_pec = models.NormalizedCV(
        candidate_id="cand_ee_01",
        education=[models.EducationRecord(level="PhD", degree="Electrical Engineering")],
        credentials=[models.CredentialLicensureRecord(name="PEC Registration", issuing_body="Pakistan Engineering Council")]
    )
    eval_pass = universal_scoring.evaluate_hard_gates(cand_with_pec, ee_profile)
    assert eval_pass.overall_status == models.GateStatus.ELIGIBLE
    assert eval_pass.passed is True

    # Candidate without PEC registration (engineering appointment requires verification)
    cand_no_pec = models.NormalizedCV(
        candidate_id="cand_ee_02",
        education=[models.EducationRecord(level="PhD", degree="Electrical Engineering")]
    )
    eval_verify = universal_scoring.evaluate_hard_gates(cand_no_pec, ee_profile)
    assert eval_verify.overall_status == models.GateStatus.NEEDS_VERIFICATION
    print("[PASS] test_hard_gates_evaluation passed.")


def test_full_pipeline_combiner_and_backward_compatibility():
    """Test full evaluate_candidate pipeline, confidence multiplier, and legacy compatibility."""
    cs_profile = profile_loader.get_profile("computer_science")
    assert cs_profile is not None

    cv = models.NormalizedCV(
        candidate_id="01_MUHAMMAD_SALMAN",
        personal_identifiers=models.PersonalIdentifiers(
            name="Dr. Muhammad Salman",
            email="salman@example.com",
            phone="+92-300-1234567"
        ),
        education=[
            models.EducationRecord(
                level="PhD",
                degree="Electrical Engineering",
                institution="International Islamic university, Islamabad",
                cgpa=3.33,
                cgpa_scale=4.0
            )
        ],
        experience=[
            models.ExperienceRecord(
                job_title="Program Coordinator",
                organization="Qurtuba University, D.I.Khan",
                duration_years=6.0,
                description="Supervising MS students and teaching computer networking."
            )
        ],
        skills=[
            models.SkillRecord(skill_name="wireless networks", category="core"),
            models.SkillRecord(skill_name="machine learning", category="core"),
            models.SkillRecord(skill_name="python", category="tool")
        ],
        artifacts=[
            models.EvidenceArtifactRecord(
                artifact_type="journal",
                title="A Hybrid Framework Integrating Deterministic Clustering in WSN",
                venue_or_platform="International Journal",
                year=2025,
                metrics={"citations": 12}
            )
        ],
        leadership_teaching=[
            models.LeadershipTeachingRecord(
                category="supervision",
                title_or_role="Supervisor",
                level="MS",
                student_name="Student A",
                thesis_title="Network Optimization"
            )
        ]
    )

    # 1. Run evaluation
    result = universal_scoring.evaluate_candidate(
        cv=cv,
        profile=cs_profile,
        role_preset=models.RolePreset.FACULTY
    )

    # 2. Check universal dimensions and scores
    assert 0.0 <= result.final_score <= 100.0
    assert 0.85 <= result.confidence_multiplier <= 1.0
    assert 0.0 <= result.evidence_coverage <= 1.0
    assert result.tier in list(models.CandidateTier)

    # 3. Verify Blind Anonymization
    assert result.anonymized_id != cv.candidate_id
    assert result.anonymized_id.startswith("cand_")

    # 4. Verify Backward-Compatibility Adapter (.to_legacy_dict())
    legacy = result.to_legacy_dict()
    required_legacy_keys = [
        "candidate_id", "candidate_name", "overall_composite_score", "candidate_tier",
        "education_score", "research_score", "supervision_score", "innovation_score",
        "breadth_score", "collaboration_score", "experience_score"
    ]
    for k in required_legacy_keys:
        assert k in legacy, f"Missing legacy key '{k}'"
        assert legacy[k] is not None

    assert legacy["candidate_id"] == "01_MUHAMMAD_SALMAN"
    assert legacy["candidate_name"] == "Dr. Muhammad Salman"
    assert isinstance(legacy["overall_composite_score"], float)

    print(f"[PASS] test_full_pipeline_combiner_and_backward_compatibility passed.")
    print(f"  Final Score: {result.final_score:.1f}/100 | Tier: {result.tier.value} | Multiplier: {result.confidence_multiplier:.3f}")


def test_credentials_licensure_scorer():
    """Test Dimension 5: Mandatory vs preferred credentials."""
    med_profile = profile_loader.get_profile("medicine")
    assert med_profile is not None

    # Candidate with active PMDC license
    cv_med = models.NormalizedCV(
        candidate_id="cand_med_01",
        credentials=[
            models.CredentialLicensureRecord(name="PMDC License", issuing_body="Pakistan Medical and Dental Council", status="active"),
            models.CredentialLicensureRecord(name="FCPS", issuing_body="CPSP", status="active")
        ]
    )
    res_med = universal_scoring.score_credentials_licensure(cv_med, med_profile)
    assert res_med.dimension == models.UniversalDimension.CREDENTIALS_LICENSURE
    assert res_med.score >= 80.0
    assert res_med.metrics["mandatory_met"] is True

    # Candidate without PMDC license
    cv_no_pmdc = models.NormalizedCV(
        candidate_id="cand_med_02",
        credentials=[]
    )
    res_no_pmdc = universal_scoring.score_credentials_licensure(cv_no_pmdc, med_profile)
    assert res_no_pmdc.score < 50.0
    assert res_no_pmdc.metrics["mandatory_met"] is False
    print(f"[PASS] test_credentials_licensure_scorer passed.")


def test_impact_and_leadership_scorers():
    """Test Dimension 6 (Impact/Recognition) and Dimension 7 (Leadership/Teaching)."""
    cs_profile = profile_loader.get_profile("computer_science")
    assert cs_profile is not None

    cv = models.NormalizedCV(
        candidate_id="cand_impact_01",
        impact_recognition=[
            models.ImpactRecognitionRecord(title="HEC Best Researcher Award", category="award"),
            models.ImpactRecognitionRecord(title="National Technology Fund Grant", category="grant", amount=5000000)
        ],
        artifacts=[
            models.EvidenceArtifactRecord(
                artifact_type="journal",
                title="Seminal Deep Learning Paper",
                venue_or_platform="IEEE TPAMI",
                metrics={"citations": 250}
            )
        ],
        leadership_teaching=[
            models.LeadershipTeachingRecord(category="supervision", title_or_role="Main Supervisor", level="PHD", student_name="Student 1"),
            models.LeadershipTeachingRecord(category="supervision", title_or_role="Main Supervisor", level="PHD", student_name="Student 2"),
            models.LeadershipTeachingRecord(category="supervision", title_or_role="Co-Supervisor", level="MS", student_name="Student 3")
        ],
        experience=[
            models.ExperienceRecord(job_title="Full Professor and Chair", organization="NUST SEECS", duration_years=6.0),
            models.ExperienceRecord(job_title="Associate Professor", organization="NUST", duration_years=4.0),
            models.ExperienceRecord(job_title="Assistant Professor", organization="NUST", duration_years=4.0),
        ]
    )

    res_impact = universal_scoring.score_impact_recognition(cv, cs_profile)
    assert res_impact.score >= 60.0

    res_lead = universal_scoring.score_leadership_teaching_service(cv, cs_profile)
    assert res_lead.score >= 70.0
    assert res_lead.metrics["phd_supervised"] == 2
    print(f"[PASS] test_impact_and_leadership_scorers passed.")


def test_role_preset_modifiers():
    """Verify role preset modifiers reweight dimensions appropriately."""
    cs_profile = profile_loader.get_profile("computer_science")
    assert cs_profile is not None

    # Faculty should weigh output & teaching higher
    faculty_w = cs_profile.get_effective_weights(models.RolePreset.FACULTY)
    # Industry professional should weigh experience higher
    ind_w = cs_profile.get_effective_weights(models.RolePreset.INDUSTRY_PROFESSIONAL)
    # Fresh graduate should weigh academic foundation higher
    fresh_w = cs_profile.get_effective_weights(models.RolePreset.FRESH_GRADUATE)

    assert faculty_w[models.UniversalDimension.EVIDENCE_OF_OUTPUT] > ind_w[models.UniversalDimension.EVIDENCE_OF_OUTPUT]
    assert ind_w[models.UniversalDimension.PROFESSIONAL_EXPERIENCE] > faculty_w[models.UniversalDimension.PROFESSIONAL_EXPERIENCE]
    assert fresh_w[models.UniversalDimension.ACADEMIC_FOUNDATION] > faculty_w[models.UniversalDimension.ACADEMIC_FOUNDATION]

    assert round(sum(faculty_w.values()), 4) == 1.0
    assert round(sum(ind_w.values()), 4) == 1.0
    assert round(sum(fresh_w.values()), 4) == 1.0
    print("[PASS] test_role_preset_modifiers passed.")


if __name__ == "__main__":
    test_percentile_scale_math()
    test_experience_diminishing_returns()
    test_academic_foundation_scorer()
    test_domain_competence_semantic_match()
    test_evidence_of_output_artifacts()
    test_credentials_licensure_scorer()
    test_impact_and_leadership_scorers()
    test_role_preset_modifiers()
    test_hard_gates_evaluation()
    test_full_pipeline_combiner_and_backward_compatibility()
    print("\nALL UNIVERSAL SCORING TESTS PASSED (100% SUCCESS)!")
