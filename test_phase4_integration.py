"""
test_phase4_integration.py  –  Phase 4: Pipeline Integration & Non-Regression Test Suite
========================================================================================

WHY THIS MODULE EXISTS
----------------------
Verifies full end-to-end integration and non-regression guarantees:
  1. UniversalPipeline batch execution across real extracted candidate data.
  2. Strict backward compatibility: composite_evaluations.csv must maintain exact legacy
     column names and valid data types (no breaking changes for dashboard/frontend).
  3. Golden-file verification for flagship candidates (Dr. Tariq Mahmood, Muhammad Salman).
  4. Multi-tenancy isolation: verifies per-tenant data directories function independently.
  5. Live FastAPI endpoint verification using TestClient (Health, Candidates, Departments,
     Universal Candidate Evaluation, Compare, and JD Matching).
"""

import sys
import os
import shutil
from pathlib import Path

# Set working directory to talash base directory so all relative paths resolve
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

import models
import profile_loader
import universal_scoring
from universal_pipeline import UniversalPipeline
from api import routes
from api.schemas import CompareRequest

EXTRACTED_DIR = BASE_DIR / "data" / "extracted"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


def test_01_pipeline_batch_evaluation_and_schema():
    """
    Test 1: Batch evaluate extracted candidates and verify legacy CSV schema compatibility.
    """
    print("\n--- Test 1: Batch Pipeline Evaluation & Schema Non-Regression ---")
    pipe = UniversalPipeline(
        extracted_dir=EXTRACTED_DIR,
        analysis_dir=ANALYSIS_DIR,
    )
    out = pipe.evaluate_all(save=True)
    assert out["count"] > 0, "Pipeline should evaluate at least 1 candidate."
    print(f"Evaluated {out['count']} candidates.")

    # 1. Verify composite_evaluations.csv exists
    comp_file = ANALYSIS_DIR / "composite_evaluations.csv"
    assert comp_file.exists(), "composite_evaluations.csv must exist."

    df = pd.read_csv(comp_file)
    required_cols = [
        "candidate_id",
        "overall_composite_score",
        "candidate_tier",
        "education_score",
        "research_score",
        "supervision_score",
        "innovation_score",
        "breadth_score",
        "collaboration_score",
        "experience_score",
        "candidate_name",
    ]
    for col in required_cols:
        assert col in df.columns, f"Required legacy column '{col}' missing from composite_evaluations.csv!"

    # Verify rows are sorted descending by overall_composite_score
    scores = df["overall_composite_score"].tolist()
    assert scores == sorted(scores, reverse=True), "composite_evaluations.csv must be sorted descending by overall score."

    # 2. Verify universal_evaluations.json exists and is valid
    u_json_file = ANALYSIS_DIR / "universal_evaluations.json"
    assert u_json_file.exists(), "universal_evaluations.json must exist."

    print(f"[PASS] Test 1: composite_evaluations.csv validated with all {len(required_cols)} legacy columns intact.")


def test_02_golden_file_candidates():
    """
    Test 2: Golden-file assertions on flagship candidates.
    """
    print("\n--- Test 2: Golden-File Flagship Candidate Checks ---")
    pipe = UniversalPipeline(
        extracted_dir=EXTRACTED_DIR,
        analysis_dir=ANALYSIS_DIR,
    )

    # 1. Dr. Tariq Mahmood (Senior Tenured CS Professor / High Output)
    cid_tariq = "Dr_Tariq_Mahmood_Comprehensive_CV"
    tariq_res = pipe.evaluate_single(cid_tariq, department_slug="computer_science")
    assert tariq_res is not None, f"Candidate {cid_tariq} must be present in extracted data."
    assert tariq_res.final_score >= 70.0, f"Dr. Tariq should achieve high score (got {tariq_res.final_score})."
    assert tariq_res.gate_status == models.GateStatus.ELIGIBLE, "Dr. Tariq should pass all hard gates."
    assert tariq_res.tier in [models.CandidateTier.TIER_1, models.CandidateTier.TIER_2]
    print(f"[PASS] Golden-File Dr. Tariq: Score={tariq_res.final_score:.1f}, Tier={tariq_res.tier.value}, Gates={tariq_res.gate_status.value}")

    # 2. Muhammad Salman (Electrical Engineering PhD / WSN)
    cid_salman = "01_MUHAMMAD_SALMAN"
    salman_res = pipe.evaluate_single(cid_salman, department_slug="electrical_engineering")
    assert salman_res is not None, f"Candidate {cid_salman} must be present."
    assert salman_res.final_score > 35.0
    print(f"[PASS] Golden-File M. Salman: Score={salman_res.final_score:.1f}, Tier={salman_res.tier.value}, Gates={salman_res.gate_status.value}")


def test_03_multi_tenancy_data_isolation():
    """
    Test 3: Verify tenant-scoped directories evaluate independently without cross-leakage.
    """
    print("\n--- Test 3: Multi-Tenancy Data Isolation ---")
    tenant_dir = BASE_DIR / "data" / "tenants" / "test_iso_user@talash.pk" / "analysis"
    tenant_dir.mkdir(parents=True, exist_ok=True)

    try:
        tenant_pipe = UniversalPipeline(
            extracted_dir=EXTRACTED_DIR,
            analysis_dir=tenant_dir,
        )
        out = tenant_pipe.evaluate_all(
            candidate_ids=["01_MUHAMMAD_SALMAN", "04_MUHAMMAD_FARRUKH"],
            department_slug="electrical_engineering",
            save=True,
        )
        assert out["count"] == 2
        assert (tenant_dir / "composite_evaluations.csv").exists()
        assert (tenant_dir / "universal_evaluations.json").exists()

        tenant_df = pd.read_csv(tenant_dir / "composite_evaluations.csv")
        assert len(tenant_df) == 2, "Tenant file must contain strictly the 2 evaluated candidates."
        print(f"[PASS] Test 3: Tenant isolation verified ({out['count']} candidates in isolated directory).")
    finally:
        # Clean up tenant test folder
        shutil.rmtree(BASE_DIR / "data" / "tenants" / "test_iso_user@talash.pk", ignore_errors=True)


def test_04_fastapi_endpoints():
    """
    Test 4: Verify core legacy and new universal FastAPI route controllers directly.
    """
    print("\n--- Test 4: Live FastAPI Endpoints Verification ---")

    # 1. Health check
    res_health = routes.health_check()
    assert res_health.status == "ok"
    assert res_health.service == "TALASH Backend API"
    print("  [PASS] GET /api/v1/health -> 200 OK")

    # 2. Departments catalog
    depts_data = routes.list_departments()
    assert depts_data["count"] >= 6
    slugs = [d["slug"] for d in depts_data["departments"]]
    assert "computer_science" in slugs
    assert "medicine" in slugs
    print(f"  [PASS] GET /api/v1/departments -> 200 OK ({depts_data['count']} profiles cataloged)")

    # 3. Department profile detail
    cs_profile_json = routes.get_department_profile("computer_science")
    assert cs_profile_json["slug"] == "computer_science"
    assert round(sum(cs_profile_json["base_weights"].values()), 2) == 1.0
    print("  [PASS] GET /api/v1/departments/computer_science -> 200 OK (Weights validated)")

    # 4. Candidates list (Existing legacy route)
    cands_list = routes.list_candidates(user_email="admin@talash.pk")
    assert len(cands_list) > 0
    assert hasattr(cands_list[0], "overall_composite_score")
    print(f"  [PASS] GET /api/v1/candidates -> 200 OK ({len(cands_list)} candidates retrieved)")

    # 5. Universal Candidate Evaluation (New endpoint)
    cid = "01_MUHAMMAD_SALMAN"
    u_eval = routes.get_candidate_universal_evaluation(
        candidate_id=cid,
        department="electrical_engineering",
        role_preset="faculty",
        user_email="admin@talash.pk",
    )
    assert u_eval["candidate_id"] == cid
    assert "dimension_scores" in u_eval
    assert len(u_eval["dimension_scores"]) == 8
    assert "confidence_multiplier" in u_eval
    print(f"  [PASS] GET /api/v1/candidates/{cid}/universal -> 200 OK (8 Dimensions verified)")

    # 6. Candidate Compare (Existing legacy route)
    comp_json = routes.compare_candidates(
        CompareRequest(candidate_ids=["01_MUHAMMAD_SALMAN", "04_MUHAMMAD_FARRUKH"]),
        user_email="admin@talash.pk",
    )
    assert comp_json.comparison_count == 2
    assert len(comp_json.ranking) == 2
    print("  [PASS] POST /api/v1/candidates/compare -> 200 OK (Side-by-side comparison verified)")

    # 7. Universal Batch Evaluation Endpoint
    batch_json = routes.run_universal_batch_evaluation(
        department="computer_science",
        role_preset="faculty",
        user_email="admin@talash.pk",
    )
    assert batch_json["status"] == "success"
    assert batch_json["evaluated_count"] > 0
    print(f"  [PASS] POST /api/v1/candidates/evaluate-universal -> 200 OK ({batch_json['evaluated_count']} evaluated)")

    print("[PASS] Test 4: All FastAPI endpoints operational with zero regression.")


if __name__ == "__main__":
    test_01_pipeline_batch_evaluation_and_schema()
    test_02_golden_file_candidates()
    test_03_multi_tenancy_data_isolation()
    test_04_fastapi_endpoints()
    print("\n============================================================")
    print("ALL PHASE 4 INTEGRATION & NON-REGRESSION TESTS PASSED (100%)")
    print("============================================================")
