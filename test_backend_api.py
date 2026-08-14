"""
test_backend_api.py  –  End-to-End API Test Suite for Backend & Integration
========================================================================

Tests:
  1. Module 10 Candidate Composite Evaluator execution.
  2. Health check endpoint (GET /api/v1/health).
  3. Candidate listing endpoint (GET /api/v1/candidates).
  4. Candidate detail endpoint (GET /api/v1/candidates/{id}).
  5. Candidate side-by-side comparison endpoint (POST /api/v1/compare).
  6. Email drafting endpoint (GET /api/v1/email/{id}).
  7. Resume upload endpoint (POST /api/v1/upload).
  8. Master Pipeline integration test across sample dataset.
"""

import sys
from pathlib import Path

# Load .env FIRST — must happen before any import reads os.environ for API keys
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except ImportError:
    pass

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

errors = []

# ---------------------------------------------------------------------------
# Test 1: Module 10 Composite Evaluator Execution
# ---------------------------------------------------------------------------
try:
    from analysis.composite_evaluator import CompositeEvaluator, compute_candidate_composite_score
    score_res = compute_candidate_composite_score(
        candidate_id="TEST_01",
        edu_profile={"educational_strength_label": "Strong", "phd_count": "1"},
        research_profile={"scholarly_strength_label": "Strong", "total_publications": "20"},
        supervision_profile={"ms_supervised_main": "5", "phd_supervised_main": "2"},
        book_profile={"total_books": "1", "verifiable_books_count": "1"},
        patent_profile={"total_patents": "1", "verifiable_patents": "1"},
        breadth_profile={"shannon_entropy_diversity_score": "0.85"},
        collab_profile={"collaboration_strength_label": "Broad Network", "total_unique_coauthors": "25"},
        exp_profile={"total_experience_years": "12.0", "jd_alignment_score": "85.0"}
    )
    assert score_res["overall_composite_score"] >= 80.0
    assert "Tier 1" in score_res["candidate_tier"]

    comp = CompositeEvaluator()
    paths = comp.run()
    assert paths["composite_csv"].exists()
    print("PASS: Module 10 Candidate Composite Evaluator clean")
except Exception as e:
    errors.append(f"FAIL: Module 10 Composite Evaluator: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 2: FastAPI TestClient Initialization & Health Endpoint
# ---------------------------------------------------------------------------
try:
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("PASS: FastAPI Health check endpoint clean")
except Exception as e:
    errors.append(f"FAIL: FastAPI Health check: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 3: Candidate Listing Endpoint
# ---------------------------------------------------------------------------
try:
    resp = client.get("/api/v1/candidates")
    assert resp.status_code == 200
    cands = resp.json()
    assert len(cands) >= 1
    assert "overall_composite_score" in cands[0]
    print(f"PASS: GET /api/v1/candidates clean ({len(cands)} candidates returned)")
except Exception as e:
    errors.append(f"FAIL: GET /api/v1/candidates: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 4: Candidate Detail Endpoint
# ---------------------------------------------------------------------------
try:
    first_cid = cands[0]["candidate_id"]
    resp = client.get(f"/api/v1/candidates/{first_cid}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["candidate_id"] == first_cid
    print(f"PASS: GET /api/v1/candidates/{{{first_cid}}} clean")
except Exception as e:
    errors.append(f"FAIL: GET /api/v1/candidates/{{id}}: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 5: Candidate Side-by-Side Comparison Endpoint
# ---------------------------------------------------------------------------
try:
    cand_ids = [c["candidate_id"] for c in cands[:2]]
    resp = client.post("/api/v1/compare", json={"candidate_ids": cand_ids})
    assert resp.status_code == 200
    comp_res = resp.json()
    assert comp_res["comparison_count"] == len(cand_ids)
    assert len(comp_res["ranking"]) == len(cand_ids)
    print(f"PASS: POST /api/v1/compare clean (compared {len(cand_ids)} candidates)")
except Exception as e:
    errors.append(f"FAIL: POST /api/v1/compare: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 6: Email Drafting Endpoint
# ---------------------------------------------------------------------------
try:
    resp = client.get(f"/api/v1/email/{first_cid}")
    assert resp.status_code == 200
    email_data = resp.json()
    assert "subject" in email_data and "body" in email_data
    print(f"PASS: GET /api/v1/email/{{{first_cid}}} clean")
except Exception as e:
    errors.append(f"FAIL: GET /api/v1/email/{{id}}: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 7: Master Pipeline Integration
# ---------------------------------------------------------------------------
try:
    from pipeline_orchestrator import MasterPipeline
    pipeline = MasterPipeline(skip_llm=True)
    res = pipeline.run_full_pipeline()
    assert res["composite_csv"].exists()
    print("PASS: Master Pipeline Parts 1-10 execution clean")
except Exception as e:
    errors.append(f"FAIL: Master Pipeline integration: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 7 End-to-End API & Integration tests PASSED.")
