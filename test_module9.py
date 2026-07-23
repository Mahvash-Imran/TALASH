"""
test_module9.py  –  Comprehensive Test Suite for Module 9 (Experience & Skill Alignment)
========================================================================================

Tests:
  1. Clean import of Module 9 classes and functions.
  2. Date standardization logic.
  3. Unified timeline generation.
  4. Overlap detection & classification.
  5. Gap detection & justification checking.
  6. Skill evidence level classification & JD scoring.
  7. Multi-module regression check (Modules 2, 3, 4, 5, 6, 7, 8, 9).
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

errors = []

# ---------------------------------------------------------------------------
# Test 1: Module Imports
# ---------------------------------------------------------------------------
try:
    from analysis.timeline_builder import (
        standardize_date,
        build_candidate_timeline,
        detect_overlaps,
        detect_experience_gaps,
        assess_career_progression,
    )
    from analysis.skill_aligner import (
        extract_and_align_skills,
        compute_jd_alignment_score,
    )
    from analysis.experience_analyser import ExperienceProfileAnalyser
    from analysis import ExperienceProfileAnalyser as EPA2
    print("PASS: Module 9 imports clean")
except Exception as e:
    errors.append(f"FAIL: Module 9 imports: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 2: Date Standardization
# ---------------------------------------------------------------------------
try:
    s1, _ = standardize_date("17-Sep")
    s2, _ = standardize_date("23-Aug")
    s3, _ = standardize_date("2024")
    s4, _ = standardize_date("Present")
    assert s1 == "2017-09"
    assert s2 == "2023-08"
    assert s3 == "2024-01"
    assert s4 is not None and len(s4) == 7
    print("PASS: Date standardization clean")
except Exception as e:
    errors.append(f"FAIL: Date standardization: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 3: Unified Timeline Generation
# ---------------------------------------------------------------------------
try:
    edu = [{"degree": "MS Electrical", "institution": "COMSATS", "start_year": "2012", "end_year": "2014"}]
    exp = [{"job_title": "Lecturer", "organization": "Qurtuba", "start_date": "15-Aug", "end_date": "18-Mar"}]
    timeline = build_candidate_timeline("C001", edu, exp)
    assert len(timeline) == 2
    assert timeline[0]["event_type"] in ("education", "experience")
    print("PASS: Unified timeline generation clean")
except Exception as e:
    errors.append(f"FAIL: Unified timeline generation: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 4: Overlap Detection & Classification
# ---------------------------------------------------------------------------
try:
    edu = [{"degree": "PhD CS", "institution": "NUST", "start_year": "2018", "end_year": "2022"}]
    exp_ta = [{"job_title": "Teaching Assistant", "organization": "NUST", "start_date": "18-Sep", "end_date": "22-Jun"}]
    t_ta = build_candidate_timeline("C001", edu, exp_ta)
    ovl_ta = detect_overlaps(t_ta)
    assert len(ovl_ta) >= 1
    assert ovl_ta[0]["classification"] == "Acceptable"

    exp_full = [{"job_title": "Software Engineer", "organization": "Tech Corp", "start_date": "18-Sep", "end_date": "22-Jun"}]
    t_full = build_candidate_timeline("C001", edu, exp_full)
    ovl_full = detect_overlaps(t_full)
    assert len(ovl_full) >= 1
    assert ovl_full[0]["classification"] == "Suspicious"
    print("PASS: Overlap detection & classification clean")
except Exception as e:
    errors.append(f"FAIL: Overlap detection: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 5: Gap Detection & Justification
# ---------------------------------------------------------------------------
try:
    exp_gap = [
        {"job_title": "Lecturer", "organization": "Uni A", "start_date": "10-Jan", "end_date": "12-Dec"},
        {"job_title": "Assistant Professor", "organization": "Uni B", "start_date": "15-Jan", "end_date": "20-Dec"}
    ]
    t_gap = build_candidate_timeline("C001", [], exp_gap)
    gaps = detect_experience_gaps(t_gap, candidate_pubs=[{"year": "2013"}, {"year": "2014"}])
    assert len(gaps) == 1
    assert gaps[0]["justification_status"] == "Justified"
    print("PASS: Gap detection & justification clean")
except Exception as e:
    errors.append(f"FAIL: Gap detection: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 6: Skill Evidence Classification & JD Alignment Scoring
# ---------------------------------------------------------------------------
try:
    skills_raw = [{"skill_name": "Python", "category": "Programming"}]
    exp_records = [{"job_title": "Python Developer", "description": "Developed Machine Learning models in Python"}]
    pub_records = [{"title": "Deep Learning and Machine Learning for Image Classification"}]

    aligned = extract_and_align_skills("C001", skills_raw, exp_records, pub_records)
    py_skill = [s for s in aligned if s["skill_name"].lower() == "python"][0]
    assert py_skill["evidence_level"] in ("Strong Evidence", "Moderate Evidence")

    jd_res = compute_jd_alignment_score(aligned, ["Python", "Machine Learning", "C++"])
    assert jd_res["jd_alignment_score"] > 0
    print("PASS: Skill evidence classification & JD scoring clean")
except Exception as e:
    errors.append(f"FAIL: Skill evidence classification: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 7: Multi-Module Regression Check (Modules 2, 3, 4, 5, 6, 7, 8, 9)
# ---------------------------------------------------------------------------
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    from analysis.research_profile import ResearchProfileAnalyser
    from analysis.supervision_analyser import SupervisionAnalyser
    from analysis.book_analyser import BookProfileAnalyser
    from analysis.patent_analyser import PatentProfileAnalyser
    from analysis.topic_analyser import TopicBreadthAnalyser
    from analysis.collaboration_analyser import CollaborationAnalyser
    from analysis.experience_analyser import ExperienceProfileAnalyser
    print("PASS: All modules (2, 3, 4, 5, 6, 7, 8, 9) import cleanly together")
except Exception as e:
    errors.append(f"FAIL: Regression check for all modules: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 7 smoke tests PASSED for Module 9.")
