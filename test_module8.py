"""
test_module8.py  –  Comprehensive Test Suite for Module 8 (Co-Author Collaboration Analysis)
========================================================================================

Tests:
  1. Clean import of Module 8 classes and functions.
  2. Author name normalization.
  3. Co-author parsing & candidate filtering.
  4. Team size profiling.
  5. Student co-author matching.
  6. Collaboration metrics & diversity index calculation.
  7. Multi-module regression check (Modules 2, 3, 4, 5, 6, 7, 8).
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
    from analysis.collaboration_verifier import (
        normalize_author_name,
        parse_coauthors,
        classify_team_size_profile,
        match_student_coauthors,
        compute_collaboration_metrics,
    )
    from analysis.collaboration_analyser import CollaborationAnalyser
    from analysis import CollaborationAnalyser as CA2
    print("PASS: Module 8 imports clean")
except Exception as e:
    errors.append(f"FAIL: Module 8 imports: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 2: Author Name Normalization
# ---------------------------------------------------------------------------
try:
    assert normalize_author_name("Dr. Sara Khan, PhD") == "Sara Khan"
    assert normalize_author_name("Prof. Ali Hassan et al.") == "Ali Hassan"
    assert normalize_author_name("Qureshi, M.F.") == "Qureshi M F"
    assert normalize_author_name("") == ""
    print("PASS: Author name normalization clean")
except Exception as e:
    errors.append(f"FAIL: Author name normalization: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 3: Co-Author Parsing & Candidate Filtering
# ---------------------------------------------------------------------------
try:
    authors_str = "Muhammad Farrukh, Dr. Sara Khan, Ahmed Ali"
    coauthors, total_cnt = parse_coauthors(authors_str, "Muhammad Farrukh")
    assert "Sara Khan" in coauthors
    assert "Ahmed Ali" in coauthors
    assert "Muhammad Farrukh" not in coauthors
    assert total_cnt == 3
    print("PASS: Co-author parsing & candidate filtering clean")
except Exception as e:
    errors.append(f"FAIL: Co-author parsing: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 4: Team Size Profiling
# ---------------------------------------------------------------------------
try:
    assert classify_team_size_profile(2.5, 5) == "Solo/Small-Group Researcher"
    assert classify_team_size_profile(4.0, 5) == "Medium-Group Researcher"
    assert classify_team_size_profile(6.2, 5) == "Large-Group Researcher"
    assert classify_team_size_profile(0.0, 0) == "No Publications"
    print("PASS: Team size profiling clean")
except Exception as e:
    errors.append(f"FAIL: Team size profiling: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 5: Student Co-Author Matching
# ---------------------------------------------------------------------------
try:
    coauthors = ["Dr. Sara Khan", "Hajra Binte Naeem", "John Smith"]
    students = ["Hajra Naeem", "Usman Ali"]
    matched = match_student_coauthors(coauthors, students)
    assert "Hajra Naeem" in matched
    assert "Usman Ali" not in matched
    print("PASS: Student co-author matching clean")
except Exception as e:
    errors.append(f"FAIL: Student co-author matching: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 6: Collaboration Metrics & Diversity Score Calculation
# ---------------------------------------------------------------------------
try:
    counts = {
        "Sara Khan": 5,
        "Ahmed Ali": 3,
        "John Doe": 1,
        "Jane Smith": 1,
        "Robert Brown": 1,
    }
    metrics = compute_collaboration_metrics(
        coauthor_counts=counts,
        total_papers=10,
        total_authors_sum=35,
        student_matches_count=1
    )
    assert metrics["total_unique_coauthors"] == 5
    assert metrics["recurring_collaborators_count"] == 2
    assert metrics["one_time_collaborators"] == 3
    assert metrics["avg_authors_per_paper"] == 3.5
    assert metrics["collaboration_diversity_score"] == 0.60  # 3/5 = 0.6
    assert metrics["team_size_profile"] == "Medium-Group Researcher"
    assert metrics["collaboration_strength_label"] == "Balanced Network"
    print("PASS: Collaboration metrics & diversity score clean")
except Exception as e:
    errors.append(f"FAIL: Collaboration metrics calculation: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 7: Multi-Module Regression Check (Modules 2, 3, 4, 5, 6, 7, 8)
# ---------------------------------------------------------------------------
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    from analysis.research_profile import ResearchProfileAnalyser
    from analysis.supervision_analyser import SupervisionAnalyser
    from analysis.book_analyser import BookProfileAnalyser
    from analysis.patent_analyser import PatentProfileAnalyser
    from analysis.topic_analyser import TopicBreadthAnalyser
    from analysis.collaboration_analyser import CollaborationAnalyser
    print("PASS: All modules (2, 3, 4, 5, 6, 7, 8) import cleanly together")
except Exception as e:
    errors.append(f"FAIL: Regression check for all modules: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 7 smoke tests PASSED for Module 8.")
