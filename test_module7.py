"""
test_module7.py  –  Comprehensive Test Suite for Module 7 (Topic Variability & Research Breadth)
=============================================================================================

Tests:
  1. Clean import of Module 7 classes and functions.
  2. Taxonomy rule-based theme classification.
  3. Shannon entropy and diversity score calculation.
  4. Temporal trend analysis.
  5. Profile type classification (Specialist, Focused Researcher, Interdisciplinary, No Publications).
  6. Multi-module regression check (Modules 2, 3, 4, 5, 6, 7).
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
    from analysis.topic_verifier import (
        classify_paper_theme_rule_based,
        calculate_shannon_entropy,
        analyze_temporal_trend,
        classify_research_profile_type,
        RESEARCH_TAXONOMY,
    )
    from analysis.topic_analyser import TopicBreadthAnalyser
    from analysis import TopicBreadthAnalyser as TBA2
    print("PASS: Module 7 imports clean")
except Exception as e:
    errors.append(f"FAIL: Module 7 imports: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 2: Taxonomy Rule-Based Theme Classification
# ---------------------------------------------------------------------------
try:
    # NLP title
    t1, _, _ = classify_paper_theme_rule_based("Deep Learning for Natural Language Processing and Text Summarization")
    assert t1 == "Natural Language Processing", f"Expected NLP, got {t1}"

    # CV title
    t2, _, _ = classify_paper_theme_rule_based("Multiple Batches of Motion History Images for Action Recognition")
    assert t2 == "Computer Vision & Image Processing", f"Expected Computer Vision, got {t2}"

    # Wireless title
    t3, _, _ = classify_paper_theme_rule_based("Adaptable Access Point Selection Method in CF-mMIMO Systems")
    assert t3 == "Wireless Networks & IoT", f"Expected Wireless Networks & IoT, got {t3}"

    # Cybersecurity title
    t4, _, _ = classify_paper_theme_rule_based("Intrusion Detection and Privacy Enhancement in Cloud Systems")
    assert t4 == "Cybersecurity & Privacy", f"Expected Cybersecurity & Privacy, got {t4}"

    # Energy title
    t5, _, _ = classify_paper_theme_rule_based("Solar Photovoltaic Cell Efficiency under Fast Pyrolysis")
    assert t5 == "Energy & Renewable Systems", f"Expected Energy & Renewable Systems, got {t5}"

    print("PASS: Taxonomy rule-based theme classification clean")
except Exception as e:
    errors.append(f"FAIL: Taxonomy classification: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 3: Shannon Entropy & Diversity Score Calculation
# ---------------------------------------------------------------------------
try:
    # Single theme -> 0 entropy
    raw1, norm1 = calculate_shannon_entropy({"Natural Language Processing": 10})
    assert raw1 == 0.0
    assert norm1 == 0.0

    # Diverse across 4 equal themes -> High normalized entropy
    raw2, norm2 = calculate_shannon_entropy({
        "Natural Language Processing": 5,
        "Computer Vision & Image Processing": 5,
        "Machine Learning & Artificial Intelligence": 5,
        "Cybersecurity & Privacy": 5
    })
    assert raw2 > 1.0
    assert norm2 == 1.0

    print("PASS: Shannon entropy calculation clean")
except Exception as e:
    errors.append(f"FAIL: Shannon entropy calculation: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 4: Temporal Trend Analysis
# ---------------------------------------------------------------------------
try:
    pubs = [
        {"title": "Paper 1", "year": "2020"},
        {"title": "Paper 2", "year": "2021"},
        {"title": "Paper 3", "year": "2022"},
    ]
    lookup = {
        "Paper 1": "Natural Language Processing",
        "Paper 2": "Natural Language Processing",
        "Paper 3": "Natural Language Processing",
    }
    trend1, pattern1 = analyze_temporal_trend(pubs, lookup)
    assert pattern1 == "Stable Focus"
    assert len(trend1) == 3

    lookup_shifting = {
        "Paper 1": "Machine Learning & Artificial Intelligence",
        "Paper 2": "Natural Language Processing",
        "Paper 3": "Computer Vision & Image Processing",
    }
    trend2, pattern2 = analyze_temporal_trend(pubs, lookup_shifting)
    assert "Shifting" in pattern2 or "Interdisciplinary" in pattern2

    print("PASS: Temporal trend analysis clean")
except Exception as e:
    errors.append(f"FAIL: Temporal trend analysis: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 5: Research Profile Type Classification
# ---------------------------------------------------------------------------
try:
    assert classify_research_profile_type(10, 80.0) == "Specialist"
    assert classify_research_profile_type(10, 60.0) == "Focused Researcher"
    assert classify_research_profile_type(10, 40.0) == "Interdisciplinary"
    assert classify_research_profile_type(0, 0.0) == "No Publications"
    print("PASS: Research profile type classification clean")
except Exception as e:
    errors.append(f"FAIL: Research profile type classification: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 6: Multi-Module Regression Check (Modules 2, 3, 4, 5, 6, 7)
# ---------------------------------------------------------------------------
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    from analysis.research_profile import ResearchProfileAnalyser
    from analysis.supervision_analyser import SupervisionAnalyser
    from analysis.book_analyser import BookProfileAnalyser
    from analysis.patent_analyser import PatentProfileAnalyser
    from analysis.topic_analyser import TopicBreadthAnalyser
    print("PASS: All modules (2, 3, 4, 5, 6, 7) import cleanly together")
except Exception as e:
    errors.append(f"FAIL: Regression check for all modules: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 6 smoke tests PASSED for Module 7.")
