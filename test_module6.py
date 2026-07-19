"""
test_module6.py  –  Comprehensive Test Suite for Module 6 (Patents Analysis)
==========================================================================

Tests:
  1. Clean import of Module 6 classes and functions.
  2. Inventor role classification (Sole, Lead, Co-Inventor, Contributing, Unknown).
  3. Filing jurisdiction classification (National (Pakistan), International, Unknown).
  4. Patent verification & Google Patents URL resolution.
  5. Data quality flagging.
  6. Rule-based innovation label logic.
  7. Regression check: Modules 2, 3, 4, 5, 6 all import cleanly together.
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
    from analysis.patent_verifier import (
        classify_inventor_role,
        classify_jurisdiction,
        build_patent_verification_link,
        check_patent_quality_flags,
    )
    from analysis.patent_analyser import PatentProfileAnalyser, _rule_based_patent_label
    from analysis import PatentProfileAnalyser as PPA2
    print("PASS: Module 6 imports clean")
except Exception as e:
    errors.append(f"FAIL: Module 6 imports: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 2: Inventor Role Classification
# ---------------------------------------------------------------------------
try:
    # Sole Inventor
    assert classify_inventor_role("Muhammad Farrukh", "Muhammad Farrukh") == "Sole Inventor"
    # Lead Inventor
    assert classify_inventor_role("Muhammad Farrukh, Ali Khan", "Muhammad Farrukh") == "Lead Inventor"
    # Co-Inventor (2nd / 3rd position)
    assert classify_inventor_role("Qureshi, M.F., Khalid, S., Muhammad Farrukh", "Muhammad Farrukh") == "Co-Inventor"
    # Contributing Innovator (4th+ position)
    assert classify_inventor_role("Auth1, Auth2, Auth3, Muhammad Farrukh, Auth5", "Muhammad Farrukh") == "Contributing Innovator"
    # Unknown (Not in list)
    assert classify_inventor_role("John Doe, Jane Smith", "Muhammad Farrukh") == "Unknown"
    # Empty string
    assert classify_inventor_role("", "Muhammad Farrukh") == "Unknown"
    print("PASS: Inventor role classification clean")
except Exception as e:
    errors.append(f"FAIL: Inventor role classification: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 3: Filing Jurisdiction Classification
# ---------------------------------------------------------------------------
try:
    # Explicit Pakistan
    c1, j1 = classify_jurisdiction("Pakistan", None)
    assert j1 == "National (Pakistan)"

    # Explicit USA
    c2, j2 = classify_jurisdiction("USA", None)
    assert j2 == "International"

    # Patent number fallback for German/International patent (e.g. 2020-GE-730032)
    c3, j3 = classify_jurisdiction("", "2020-GE-730032")
    assert j3 == "International"
    assert "GE" in c3 or "Germany" in c3 or "International" in c3

    # Patent number fallback for US patent (e.g. US10123456B2)
    c4, j4 = classify_jurisdiction(None, "US10123456B2")
    assert j4 == "International"

    # Unknown
    c5, j5 = classify_jurisdiction(None, None)
    assert j5 == "Unknown"
    print("PASS: Jurisdiction classification clean")
except Exception as e:
    errors.append(f"FAIL: Jurisdiction classification: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 4: Patent Verification & Link Building
# ---------------------------------------------------------------------------
try:
    # Explicit link
    link1, ver1 = build_patent_verification_link("123", "Title", "https://patents.google.com/patent/123")
    assert ver1 is True
    assert link1 == "https://patents.google.com/patent/123"

    # Patent number link construction
    link2, ver2 = build_patent_verification_link("US10123456B2", "Title", None)
    assert ver2 is True
    assert "patents.google.com/patent/US10123456B2" in link2

    # Title-only search construction
    link3, ver3 = build_patent_verification_link(None, "Fuel Flow Sensor", "")
    assert ver3 is True
    assert "patents.google.com/?q=Fuel+Flow+Sensor" in link3

    # No info → Unverifiable
    link4, ver4 = build_patent_verification_link("", "", "")
    assert ver4 is False
    assert link4 is None
    print("PASS: Patent verification & link resolution clean")
except Exception as e:
    errors.append(f"FAIL: Verification & link resolution: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 5: Quality Flags & Rule-Based Label
# ---------------------------------------------------------------------------
try:
    row = {
        "title": "Fuel Sensor", "patent_number": "123", "inventors": "Ahmed Naeem",
        "country": "Pakistan", "date": "2020", "inventor_role": "Sole Inventor"
    }
    flags = check_patent_quality_flags(row, verifiable=True)
    assert flags == "OK"

    # Test label
    assert _rule_based_patent_label(0, 0, 0, 0) == "No Patents Listed"
    assert _rule_based_patent_label(1, 1, 1, 1) == "Strong"
    assert _rule_based_patent_label(1, 0, 0, 1) == "Moderate"
    print("PASS: Quality flags & label logic clean")
except Exception as e:
    errors.append(f"FAIL: Quality flags & label logic: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 6: Multi-Module Regression Check (Modules 2, 3, 4, 5, 6)
# ---------------------------------------------------------------------------
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    from analysis.research_profile import ResearchProfileAnalyser
    from analysis.supervision_analyser import SupervisionAnalyser
    from analysis.book_analyser import BookProfileAnalyser
    from analysis.patent_analyser import PatentProfileAnalyser
    print("PASS: All modules (2, 3, 4, 5, 6) import cleanly together")
except Exception as e:
    errors.append(f"FAIL: Regression check for all modules: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 6 smoke tests PASSED for Module 6.")
