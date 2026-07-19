"""
test_module5.py  –  Comprehensive Test Suite for Module 5 (Books Analysis)
========================================================================

Tests:
  1. Clean import of Module 5 classes and functions.
  2. Authorship role classification (Sole, Lead, Co-Author, Contributing, Unknown).
  3. Publisher credibility evaluation (Recognized Academic, Self-Published, Unknown).
  4. ISBN validation for ISBN-10 and ISBN-13 (valid, invalid, empty).
  5. Verifiability and Data Quality Flagging.
  6. Zero-books candidate handling.
  7. Regression check: Modules 2, 3, 4, 5 all import cleanly.
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
    from analysis.book_verifier import (
        classify_authorship_role,
        evaluate_publisher_credibility,
        validate_isbn,
        check_book_quality_and_verifiability,
        is_valid_isbn10,
        is_valid_isbn13,
    )
    from analysis.book_analyser import BookProfileAnalyser, _rule_based_book_label
    from analysis import BookProfileAnalyser as BPA2
    print("PASS: Module 5 imports clean")
except Exception as e:
    errors.append(f"FAIL: Module 5 imports: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 2: Authorship Role Classification
# ---------------------------------------------------------------------------
try:
    # Sole Author
    assert classify_authorship_role("Fiza Murtaza", "Fiza Murtaza") == "Sole Author"
    # Lead Author
    assert classify_authorship_role("Fiza Murtaza, Ali Khan", "Fiza Murtaza") == "Lead Author"
    # Co-Author
    assert classify_authorship_role("Ali Khan, Fiza Murtaza, Sara Ahmed", "Fiza Murtaza") == "Co-Author"
    # Contributing / Chapter Author
    assert classify_authorship_role("Edited by John Doe (Chapter by Fiza Murtaza)", "Fiza Murtaza") == "Contributing Author"
    # Unknown (Candidate not in author list)
    assert classify_authorship_role("John Smith, Jane Doe", "Fiza Murtaza") == "Unknown"
    # Empty author
    assert classify_authorship_role("", "Fiza Murtaza") == "Unknown"
    print("PASS: Authorship role classification clean")
except Exception as e:
    errors.append(f"FAIL: Authorship role classification: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 3: Publisher Credibility Assessment
# ---------------------------------------------------------------------------
try:
    # Recognized Academic
    assert evaluate_publisher_credibility("Springer Nature") == "Recognized Academic"
    assert evaluate_publisher_credibility("Elsevier") == "Recognized Academic"
    assert evaluate_publisher_credibility("Cambridge University Press") == "Recognized Academic"
    assert evaluate_publisher_credibility("IEEE Press") == "Recognized Academic"
    assert evaluate_publisher_credibility("Higher Education Commission Pakistan") == "Recognized Academic"
    # Self-Published
    assert evaluate_publisher_credibility("Amazon KDP") == "Self-Published"
    assert evaluate_publisher_credibility("Lulu.com") == "Self-Published"
    assert evaluate_publisher_credibility("Independently Published") == "Self-Published"
    # Unknown
    assert evaluate_publisher_credibility("Random Local Printing Press") == "Unknown"
    assert evaluate_publisher_credibility("") == "Unknown"
    print("PASS: Publisher credibility assessment clean")
except Exception as e:
    errors.append(f"FAIL: Publisher credibility assessment: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 4: ISBN Validation
# ---------------------------------------------------------------------------
try:
    # Valid ISBN-10
    assert validate_isbn("0-596-52068-9") is True
    assert validate_isbn("0596520689") is True

    # Valid ISBN-13
    assert validate_isbn("978-3-16-148410-0") is True
    assert validate_isbn("9783161484100") is True

    # Invalid ISBN (bad checksum)
    assert validate_isbn("978-3-16-148410-9") is False
    assert validate_isbn("1234567890") is False

    # Empty / None
    assert validate_isbn(None) is None
    assert validate_isbn("") is None
    assert validate_isbn("nan") is None
    print("PASS: ISBN validation clean")
except Exception as e:
    errors.append(f"FAIL: ISBN validation: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 5: Verifiability & Data Quality Flags
# ---------------------------------------------------------------------------
try:
    # Valid ISBN + valid link
    row1 = {
        "title": "Machine Learning", "authors": "Fiza Murtaza", "publisher": "Springer",
        "year": "2021", "isbn": "978-3-16-148410-0", "link": "https://springer.com/book/123",
        "publisher_credibility": "Recognized Academic"
    }
    ver1, flags1 = check_book_quality_and_verifiability(row1, isbn_valid=True)
    assert ver1 is True
    assert flags1 == "OK"

    # Missing ISBN but valid link → Verifiable = True, flag includes MISSING_ISBN
    row2 = {
        "title": "Machine Learning 2", "authors": "Fiza Murtaza", "publisher": "Springer",
        "year": "2022", "link": "https://springer.com/book/456",
        "publisher_credibility": "Recognized Academic"
    }
    ver2, flags2 = check_book_quality_and_verifiability(row2, isbn_valid=None)
    assert ver2 is True
    assert "MISSING_ISBN" in flags2

    # No ISBN and no link → Unverifiable
    row3 = {
        "title": "My Notebook", "authors": "Fiza Murtaza", "publisher": "Self",
        "year": "2020", "publisher_credibility": "Self-Published"
    }
    ver3, flags3 = check_book_quality_and_verifiability(row3, isbn_valid=None)
    assert ver3 is False
    assert "UNVERIFIABLE" in flags3
    assert "SELF_PUBLISHED" in flags3

    print("PASS: Verifiability and quality flags clean")
except Exception as e:
    errors.append(f"FAIL: Verifiability & flags: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 6: Rule-Based Label Logic
# ---------------------------------------------------------------------------
try:
    assert _rule_based_book_label(0, 0, 0, 0, 0) == "No Books Listed"
    assert _rule_based_book_label(1, 1, 0, 1, 1) == "Strong"
    assert _rule_based_book_label(1, 0, 0, 0, 1) == "Moderate"
    assert _rule_based_book_label(1, 0, 0, 0, 0) == "Limited"
    print("PASS: Rule-based label logic clean")
except Exception as e:
    errors.append(f"FAIL: Rule-based label logic: {e}")
    print(errors[-1])

# ---------------------------------------------------------------------------
# Test 7: Regression Check (Modules 2, 3, 4, 5)
# ---------------------------------------------------------------------------
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    from analysis.research_profile import ResearchProfileAnalyser
    from analysis.supervision_analyser import SupervisionAnalyser
    from analysis.book_analyser import BookProfileAnalyser
    print("PASS: All modules (2, 3, 4, 5) import cleanly together")
except Exception as e:
    errors.append(f"FAIL: Regression check for all modules: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 7 smoke tests PASSED for Module 5.")
