import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
errors = []

# Test 1: imports
try:
    from analysis.supervision_analyser import (
        SupervisionAnalyser, find_joint_papers,
        _rule_based_supervision_label, _normalize_name
    )
    from analysis import SupervisionAnalyser as SA2
    print("PASS: All Module 4 imports OK")
except Exception as e:
    errors.append(f"FAIL: imports: {e}")
    print(errors[-1])

# Test 2: name normalization
try:
    assert _normalize_name("Hassan, Ali") == "hassan ali"
    assert _normalize_name("M. Ali Hassan") == "m ali hassan"
    print("PASS: Name normalization OK")
except Exception as e:
    errors.append(f"FAIL: normalize: {e}")
    print(errors[-1])

# Test 3: rule-based labels
try:
    assert _rule_based_supervision_label(0,0,0,0,0,True)  == "Needs Clarification"
    assert _rule_based_supervision_label(0,0,0,0,0,False) == "Needs Clarification"
    assert _rule_based_supervision_label(5,0,0,0,0,False) == "Strong"
    assert _rule_based_supervision_label(0,0,2,0,0,False) == "Strong"
    assert _rule_based_supervision_label(2,0,0,0,0,False) == "Moderate"
    assert _rule_based_supervision_label(1,0,0,0,0,False) == "Limited"
    print("PASS: Rule-based labels OK")
except Exception as e:
    errors.append(f"FAIL: labels: {e}")
    print(errors[-1])

# Test 4: find_joint_papers - match found
try:
    pubs = [
        {"title": "Deep Learning for NLP", "year": "2022", "venue": "IEEE Access",
         "authors": "Waqas Toor, Ali Hassan, John Smith", "doi": None},
    ]
    results = find_joint_papers("Ali Hassan", pubs, "Waqas Toor")
    assert len(results) == 1, f"Expected 1 match, got {len(results)}"
    assert results[0]["student_position"] == "Author 2"
    cpos = results[0]["candidate_position"]
    assert "First" in cpos or "before" in cpos, f"Unexpected cpos: {cpos}"
    print(f"PASS: Joint paper match: candidate_position={cpos}")
except Exception as e:
    errors.append(f"FAIL: joint_papers match: {e}")
    print(errors[-1])

# Test 5: find_joint_papers - no match
try:
    pubs = [
        {"title": "Some Paper", "year": "2020", "venue": "Nature",
         "authors": "John Doe, Jane Smith", "doi": None},
    ]
    results = find_joint_papers("Completely Different Name", pubs, "John Doe")
    assert len(results) == 0, f"Expected 0 matches, got {len(results)}"
    print("PASS: No-match case OK")
except Exception as e:
    errors.append(f"FAIL: no-match: {e}")
    print(errors[-1])

# Test 6: empty/nan student name
try:
    assert find_joint_papers("", [], "Anyone") == []
    assert find_joint_papers("nan", [], "Anyone") == []
    print("PASS: Empty/nan student name handled OK")
except Exception as e:
    errors.append(f"FAIL: empty student: {e}")
    print(errors[-1])

# Test 7: corresponding/last-author detection
try:
    pubs = [
        {"title": "Paper X", "year": "2023", "venue": "Journal Y",
         "authors": "Ali Hassan, Bob Jones, Waqas Toor", "doi": None},
    ]
    results = find_joint_papers("Ali Hassan", pubs, "Waqas Toor")
    assert len(results) == 1
    cpos = results[0]["candidate_position"]
    assert "Corresponding" in cpos or "Last" in cpos, f"Expected last-author, got: {cpos}"
    print(f"PASS: Corresponding/last author detection: {cpos}")
except Exception as e:
    errors.append(f"FAIL: last author: {e}")
    print(errors[-1])

# Test 8: Modules 2 and 3 still load
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    from analysis.research_profile import ResearchProfileAnalyser
    print("PASS: Modules 2 and 3 still import cleanly")
except Exception as e:
    errors.append(f"FAIL: existing modules: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s)")
    sys.exit(1)
else:
    print("All 8 smoke tests PASSED.")
