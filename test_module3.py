import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
errors = []

# Test 1: Module 2 imports still work
try:
    from analysis.educational_profile import EducationalProfileAnalyser
    print("PASS: EducationalProfileAnalyser imports cleanly")
except Exception as e:
    errors.append(f"FAIL: Module 2 import: {e}")
    print(errors[-1])

# Test 2: Module 3 imports
try:
    from analysis.research_profile import ResearchProfileAnalyser, _compute_aggregate, _rule_based_research_label
    from analysis.journal_verifier import JournalVerifier, _classify_journal_quality
    from analysis.conference_verifier import ConferenceVerifier, _classify_conference_quality
    from analysis.authorship_detector import AuthorshipDetector
    print("PASS: All Module 3 classes import cleanly")
except Exception as e:
    errors.append(f"FAIL: Module 3 imports: {e}")
    print(errors[-1])

# Test 3: Authorship detection
try:
    det = AuthorshipDetector()
    r1 = det.detect_role("Muhammad Salman Qamar", "Muhammad Salman Qamar, Ihsan ul Haq, Muhammad Fahad Munir")
    r2 = det.detect_role("Muhammad Salman Qamar", "Ihsan ul Haq, Muhammad Fahad Munir, Muhammad Salman Qamar")
    r3 = det.detect_role("NOBODY HERE", "John Doe, Jane Smith")
    print(f"PASS: Authorship detection: first={r1}, last={r2}, notfound={r3}")
    assert r1 == "First Author", f"Expected First Author, got {r1}"
    assert "Corresponding" in r2, f"Expected Corresponding, got {r2}"
    assert r3 == "Unknown", f"Expected Unknown, got {r3}"
except Exception as e:
    errors.append(f"FAIL: AuthorshipDetector: {e}")
    print(errors[-1])

# Test 4: Journal quality labels
try:
    l1 = _classify_journal_quality(True, "Q1", True, False)
    l2 = _classify_journal_quality(None, "Cannot Verify", None, False)
    l3 = _classify_journal_quality(True, "Q1", True, True)
    assert l1 == "High Impact", f"Expected High Impact, got {l1}"
    assert l2 == "Unverified Venue", f"Expected Unverified Venue, got {l2}"
    assert l3 == "Potential Predatory", f"Expected Potential Predatory, got {l3}"
    print(f"PASS: Journal quality: {l1}, {l2}, {l3}")
except Exception as e:
    errors.append(f"FAIL: Journal quality: {e}")
    print(errors[-1])

# Test 5: Conference quality
try:
    c1 = _classify_conference_quality("A*", ["IEEE Xplore"])
    c2 = _classify_conference_quality("B", [])
    c3 = _classify_conference_quality(None, [])
    assert c1 == "Top-Tier (A*)", f"Expected Top-Tier, got {c1}"
    assert "Moderate" in c2, f"Expected Moderate in {c2}"
    print(f"PASS: Conference quality: {c1}, {c2}, {c3}")
except Exception as e:
    errors.append(f"FAIL: Conference quality: {e}")
    print(errors[-1])

# Test 6: Truncation detection
try:
    jv = JournalVerifier(skip_llm=True)
    t1 = jv._is_truncated("International Conference on En")
    t2 = jv._is_truncated("IEEE Access")
    t3 = jv._is_truncated("Multimedia Tools and Applications")
    assert t1 == True, f"Should be truncated: t1={t1}"
    assert t2 == False, f"Should NOT be truncated: t2={t2}"
    assert t3 == False, f"Should NOT be truncated: t3={t3}"
    print(f"PASS: Truncation detection: short={t1}, ieee={t2}, long={t3}")
except Exception as e:
    errors.append(f"FAIL: Truncation: {e}")
    print(errors[-1])

# Test 7: Edition number
try:
    cv = ConferenceVerifier(skip_llm=True)
    e1 = cv._detect_edition_number("13th IEEE International Conference on Signal Processing")
    e2 = cv._detect_edition_number("2nd International Conference on Machine Learning")
    e3 = cv._detect_edition_number("IEEE Access")
    assert e1 == 13, f"Expected 13, got {e1}"
    assert e2 == 2, f"Expected 2, got {e2}"
    assert e3 is None, f"Expected None, got {e3}"
    print(f"PASS: Edition detection: e1={e1}, e2={e2}, e3={e3}")
except Exception as e:
    errors.append(f"FAIL: Edition: {e}")
    print(errors[-1])

# Test 8: Proceedings indexing
try:
    cv2 = ConferenceVerifier(skip_llm=True)
    idx = cv2._detect_proceedings_indexing("IEEE 21st International Conference on Signal Processing")
    assert "IEEE Xplore" in idx, f"Expected IEEE Xplore in {idx}"
    idx2 = cv2._detect_proceedings_indexing("ACM SIGKDD Conference on Knowledge Discovery")
    assert "ACM Digital Library" in idx2, f"Expected ACM in {idx2}"
    print(f"PASS: Proceedings indexing: ieee={idx}, acm={idx2}")
except Exception as e:
    errors.append(f"FAIL: Proceedings: {e}")
    print(errors[-1])

# Test 9: Aggregate
try:
    agg = _compute_aggregate("TEST_CAND", [], [])
    assert agg["total_journals"] == 0
    assert agg["total_conferences"] == 0
    rule = _rule_based_research_label(agg)
    assert rule == "Needs Clarification", f"Expected Needs Clarification, got {rule}"
    print(f"PASS: Empty aggregate -> rule={rule}")
except Exception as e:
    errors.append(f"FAIL: Aggregate: {e}")
    print(errors[-1])

# Test 10: Beall list loading
try:
    jv2 = JournalVerifier(skip_llm=True)
    r_pred = jv2._is_predatory("IISTE Journals", None)
    r_legit = jv2._is_predatory("IEEE Transactions on Neural Networks", None)
    assert r_pred == True, f"IISTE should be predatory, got {r_pred}"
    assert r_legit == False, f"IEEE should NOT be predatory, got {r_legit}"
    print(f"PASS: Beall List: IISTE={r_pred}, IEEE={r_legit}")
except Exception as e:
    errors.append(f"FAIL: Beall list: {e}")
    print(errors[-1])

print()
if errors:
    print(f"FAILED: {len(errors)} test(s) failed")
    sys.exit(1)
else:
    print("All 10 smoke tests PASSED.")
