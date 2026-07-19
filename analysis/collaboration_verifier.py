"""
collaboration_verifier.py  –  Module 8 Helper: Co-Author Parsing, Normalization & Metrics
========================================================================================

WHY THIS FILE EXISTS
--------------------
Provides deterministic logic for Module 8 (Co-Author Collaboration Analysis):
  1. Author List Parser & Normalizer: Filters candidate out, strips initials variants, handles 'et al.'.
  2. Recurring vs. One-Time Collaborator Analyzer.
  3. Team Size Profiler: Solo/Small-Group (<3), Medium-Group (3-5), Large-Group (>5).
  4. Student Co-Author Cross-Referencer (via rapidfuzz).
  5. Collaboration Diversity Index Calculation.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz


def normalize_author_name(name: str) -> str:
    """
    Normalizes an author's name string:
      - Strips 'et al.', titles ('Dr.', 'Prof.', etc.)
      - Cleans punctuation and collapses spaces
      - Converts to title case for uniform reporting
    """
    if not name:
        return ""

    s = str(name).strip()

    # Remove 'et al.', 'et al', 'and others'
    s = re.sub(r"\b(et\s+al\.?|and\s+others)\b", "", s, flags=re.IGNORECASE).strip()

    # Remove honorifics/titles
    s = re.sub(r"\b(Dr|Prof|Mr|Mrs|Ms|Engr|PhD|Ph\.D)\b\.?\s*", "", s, flags=re.IGNORECASE).strip()

    # Clean punctuation
    s = re.sub(r"[,.\-–]", " ", s)

    # Collapse extra whitespace
    tokens = [t for t in s.split() if t]
    if not tokens:
        return ""

    # Title-case normalized name
    return " ".join(t.capitalize() for t in tokens)


def parse_coauthors(
    authors_str: Optional[str],
    candidate_name: str,
    threshold: int = 80
) -> Tuple[List[str], int]:
    """
    Parses a paper's full author string into normalized co-author names,
    excluding the candidate.
    Returns: (list_of_coauthors, total_author_count_including_candidate)
    """
    if not authors_str or str(authors_str).strip().lower() in ("nan", "none", "", "null"):
        return [], 0

    raw = str(authors_str).strip()

    # Handle 'et al.' in raw text
    has_et_al = bool(re.search(r"\bet\s+al", raw, re.IGNORECASE))
    clean_raw = re.sub(r"\b(et\s+al\.?|and\s+others)\b", "", raw, flags=re.IGNORECASE)

    # Split by semicolon, 'and', '&', or comma
    if ";" in clean_raw:
        parts = [a.strip() for a in re.split(r";|\band\b|&", clean_raw) if a.strip()]
    else:
        parts = [a.strip() for a in re.split(r",|\band\b|&", clean_raw) if a.strip()]

    norm_cand = normalize_author_name(candidate_name).lower()
    coauthors = []
    total_authors_count = 0

    for ra in parts:
        norm_auth = normalize_author_name(ra)
        if not norm_auth:
            continue
        total_authors_count += 1

        # Check if this author is the candidate
        if norm_cand and max(fuzz.token_sort_ratio(norm_cand, norm_auth.lower()), fuzz.token_set_ratio(norm_cand, norm_auth.lower())) >= threshold:
            continue

        coauthors.append(norm_auth)

    if has_et_al:
        total_authors_count += 3  # Estimate +3 for et al.

    # Ensure total_authors_count is at least max(len(coauthors) + (1 if candidate), 1)
    total_authors_count = max(total_authors_count, len(coauthors) + (1 if norm_cand else 0))

    return coauthors, total_authors_count


def classify_team_size_profile(avg_authors_per_paper: float, total_pubs: int) -> str:
    """
    Classifies team size profile based on average authors per paper:
      - No Publications: 0 papers
      - Solo/Small-Group Researcher: < 3 authors average
      - Medium-Group Researcher: 3 <= avg <= 5 authors average
      - Large-Group Researcher: > 5 authors average
    """
    if total_pubs == 0:
        return "No Publications"
    if avg_authors_per_paper < 3.0:
        return "Solo/Small-Group Researcher"
    if avg_authors_per_paper <= 5.0:
        return "Medium-Group Researcher"
    return "Large-Group Researcher"


def match_student_coauthors(
    coauthors_list: List[str],
    student_names: List[str],
    threshold: int = 80
) -> Set[str]:
    """
    Matches co-authors against a list of supervised student names using rapidfuzz.
    Returns a set of matched student names.
    """
    matched = set()
    if not coauthors_list or not student_names:
        return matched

    norm_students = [
        (s, normalize_author_name(s).lower())
        for s in student_names
        if str(s).strip().lower() not in ("nan", "none", "")
    ]
    if not norm_students:
        return matched

    for co in coauthors_list:
        norm_co = normalize_author_name(co).lower()
        if not norm_co:
            continue

        for s_orig, s_norm in norm_students:
            score = max(fuzz.token_sort_ratio(norm_co, s_norm), fuzz.token_set_ratio(norm_co, s_norm))
            if score >= threshold:
                matched.add(s_orig)
                break

    return matched


def compute_collaboration_metrics(
    coauthor_counts: Dict[str, int],
    total_papers: int,
    total_authors_sum: int,
    student_matches_count: int,
) -> Dict[str, Any]:
    """
    Computes summary collaboration metrics for a candidate.
    """
    if total_papers == 0:
        return {
            "total_unique_coauthors": 0,
            "recurring_collaborators_count": 0,
            "one_time_collaborators": 0,
            "avg_authors_per_paper": 0.0,
            "student_collaborations": 0,
            "collaboration_diversity_score": 0.0,
            "team_size_profile": "No Publications",
            "top_collaborators": [],
            "collaboration_strength_label": "No Publications",
        }

    total_unique = len(coauthor_counts)
    recurring = [
        {"name": name, "paper_count": count}
        for name, count in coauthor_counts.items()
        if count >= 2
    ]
    recurring_sorted = sorted(recurring, key=lambda x: x["paper_count"], reverse=True)

    one_time_count = sum(1 for count in coauthor_counts.values() if count == 1)
    recurring_count = len(recurring)

    avg_authors = round(total_authors_sum / total_papers, 1)
    team_profile = classify_team_size_profile(avg_authors, total_papers)

    # Diversity Score: Ratio of one-time collaborators to total unique coauthors
    if total_unique > 0:
        diversity_score = round(one_time_count / total_unique, 2)
    else:
        diversity_score = 0.0

    # Collaboration strength label
    if total_unique >= 15 or diversity_score >= 0.70:
        strength_label = "Broad Network"
    elif total_unique >= 5:
        strength_label = "Balanced Network"
    elif total_unique > 0:
        strength_label = "Closed Network"
    else:
        strength_label = "Solo Researcher"

    return {
        "total_unique_coauthors": total_unique,
        "recurring_collaborators_count": recurring_count,
        "one_time_collaborators": one_time_count,
        "avg_authors_per_paper": avg_authors,
        "student_collaborations": student_matches_count,
        "collaboration_diversity_score": diversity_score,
        "team_size_profile": team_profile,
        "top_collaborators": recurring_sorted[:5],
        "collaboration_strength_label": strength_label,
    }
