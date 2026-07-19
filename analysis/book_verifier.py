"""
book_verifier.py  –  Module 5 Helper: Authorship, Publisher Credibility & ISBN Verification
========================================================================================

WHY THIS FILE EXISTS
--------------------
Provides deterministic logic for Module 5 (Books Authored / Co-Authored):
  1. Authorship Role Classifier: Sole Author, Lead Author, Co-Author, Contributing Author, Unknown.
  2. Publisher Credibility Assessment: Recognized Academic, Self-Published, Unknown.
  3. ISBN Validation: Validates ISBN-10 and ISBN-13 formats and checksums.
  4. Verifiability & Data Quality Flagging.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz

try:
    import isbnlib
except ImportError:
    isbnlib = None

# ---------------------------------------------------------------------------
# Recognized Academic Publishers Reference List
# ---------------------------------------------------------------------------
RECOGNIZED_ACADEMIC_PUBLISHERS = [
    # Top International Academic & Science Publishers
    "Springer", "Springer Nature", "Springer-Verlag", "Springer International Publishing",
    "Elsevier", "Academic Press", "North-Holland",
    "Wiley", "John Wiley & Sons", "Wiley-Blackwell", "Wiley-IEEE Press",
    "CRC Press", "Taylor & Francis", "Routledge", "CRC",
    "Cambridge University Press", "Oxford University Press", "MIT Press",
    "IEEE", "IEEE Press", "ACM", "ACM Books",
    "Palgrave Macmillan", "Palgrave",
    "Sage", "SAGE Publications", "SAGE",
    "Edward Elgar", "Edward Elgar Publishing",
    "McGraw-Hill", "McGraw-Hill Education", "McGraw Hill",
    "Pearson", "Pearson Education", "Prentice Hall",
    "De Gruyter", "Walter de Gruyter",
    "World Scientific", "World Scientific Publishing",
    "Nova Science Publishers", "Nova Science",
    "IGI Global", "IGI Global Publishing",
    "IntechOpen", "Intech",
    "Harvard University Press", "Princeton University Press", "Yale University Press",
    "Columbia University Press", "Stanford University Press", "University of Chicago Press",
    "Bentham Science", "IOS Press", "MDPI", "Frontiers Media",
    # Regional / National Academic Institutions & Presses (Pakistan & South Asia)
    "Higher Education Commission", "HEC", "HEC Pakistan", "HEC Press",
    "National Book Foundation", "NBF", "NBF Pakistan",
    "Oxford University Press Pakistan", "OUP Pakistan",
    "University of Engineering and Technology", "UET Press",
    "NUST Publishing", "NUST Press", "FAST-NUST",
]

# ---------------------------------------------------------------------------
# Self-Published / Vanity Press Reference List
# ---------------------------------------------------------------------------
SELF_PUBLISHED_PLATFORMS = [
    "Amazon", "Amazon KDP", "Kindle Direct Publishing", "CreateSpace",
    "Lulu", "Lulu.com", "Lulu Press",
    "AuthorHouse", "Xlibris", "iUniverse", "BookSurge", "Vantage Press",
    "PublishDrive", "Smashwords", "Draft2Digital", "Blurb", "Kobo Writing Life",
    "Self-Published", "Independent", "Independently Published", "Independently published",
]


# ---------------------------------------------------------------------------
# Native ISBN Validation Fallback
# ---------------------------------------------------------------------------

def _clean_isbn_string(isbn_str: str) -> str:
    """Remove hyphens, spaces, and convert X to uppercase."""
    if not isbn_str:
        return ""
    cleaned = re.sub(r"[^0-9Xx]", "", str(isbn_str).strip())
    return cleaned.upper()


def is_valid_isbn10(isbn: str) -> bool:
    """Validate ISBN-10 checksum."""
    s = _clean_isbn_string(isbn)
    if len(s) != 10:
        return False
    if not s[:9].isdigit():
        return False
    val = 0
    for i in range(9):
        val += int(s[i]) * (10 - i)
    checksum = 10 if s[9] == 'X' else (int(s[9]) if s[9].isdigit() else -1)
    if checksum < 0:
        return False
    val += checksum
    return val % 11 == 0


def is_valid_isbn13(isbn: str) -> str:
    """Validate ISBN-13 checksum."""
    s = _clean_isbn_string(isbn)
    if len(s) != 13 or not s.isdigit():
        return False
    val = sum(int(s[i]) * (1 if i % 2 == 0 else 3) for i in range(12))
    check = (10 - (val % 10)) % 10
    return check == int(s[12])


def validate_isbn(isbn_str: Optional[str]) -> Optional[bool]:
    """
    Validates an ISBN string.
    Returns:
        True  - Valid ISBN-10 or ISBN-13
        False - Format provided but failed validation/checksum
        None  - No ISBN provided / empty / NaN
    """
    if not isbn_str or str(isbn_str).strip().lower() in ("nan", "none", "", "null", "n/a"):
        return None

    cleaned = _clean_isbn_string(isbn_str)
    if not cleaned:
        return None

    # Use isbnlib if available
    if isbnlib is not None:
        try:
            return isbnlib.is_isbn10(cleaned) or isbnlib.is_isbn13(cleaned)
        except Exception:
            pass

    # Native fallback check
    return is_valid_isbn10(cleaned) or is_valid_isbn13(cleaned)


# ---------------------------------------------------------------------------
# Authorship Role Classification
# ---------------------------------------------------------------------------

def _normalize_name_token(name: str) -> str:
    name = re.sub(r"[,.\-–]", " ", name)
    return " ".join(name.lower().split())


def classify_authorship_role(
    authors_str: Optional[str],
    candidate_name: str,
    threshold: int = 80
) -> str:
    """
    Determines the candidate's authorship role for a book:
      - Sole Author: only candidate listed
      - Lead Author: candidate listed first among multiple authors
      - Co-Author: candidate listed among multiple authors, not first
      - Contributing Author: explicit chapter/contributor keywords present
      - Unknown: candidate not found in authors or authors missing
    """
    if not authors_str or str(authors_str).strip().lower() in ("nan", "none", "", "null"):
        return "Unknown"

    authors_raw = str(authors_str).strip()
    norm_cand = _normalize_name_token(candidate_name)

    # Check for chapter / contributing indications
    is_chapter_edited = any(
        kw in authors_raw.lower()
        for kw in ["editor", "ed.", "eds.", "chapter", "contributing", "contributor"]
    )

    # Split author names by common separators (comma, and, &, semicolon)
    # Be careful with "ed. by" or "edited by"
    clean_authors = re.sub(r"\(ed\.\)|\(eds\.\)|edited by", "", authors_raw, flags=re.IGNORECASE)
    author_list = [a.strip() for a in re.split(r"[,;&]|and\b", clean_authors) if a.strip()]

    if not author_list:
        return "Unknown"

    # Find position of candidate in author_list using fuzzy match
    cand_index = -1
    for idx, auth in enumerate(author_list):
        norm_auth = _normalize_name_token(auth)
        if fuzz.token_sort_ratio(norm_cand, norm_auth) >= threshold:
            cand_index = idx
            break

    if cand_index == -1:
        # Candidate not matched in author list
        if is_chapter_edited:
            return "Contributing Author"
        return "Unknown"

    num_authors = len(author_list)

    if num_authors == 1:
        if is_chapter_edited:
            return "Contributing Author"
        return "Sole Author"

    if cand_index == 0:
        if is_chapter_edited:
            return "Contributing Author"
        return "Lead Author"

    if is_chapter_edited:
        return "Contributing Author"

    return "Co-Author"


# ---------------------------------------------------------------------------
# Publisher Credibility Assessment
# ---------------------------------------------------------------------------

def evaluate_publisher_credibility(
    publisher_str: Optional[str],
    threshold: int = 85
) -> str:
    """
    Evaluates publisher credibility:
      - Recognized Academic
      - Self-Published
      - Unknown
    """
    if not publisher_str or str(publisher_str).strip().lower() in ("nan", "none", "", "null"):
        return "Unknown"

    pub_clean = str(publisher_str).strip()
    pub_lower = pub_clean.lower()

    # Direct substring / keyword checks first for high-confidence match
    for self_pub in SELF_PUBLISHED_PLATFORMS:
        if self_pub.lower() in pub_lower:
            return "Self-Published"

    for recog in RECOGNIZED_ACADEMIC_PUBLISHERS:
        recog_lower = recog.lower()
        if recog_lower in pub_lower or pub_lower in recog_lower:
            return "Recognized Academic"

    # Fuzzy matching against recognized list
    for recog in RECOGNIZED_ACADEMIC_PUBLISHERS:
        if fuzz.partial_ratio(pub_lower, recog.lower()) >= threshold:
            return "Recognized Academic"

    # Fuzzy matching against self-published list
    for self_pub in SELF_PUBLISHED_PLATFORMS:
        if fuzz.partial_ratio(pub_lower, self_pub.lower()) >= threshold:
            return "Self-Published"

    return "Unknown"


# ---------------------------------------------------------------------------
# Data Quality Flags & Verifiability Helper
# ---------------------------------------------------------------------------

def check_book_quality_and_verifiability(
    row: Dict[str, Any],
    isbn_valid: Optional[bool]
) -> Tuple[bool, str]:
    """
    Determines if a book record is verifiable and computes a data_quality_flags string.
      - Verifiable: True if valid ISBN OR valid online link present, else False.
      - Flags: MISSING_TITLE, MISSING_AUTHORS, MISSING_PUBLISHER, MISSING_YEAR,
               MISSING_ISBN, INVALID_ISBN, MISSING_LINK, UNVERIFIABLE, SELF_PUBLISHED
    """
    flags = []

    title = str(row.get("title") or "").strip()
    if not title or title.lower() in ("nan", "none", ""):
        flags.append("MISSING_TITLE")

    authors = str(row.get("authors") or "").strip()
    if not authors or authors.lower() in ("nan", "none", ""):
        flags.append("MISSING_AUTHORS")

    publisher = str(row.get("publisher") or "").strip()
    if not publisher or publisher.lower() in ("nan", "none", ""):
        flags.append("MISSING_PUBLISHER")

    year = str(row.get("year") or "").strip()
    if not year or year.lower() in ("nan", "none", ""):
        flags.append("MISSING_YEAR")

    isbn = str(row.get("isbn") or "").strip()
    has_isbn = bool(isbn and isbn.lower() not in ("nan", "none", ""))
    if not has_isbn:
        flags.append("MISSING_ISBN")
    elif isbn_valid is False:
        flags.append("INVALID_ISBN")

    link = str(row.get("link") or row.get("online_link") or "").strip()
    has_link = bool(link and link.lower() not in ("nan", "none", "") and link.startswith("http"))
    if not has_link:
        flags.append("MISSING_LINK")

    verifiable = bool((has_isbn and isbn_valid is True) or has_link)
    if not verifiable:
        flags.append("UNVERIFIABLE")

    if row.get("publisher_credibility") == "Self-Published":
        flags.append("SELF_PUBLISHED")

    flag_str = " | ".join(flags) if flags else "OK"
    return verifiable, flag_str
