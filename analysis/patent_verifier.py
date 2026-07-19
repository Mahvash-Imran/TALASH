"""
patent_verifier.py  –  Module 6 Helper: Inventor Role, Jurisdiction & Verification Logic
=======================================================================================

WHY THIS FILE EXISTS
--------------------
Provides deterministic logic for Module 6 (Patents Analysis):
  1. Inventor Role Classification: Sole Inventor, Lead Inventor, Co-Inventor, Contributing Innovator, Unknown.
  2. Country & Jurisdiction Classification: National (Pakistan) vs International (USPTO, WIPO, EPO, GE, etc.).
  3. Patent Verification & Link Resolution: Validates online link or constructs Google Patents query URL.
  4. Data Quality Flagging.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Country / Jurisdiction Lookup Reference Sets
# ---------------------------------------------------------------------------
PAKISTAN_KEYWORDS = {"pakistan", "pk", "ipo", "ipo-pakistan", "ipo pakistan", "islamabad"}

INTERNATIONAL_PATENT_OFFICE_PREFIXES = {
    "US": "USA (USPTO)",
    "GE": "Germany / International (DPMA/GE)",
    "DE": "Germany (DPMA)",
    "EP": "European Patent Office (EPO)",
    "WO": "WIPO / PCT",
    "PCT": "WIPO / PCT",
    "CN": "China (CNIPA)",
    "JP": "Japan (JPO)",
    "KR": "South Korea (KIPO)",
    "GB": "United Kingdom (UKIPO)",
    "UK": "United Kingdom (UKIPO)",
    "CA": "Canada (CIPO)",
    "AU": "Australia (IP Australia)",
}


def normalize_name_token(name: str) -> str:
    name = re.sub(r"[,.\-–]", " ", name)
    return " ".join(name.lower().split())


# ---------------------------------------------------------------------------
# Inventor Role Classification
# ---------------------------------------------------------------------------

def _is_initials(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    cleaned = s.replace(".", "").strip()
    return len(cleaned) <= 3 and (s.isupper() or cleaned.isupper())


def _parse_inventor_list(inv_raw: str) -> List[str]:
    """Parse raw inventors string into individual inventor names."""
    clean_inv = re.sub(r"inventors?:?", "", inv_raw, flags=re.IGNORECASE).strip()
    if not clean_inv:
        return []

    # Split by semicolon, 'and', or '&'
    parts = [a.strip() for a in re.split(r";|\band\b|&", clean_inv) if a.strip()]
    final_list = []

    for p in parts:
        sub_items = [x.strip() for x in p.split(",") if x.strip()]
        i = 0
        while i < len(sub_items):
            item = sub_items[i]
            if i + 1 < len(sub_items) and _is_initials(sub_items[i + 1]):
                final_list.append(f"{item}, {sub_items[i + 1]}")
                i += 2
            else:
                final_list.append(item)
                i += 1

    return final_list if final_list else [clean_inv]


def classify_inventor_role(
    inventors_str: Optional[str],
    candidate_name: str,
    threshold: int = 80
) -> str:
    """
    Classifies candidate's inventor role:
      - Sole Inventor: Candidate is the single listed inventor.
      - Lead Inventor: Candidate listed first among multiple inventors.
      - Co-Inventor: Candidate listed 2nd or 3rd among multiple inventors.
      - Contributing Innovator: Candidate listed 4th or later in a larger team.
      - Unknown: Candidate name not found in inventors string or inventors string empty.
    """
    if not inventors_str or str(inventors_str).strip().lower() in ("nan", "none", "", "null"):
        return "Unknown"

    inv_raw = str(inventors_str).strip()
    norm_cand = normalize_name_token(candidate_name)

    inventor_list = _parse_inventor_list(inv_raw)
    if not inventor_list:
        return "Unknown"

    cand_index = -1
    for idx, inv in enumerate(inventor_list):
        norm_inv = normalize_name_token(inv)
        if fuzz.token_sort_ratio(norm_cand, norm_inv) >= threshold:
            cand_index = idx
            break

    if cand_index == -1:
        return "Unknown"

    num_inventors = len(inventor_list)

    if num_inventors == 1:
        return "Sole Inventor"
    if cand_index == 0:
        return "Lead Inventor"
    if cand_index in (1, 2):
        return "Co-Inventor"
    return "Contributing Innovator"


# ---------------------------------------------------------------------------
# Country of Filing & Jurisdiction Classification
# ---------------------------------------------------------------------------

def classify_jurisdiction(
    country_str: Optional[str],
    patent_number_str: Optional[str]
) -> Tuple[str, str]:
    """
    Returns (country_normalized, jurisdiction):
      - jurisdiction: "National (Pakistan)", "International", or "Unknown"
    """
    country_clean = str(country_str or "").strip()
    pat_num_clean = str(patent_number_str or "").strip()

    country_lower = country_clean.lower()

    # Check explicit country string first
    if any(k in country_lower for k in PAKISTAN_KEYWORDS):
        return "Pakistan", "National (Pakistan)"

    if country_clean and country_lower not in ("nan", "none", "", "null"):
        return country_clean, "International"

    # Infer jurisdiction from patent number format
    if pat_num_clean and pat_num_clean.lower() not in ("nan", "none", "", "null"):
        # Match pattern like "2020-GE-730032" or "US10123456B2" or "WO2021123456"
        m = re.search(r"\b([A-Za-z]{2,3})\b", pat_num_clean)
        if m:
            code = m.group(1).upper()
            if code in ("PK", "PAK"):
                return "Pakistan", "National (Pakistan)"
            if code in INTERNATIONAL_PATENT_OFFICE_PREFIXES:
                return INTERNATIONAL_PATENT_OFFICE_PREFIXES[code], "International"

        return "International (Inferred)", "International"

    return "Unknown", "Unknown"


# ---------------------------------------------------------------------------
# Patent Verification & Link Building
# ---------------------------------------------------------------------------

def build_patent_verification_link(
    patent_number: Optional[str],
    title: Optional[str],
    existing_link: Optional[str]
) -> Tuple[Optional[str], bool]:
    """
    Returns (verification_link, verifiable):
      - verification_link: URL to public database if present/constructible, else None
      - verifiable: True if valid link or constructible search URL exists, else False
    """
    link_clean = str(existing_link or "").strip()
    pat_clean = str(patent_number or "").strip()
    title_clean = str(title or "").strip()

    has_link = bool(link_clean and link_clean.lower() not in ("nan", "none", "") and link_clean.startswith("http"))
    if has_link:
        return link_clean, True

    has_pat = bool(pat_clean and pat_clean.lower() not in ("nan", "none", ""))
    if has_pat:
        # Sanitize patent number for Google Patents URL
        sanitized = re.sub(r"[^\w]", "", pat_clean)
        constructed_url = f"https://patents.google.com/patent/{sanitized}/en"
        return constructed_url, True

    has_title = bool(title_clean and title_clean.lower() not in ("nan", "none", ""))
    if has_title:
        query = re.sub(r"\s+", "+", title_clean)
        constructed_url = f"https://patents.google.com/?q={query}"
        return constructed_url, True

    return None, False


# ---------------------------------------------------------------------------
# Data Quality Flagging
# ---------------------------------------------------------------------------

def check_patent_quality_flags(row: Dict[str, Any], verifiable: bool) -> str:
    """
    Generates a pipe-separated string of data quality flags for a patent row.
    """
    flags = []

    title = str(row.get("title") or "").strip()
    if not title or title.lower() in ("nan", "none", ""):
        flags.append("MISSING_TITLE")

    pat_num = str(row.get("patent_number") or "").strip()
    if not pat_num or pat_num.lower() in ("nan", "none", ""):
        flags.append("MISSING_PATENT_NUMBER")

    inventors = str(row.get("inventors") or "").strip()
    if not inventors or inventors.lower() in ("nan", "none", ""):
        flags.append("MISSING_INVENTORS")

    country = str(row.get("country") or "").strip()
    if not country or country.lower() in ("nan", "none", ""):
        flags.append("MISSING_COUNTRY")

    date_str = str(row.get("date") or "").strip()
    if not date_str or date_str.lower() in ("nan", "none", ""):
        flags.append("MISSING_DATE")

    if not verifiable:
        flags.append("UNVERIFIABLE")

    if row.get("inventor_role") == "Unknown":
        flags.append("INVENTOR_ROLE_UNDETECTABLE")

    return " | ".join(flags) if flags else "OK"
