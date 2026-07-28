"""
email_drafter.py  –  Candidate-Specific Missing-Info Detection & Email Drafting
================================================================================

WHY THIS FILE EXISTS
--------------------
Detects genuinely missing or unjustified records per candidate using actual data
from all analysis modules. Drafts a targeted, professional follow-up email that
only requests items that are provably absent or flagged — never asks about
things not mentioned in the candidate's CV.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_EMAIL_SYSTEM_PROMPT = (
    "You are a senior academic recruitment officer for a faculty selection committee. "
    "You will receive a candidate's profile and a list of specific, data-verified items "
    "that require clarification or additional documentation. "
    "Write a polite, formal, and highly personalized follow-up email. "
    "Do not use generic phrasing. Each email must feel individually composed. "
    "Do not repeat the same sentence structure across multiple candidates. "
    "Sign off as: TALASH Team."
)

_EMAIL_USER_TEMPLATE = (
    "Candidate: {candidate_name} (ID: {candidate_id})\n\n"
    "ITEMS REQUIRING CLARIFICATION OR DOCUMENTATION:\n"
    "{missing_items_numbered}\n\n"
    "Write a Subject line and formal email Body. Sign off as: TALASH Team."
)

# --- Gap classification helpers ---

_EDU_TRANSITION_KEYWORDS = ["matric", "fsc", "f.sc", "hssc", "ssc", "pre-engineering",
                             "pre engineering", "a-levels", "a levels", "o-level", "science @ "]


def _is_routine_edu_transition(gap_row: pd.Series) -> bool:
    """
    Returns True if a gap is a normal education-to-education transition
    (e.g., matric → intermediate → BS) that is routine in Pakistan's academic system
    and should NOT be flagged in the email.
    """
    prev = str(gap_row.get("prev_activity", "")).lower()
    nxt  = str(gap_row.get("next_activity", "")).lower()
    dur  = float(gap_row.get("duration_months", 0) or 0)

    both_edu = prev.startswith("[education]") and nxt.startswith("[education]")
    if not both_edu:
        return False

    # Short edu-to-edu gap (≤30 months) between pre-degree and degree = normal
    if dur <= 30:
        for kw in _EDU_TRANSITION_KEYWORDS:
            if kw in prev:
                return True

    return False


def _extract_career_gaps(candidate_id: str, gap_df: pd.DataFrame) -> List[Dict]:
    """
    Returns only genuine career gaps: unjustified gaps that are NOT
    routine education-to-education transitions.
    """
    cand_gaps = gap_df[
        (gap_df["candidate_id"] == candidate_id) &
        (gap_df["justification_status"] == "Unjustified Gap")
    ]
    real_gaps = []
    for _, row in cand_gaps.iterrows():
        if not _is_routine_edu_transition(row):
            real_gaps.append({
                "start": str(row.get("gap_start", "")),
                "end":   str(row.get("gap_end", "")),
                "months": int(float(row.get("duration_months", 0) or 0)),
                "from":  str(row.get("prev_activity", "")),
                "to":    str(row.get("next_activity", "")),
            })
    return real_gaps


# --- Main extraction function ---

def extract_missing_candidate_info(
    candidate_id: str,
    output_dir: str = "data/analysis"
) -> List[str]:
    """
    Scans all analysis module outputs for a candidate and returns ONLY those
    items that are genuinely missing, flagged, or unjustified — never asks
    about things not present in the candidate's submitted CV.
    """
    out = Path(output_dir)
    missing_items: List[str] = []

    # ── 1. Supervision: only if CV mentions supervision but data is incomplete,
    #        OR if the supervisory_assessment explicitly says no data was found
    sup_p = out / "supervision_profiles.csv"
    if sup_p.exists():
        try:
            df  = pd.read_csv(sup_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                flag       = str(row.iloc[0].get("email_flag", "")).strip()
                missing_fl = str(row.iloc[0].get("data_missing", "")).strip().lower()
                assessment = str(row.iloc[0].get("supervisory_assessment", "")).lower()
                total_sup  = float(row.iloc[0].get("total_students_supervised") or 0)

                # Only flag supervision if the system found evidence the field is missing
                # AND the candidate actually has supervisory roles implied (not just blank CV)
                if flag == "REQUEST_SUPERVISION_DATA" and total_sup == 0:
                    if "no supervision information was found" in assessment:
                        # CV didn't mention it — don't demand it; just ask if applicable
                        missing_items.append(
                            "Postgraduate supervision history (if applicable): "
                            "your CV does not include details of MS or PhD students you have supervised. "
                            "If you have supervised postgraduate students, please provide their names, "
                            "degree level, thesis title, and year of graduation."
                        )
        except Exception:
            pass

    # ── 2. Career gaps: only real unjustified gaps, excluding routine
    #        education transitions common in Pakistan's academic system
    gap_p = out / "experience_gaps.csv"
    if gap_p.exists():
        try:
            gap_df     = pd.read_csv(gap_p, dtype=str).fillna("")
            real_gaps  = _extract_career_gaps(candidate_id, gap_df)
            if real_gaps:
                if len(real_gaps) == 1:
                    g = real_gaps[0]
                    from_label = g["from"].replace("[EDUCATION]", "education period").replace("[EXPERIENCE]", "position")
                    to_label   = g["to"].replace("[EDUCATION]", "education period").replace("[EXPERIENCE]", "position")
                    missing_items.append(
                        f"Career timeline clarification: a gap of {g['months']} months "
                        f"({g['start']} to {g['end']}) between your {from_label.strip()} and "
                        f"your {to_label.strip()} is not accounted for. "
                        f"Please briefly describe your activities during this period."
                    )
                else:
                    gap_descriptions = "; ".join(
                        [f"{g['months']} months ({g['start']}–{g['end']})" for g in real_gaps]
                    )
                    missing_items.append(
                        f"Career timeline clarification: {len(real_gaps)} periods in your professional "
                        f"history are unaccounted for ({gap_descriptions}). "
                        f"Please provide a brief explanation of your activities during each of these intervals."
                    )
        except Exception:
            pass

    # ── 3. Research: only if unverified venues or predatory-suspected publications exist
    res_p = out / "research_aggregates.csv"
    if res_p.exists():
        try:
            df  = pd.read_csv(res_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                unverified  = int(float(row.iloc[0].get("unverified_venue") or 0))
                predatory   = int(float(row.iloc[0].get("predatory_suspected") or 0))
                total_pubs  = int(float(row.iloc[0].get("total_journals") or 0)) + \
                              int(float(row.iloc[0].get("total_conferences") or 0))

                if predatory > 0 and total_pubs > 0:
                    missing_items.append(
                        f"Publication venue verification: {predatory} of your listed publication(s) "
                        f"appear in journals or venues flagged as potentially predatory or unverified. "
                        f"Please provide the official journal ISSN, indexing status (Scopus/WoS), "
                        f"and DOI for these publications."
                    )
                elif unverified > 0 and total_pubs > 0:
                    missing_items.append(
                        f"Indexing confirmation: {unverified} publication(s) listed in your CV could not "
                        f"be verified against Scopus or Web of Science records. "
                        f"Please provide the journal name, volume, issue, year, and DOI for each."
                    )
        except Exception:
            pass

    # ── 4. Books: only if the candidate listed books but none are verifiable
    bk_p = out / "book_aggregates.csv"
    if bk_p.exists():
        try:
            df  = pd.read_csv(bk_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                total_books = int(float(row.iloc[0].get("total_books") or 0))
                verifiable  = int(float(row.iloc[0].get("verifiable_books_count") or 0))
                if total_books > 0 and verifiable == 0:
                    missing_items.append(
                        f"Book publication details: your CV lists {total_books} book(s) but none could be "
                        f"verified. Please provide the publisher name, year of publication, and ISBN "
                        f"for each authored or co-authored book."
                    )
        except Exception:
            pass

    # ── 5. Patents: only if the candidate listed patents but none have registration numbers
    pat_p = out / "patent_aggregates.csv"
    if pat_p.exists():
        try:
            df  = pd.read_csv(pat_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                total_pat  = int(float(row.iloc[0].get("total_patents") or 0))
                verifiable = int(float(row.iloc[0].get("verifiable_patents") or 0))
                if total_pat > 0 and verifiable == 0:
                    missing_items.append(
                        f"Patent registration details: your CV references {total_pat} patent(s), "
                        f"but no application or grant numbers were found. "
                        f"Please provide the registration number, issuing authority, and grant date for each."
                    )
        except Exception:
            pass

    return missing_items


# --- Email drafting ---

_OPENING_VARIANTS = [
    "We hope this message finds you well.",
    "Thank you for applying for the faculty position.",
    "We appreciate your interest in joining our faculty.",
    "We have completed an initial review of your application.",
    "Your application has been received and reviewed by our evaluation system.",
]

_CLOSING_VARIANTS = [
    "We look forward to receiving your response.",
    "Please do not hesitate to reach out if you have any questions.",
    "We appreciate your cooperation and look forward to completing your evaluation.",
    "Your application remains under active consideration pending the above clarifications.",
    "Kindly ensure the requested materials are submitted at the earliest convenience.",
]


def draft_followup_email(
    candidate_id: str,
    candidate_name: str,
    missing_items: List[str],
    api_key: Optional[str] = None,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    base_url: Optional[str] = None,
    skip_llm: bool = False,
) -> Dict[str, str]:
    """
    Drafts a targeted, candidate-specific follow-up email. Only raises items
    that are data-verified as missing. Never follows a fixed template.
    Signs off as: TALASH Team.
    """
    if not missing_items:
        return {
            "candidate_id": candidate_id,
            "subject": f"Application Under Review – {candidate_name}",
            "body": (
                f"Dear {candidate_name},\n\n"
                f"Thank you for submitting your application. "
                f"The automated review of your profile is complete and all submitted records "
                f"appear to be in order at this stage.\n\n"
                f"We will be in touch regarding next steps in the evaluation process.\n\n"
                f"Regards,\nTALASH Team"
            ),
            "has_missing_info": False,
        }

    numbered = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(missing_items)])
    item_count = len(missing_items)
    api_k = api_key or os.environ.get("OPENAI_API_KEY", "")
    is_valid_key = bool(api_k and not str(api_k).startswith("your_") and len(str(api_k).strip()) > 20)

    if skip_llm or not is_valid_key:
        subject  = f"Information Required: Faculty Application – {candidate_name}"
        opening  = random.choice(_OPENING_VARIANTS)
        closing  = random.choice(_CLOSING_VARIANTS)
        item_word = "item requires" if item_count == 1 else "items require"

        body = (
            f"Dear {candidate_name},\n\n"
            f"{opening} "
            f"Following the automated evaluation of your submitted credentials, "
            f"{item_count} {item_word} your attention before your profile can be finalised:\n\n"
            f"{numbered}\n\n"
            f"Please submit the requested information within seven working days. "
            f"{closing}\n\n"
            f"Regards,\n"
            f"TALASH Team"
        )
        return {
            "candidate_id": candidate_id,
            "subject": subject,
            "body": body,
            "has_missing_info": True,
        }

    # LLM path
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        user_prompt = _EMAIL_USER_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            missing_items_numbered=numbered,
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EMAIL_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=650,
        )
        text = resp.choices[0].message.content or ""

        subject = f"Information Required: Faculty Application – {candidate_name}"
        if "Subject:" in text:
            parts   = text.split("Subject:", 1)[1].split("\n", 1)
            subject = parts[0].strip()
            text    = parts[1].strip() if len(parts) > 1 else text

        return {
            "candidate_id": candidate_id,
            "subject": subject,
            "body": text.strip(),
            "has_missing_info": True,
        }

    except Exception as e:
        logger.warning("LLM email drafting failed for '%s': %s", candidate_id, e)
        numbered_fb = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(missing_items)])
        return {
            "candidate_id": candidate_id,
            "subject": f"Information Required: Faculty Application – {candidate_name}",
            "body": (
                f"Dear {candidate_name},\n\n"
                f"Please provide clarification for the following:\n\n{numbered_fb}\n\n"
                f"Regards,\nTALASH Team"
            ),
            "has_missing_info": True,
        }
