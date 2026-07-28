"""
email_drafter.py  –  Missing-Info Detection & Personalized LLM Email Drafter
==========================================================================

WHY THIS FILE EXISTS
--------------------
Detects missing, incomplete, or unverified records per candidate across all 9 modules
and drafts a personalized, professional follow-up email requesting the specific missing items.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_EMAIL_SYSTEM_PROMPT = (
    "You are an academic recruitment officer for a university faculty selection committee. "
    "You will receive a candidate's name and a list of specific missing or unverified CV information items. "
    "Your task is to write a polite, professional, and personalized follow-up email requesting the candidate to provide the missing details. "
    "Do not use generic bulk template phrasing. Be specific, concise, and courteous."
)

_EMAIL_USER_TEMPLATE = (
    "Draft a personalized follow-up email to candidate {candidate_name} ({candidate_id}).\n\n"
    "MISSING OR INCOMPLETE DATA ITEMS:\n"
    "{missing_items_bulleted}\n\n"
    "Format with Subject line and formal Body text:"
)


def extract_missing_candidate_info(candidate_id: str, output_dir: str = "data/analysis") -> List[str]:
    """
    Scans candidate profile data across modules and returns a list of specific,
    candidate-tailored missing information items with concrete details where available.
    """
    out = Path(output_dir)
    missing_items = []

    # 1. Check Supervision Data
    sup_p = out / "supervision_profiles.csv"
    if sup_p.exists():
        try:
            df = pd.read_csv(sup_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                flag_val = str(row.iloc[0].get("email_flag") or "").strip()
                data_missing = str(row.iloc[0].get("data_missing") or "").strip().lower()
                total_sup = float(row.iloc[0].get("total_students_supervised") or 0)
                ms_sup = float(row.iloc[0].get("total_ms_supervised") or 0)
                phd_sup = float(row.iloc[0].get("total_phd_supervised") or 0)
                if (flag_val == "REQUEST_SUPERVISION_DATA" or data_missing == "true") and total_sup == 0:
                    missing_items.append(
                        "Postgraduate student supervision record: please provide the full list of MS and PhD "
                        "thesis students you have supervised, including thesis titles, student names, and graduation years."
                    )
                elif ms_sup == 0 and phd_sup == 0 and total_sup == 0:
                    missing_items.append(
                        "Confirmation of postgraduate supervision activity: your CV does not list any MS/PhD students "
                        "supervised. If applicable, please provide supervision details."
                    )
        except Exception:
            pass

    # 2. Check Education — unexplained or significant gaps and drift
    edu_p = out / "educational_profiles.csv"
    if edu_p.exists():
        try:
            df = pd.read_csv(edu_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                unexplained = float(row.iloc[0].get("unexplained_gaps") or 0)
                significant = float(row.iloc[0].get("significant_gaps") or 0)
                total_gaps = float(row.iloc[0].get("total_gaps") or 0)
                drift = str(row.iloc[0].get("specialization_drift") or "").strip().lower()
                drift_details = str(row.iloc[0].get("drift_details") or "").strip()
                if unexplained > 0:
                    missing_items.append(
                        f"Academic timeline clarification: {int(unexplained)} period(s) in your academic history "
                        f"are unaccounted for. Please provide the institution name(s), dates, and activities "
                        f"(coursework, research, or other engagements) during these intervals."
                    )
                elif significant > 0 and drift not in ("", "none", "low", "nan", "false"):
                    drift_note = f" ({drift_details})" if drift_details and drift_details.lower() not in ("nan", "") else ""
                    missing_items.append(
                        f"Clarification of specialization change between degrees{drift_note}: "
                        f"your qualifications show a shift in academic focus. Please explain the academic "
                        f"rationale and how it relates to the applied faculty position."
                    )
        except Exception:
            pass

    # 3. Check Book ISBNs & Links
    bk_p = out / "book_aggregates.csv"
    if bk_p.exists():
        try:
            df = pd.read_csv(bk_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                total_books = float(row.iloc[0].get("total_books") or 0)
                v_cnt = float(row.iloc[0].get("verifiable_books_count") or 0)
                if total_books > 0 and v_cnt == 0:
                    missing_items.append(
                        f"Verifiable publication details for your {int(total_books)} book(s): "
                        f"please provide publisher name, year of publication, and ISBN number for each authored or co-authored book."
                    )
        except Exception:
            pass

    # 4. Check Patent Numbers
    pat_p = out / "patent_aggregates.csv"
    if pat_p.exists():
        try:
            df = pd.read_csv(pat_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                total_patents = float(row.iloc[0].get("total_patents") or 0)
                v_cnt = float(row.iloc[0].get("verifiable_patents") or 0)
                if total_patents > 0 and v_cnt == 0:
                    missing_items.append(
                        f"Patent registration details for your {int(total_patents)} listed patent(s): "
                        f"please provide application/grant registration numbers, the issuing authority, and official links."
                    )
        except Exception:
            pass

    # 5. Check Unjustified Experience Gaps — include the count for specificity
    exp_p = out / "experience_profiles.csv"
    if exp_p.exists():
        try:
            df = pd.read_csv(exp_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                unjustified = float(row.iloc[0].get("unjustified_gaps") or 0)
                gap_count = float(row.iloc[0].get("gap_count") or 0)
                if unjustified > 0:
                    missing_items.append(
                        f"Employment timeline justification: your professional history contains "
                        f"{int(unjustified)} unjustified gap period(s) exceeding three months. "
                        f"Please provide a brief explanation of your activities (e.g., freelance work, further study, "
                        f"career break) for each gap period."
                    )
        except Exception:
            pass

    # 6. Check Research — unverified publications (no DOI)
    res_p = out / "research_aggregates.csv"
    if res_p.exists():
        try:
            df = pd.read_csv(res_p, dtype=str).fillna("")
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                total_pubs = float(row.iloc[0].get("total_publications") or 0)
                with_doi = float(row.iloc[0].get("publications_with_doi") or 0)
                if total_pubs > 0 and with_doi < total_pubs * 0.5:
                    missing_doi = int(total_pubs - with_doi)
                    missing_items.append(
                        f"DOI or indexing links for {missing_doi} of your listed publication(s): "
                        f"more than half of your publications do not have verifiable DOIs or journal links. "
                        f"Please provide the journal name, volume, issue, and DOI for each unverified publication."
                    )
        except Exception:
            pass

    return missing_items


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
    Drafts a personalized follow-up email to a candidate requesting missing information.
    """
    if not missing_items:
        return {
            "candidate_id": candidate_id,
            "subject": f"Application Record Complete – {candidate_name}",
            "body": (
                f"Dear {candidate_name},\n\n"
                f"Thank you for submitting your application. We have completed the initial automated "
                f"review of your credentials via the TALASH evaluation system and confirm that all "
                f"required academic, publication, and professional records appear complete at this stage.\n\n"
                f"We will be in touch with further steps in the selection process.\n\n"
                f"Best regards,\nFaculty Recruitment Committee\nHigher Education Commission of Pakistan"
            ),
            "has_missing_info": False,
        }

    bulleted = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(missing_items)])
    item_count = len(missing_items)
    api_k = api_key or os.environ.get("OPENAI_API_KEY", "")
    is_valid_key = bool(api_k and not str(api_k).startswith("your_") and len(str(api_k).strip()) > 20)

    if skip_llm or not is_valid_key:
        subject = f"Information Request: Faculty Application – {candidate_name}"
        item_word = "item" if item_count == 1 else "items"
        body = (
            f"Dear {candidate_name},\n\n"
            f"Thank you for your interest in the faculty position at our institution. "
            f"The TALASH automated evaluation system has completed an initial screening of your submitted credentials.\n\n"
            f"During our review, we identified {item_count} {item_word} in your profile that require "
            f"clarification or additional supporting documentation:\n\n"
            f"{bulleted}\n\n"
            f"Please submit the requested materials within seven working days. "
            f"Incomplete submissions may delay or affect the outcome of your evaluation.\n\n"
            f"Sincerely,\n"
            f"Faculty Selection Committee\n"
            f"Higher Education Commission of Pakistan\n"
            f"hiring@hec.gov.pk"
        )
        return {
            "candidate_id": candidate_id,
            "subject": subject,
            "body": body,
            "has_missing_info": True,
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

        user_prompt = _EMAIL_USER_TEMPLATE.format(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            missing_items_bulleted=bulleted,
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _EMAIL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        text = resp.choices[0].message.content or ""

        # Extract Subject line if present
        subject = f"Information Request: Faculty Application – {candidate_name}"
        if "Subject:" in text:
            parts = text.split("Subject:", 1)[1].split("\n", 1)
            subject = parts[0].strip()
            text = parts[1].strip() if len(parts) > 1 else text

        return {
            "candidate_id": candidate_id,
            "subject": subject,
            "body": text.strip(),
            "has_missing_info": True,
        }

    except Exception as e:
        logger.warning("LLM email drafting failed for '%s': %s", candidate_id, e)
        return {
            "candidate_id": candidate_id,
            "subject": f"Information Request: Faculty Application – {candidate_name}",
            "body": f"Dear {candidate_name},\n\nPlease provide clarification for the following items:\n{bulleted}\n\nRegards,\nFaculty Recruitment Committee",
            "has_missing_info": True,
        }
