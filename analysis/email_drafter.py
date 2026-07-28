"""
email_drafter.py  –  Missing-Info Detection & Personalized LLM Email Drafter
==========================================================================

WHY THIS FILE EXISTS
--------------------
Detects missing, incomplete, or unverified records per candidate across all 9 modules
and drafts a personalized, professional follow-up email requesting the specific missing items.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    Scans candidate profile data across modules and returns a list of missing information items.
    """
    out = Path(output_dir)
    missing_items = []

    # 1. Check Supervision Data
    sup_p = out / "supervision_profiles.csv"
    if sup_p.exists():
        try:
            df = pd.read_csv(sup_p, dtype=str)
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                flag_val = str(row.iloc[0].get("email_flag") or "").strip()
                data_missing = str(row.iloc[0].get("data_missing") or "").strip().lower()
                # Flag is "REQUEST_SUPERVISION_DATA" or data_missing is true
                if flag_val == "REQUEST_SUPERVISION_DATA" or data_missing == "true":
                    total_sup = float(row.iloc[0].get("total_students_supervised") or 0)
                    if total_sup == 0:
                        missing_items.append("Postgraduate student supervision record (list of MS/PhD thesis students supervised and graduation years).")
        except Exception:
            pass

    # 2. Check Education — unexplained or significant gaps
    edu_p = out / "educational_profiles.csv"
    if edu_p.exists():
        try:
            df = pd.read_csv(edu_p, dtype=str)
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty:
                unexplained = float(row.iloc[0].get("unexplained_gaps") or 0)
                significant = float(row.iloc[0].get("significant_gaps") or 0)
                drift = str(row.iloc[0].get("specialization_drift") or "").lower()
                if unexplained > 0:
                    missing_items.append(f"Explanation for {int(unexplained)} unexplained gap(s) in academic record (institutions, dates, or activities during those periods).")
                elif significant > 0 and drift not in ("", "none", "low", "nan"):
                    missing_items.append("Clarification of specialization change between degrees and the academic rationale behind it.")
        except Exception:
            pass

    # 3. Check Book ISBNs & Links
    bk_p = out / "book_aggregates.csv"
    if bk_p.exists():
        try:
            df = pd.read_csv(bk_p, dtype=str)
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty and float(row.iloc[0].get("total_books") or 0) > 0:
                v_cnt = float(row.iloc[0].get("verifiable_books_count") or 0)
                if v_cnt == 0:
                    missing_items.append("Publisher details and ISBN numbers for authored/co-authored book publications.")
        except Exception:
            pass

    # 4. Check Patent Numbers
    pat_p = out / "patent_aggregates.csv"
    if pat_p.exists():
        try:
            df = pd.read_csv(pat_p, dtype=str)
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty and float(row.iloc[0].get("total_patents") or 0) > 0:
                v_cnt = float(row.iloc[0].get("verifiable_patents") or 0)
                if v_cnt == 0:
                    missing_items.append("Patent application/grant registration numbers and issuing authority links.")
        except Exception:
            pass

    # 5. Check Unjustified Experience Gaps
    exp_p = out / "experience_profiles.csv"
    if exp_p.exists():
        try:
            df = pd.read_csv(exp_p, dtype=str)
            row = df[df["candidate_id"] == candidate_id]
            if not row.empty and float(row.iloc[0].get("unjustified_gaps") or 0) > 0:
                missing_items.append("Employment/activity justification for career timeline gap(s) exceeding 3 months.")
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
            "body": f"Dear {candidate_name},\n\nThank you for submitting your application. We have verified your submission and confirm that all required academic, publication, and professional records are complete.\n\nBest regards,\nFaculty Recruitment Committee",
            "has_missing_info": False,
        }

    bulleted = "\n".join([f"- {item}" for item in missing_items])
    api_k = api_key or os.environ.get("OPENAI_API_KEY", "")
    is_valid_key = bool(api_k and not str(api_k).startswith("your_") and len(str(api_k).strip()) > 20)

    if skip_llm or not is_valid_key:
        subject = f"Information Request: Faculty Application – {candidate_name}"
        body = (
            f"Dear {candidate_name},\n\n"
            f"Thank you for your interest in our faculty position. During our initial screening, "
            f"we identified a few items in your record that require clarification or additional documentation:\n\n"
            f"{bulleted}\n\n"
            f"Please submit the requested details at your earliest convenience to complete your evaluation profile.\n\n"
            f"Sincerely,\n"
            f"Faculty Selection Committee"
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
