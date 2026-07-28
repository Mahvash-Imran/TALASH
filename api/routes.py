"""
routes.py  –  FastAPI Endpoint Controllers
==========================================
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, Query

from .schemas import (
    HealthResponse,
    UploadResponse,
    CandidateSummary,
    CompareRequest,
    CompareResponse,
    EmailDraftResponse,
)
from pipeline_orchestrator import MasterPipeline
from analysis.email_drafter import extract_missing_candidate_info, draft_followup_email
from analysis.composite_evaluator import compute_candidate_composite_score

router = APIRouter(prefix="/api/v1", tags=["TALASH Candidates & Evaluation API"])

DATA_DIR = Path("data/analysis")
EXTRACTED_DIR = Path("data/extracted")
UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", version="1.0.0", service="TALASH Backend API")


@router.post("/upload", response_model=UploadResponse)
async def upload_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a candidate PDF resume and triggers background processing.
    Uses BackgroundTasks for non-blocking execution.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    dest_path = UPLOADS_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cid = file.filename.replace(".pdf", "").replace(" ", "_")

    # Schedule background processing
    background_tasks.add_task(_process_uploaded_cv_background, str(dest_path), cid)

    return UploadResponse(
        filename=file.filename,
        candidate_id=cid,
        status="queued",
        message="Resume uploaded successfully. Background processing started."
    )


def _process_uploaded_cv_background(pdf_path: str, candidate_id: str):
    """Background task function for non-blocking analysis."""
    try:
        pipeline = MasterPipeline(skip_llm=True)
        pipeline.run_full_pipeline()
    except Exception as e:
        print(f"Background processing error for {candidate_id}: {e}")


@router.post("/analyze/{candidate_id}")
def analyze_candidate(candidate_id: str, background_tasks: BackgroundTasks):
    """
    Triggers full pipeline execution (Parts 1-10) for a candidate as a background task.
    """
    background_tasks.add_task(_process_uploaded_cv_background, "", candidate_id)
    return {
        "candidate_id": candidate_id,
        "status": "queued",
        "message": f"Full pipeline analysis queued for candidate {candidate_id}."
    }


@router.get("/candidates", response_model=List[CandidateSummary])
def list_candidates():
    """
    Returns a summary list of all evaluated candidates with composite scores, tiers,
    processing status, missing-info count, and upload timestamp.
    """
    import datetime

    comp_file = DATA_DIR / "composite_evaluations.csv"
    if not comp_file.exists():
        pipeline = MasterPipeline(skip_llm=True)
        pipeline.run_full_pipeline()

    if not comp_file.exists():
        raise HTTPException(status_code=404, detail="No candidate evaluations found.")

    # Load missing info counts from email drafter analysis (if available)
    missing_counts: dict = {}
    for module_csv in ["edu_gaps.csv", "supervision_profiles.csv"]:
        mf = DATA_DIR / module_csv
        if mf.exists():
            try:
                mdf = pd.read_csv(mf, dtype=str).fillna("")
                for _, mr in mdf.iterrows():
                    cid = str(mr.get("candidate_id", "")).strip()
                    if cid:
                        # Count columns that look empty / flagged
                        empties = sum(1 for v in mr.values if str(v).strip() in ("", "nan", "N/A", "MISSING", "unknown"))
                        missing_counts[cid] = missing_counts.get(cid, 0) + max(0, empties - 3)
            except Exception:
                pass

    # CV upload timestamps
    cv_dir = Path("data/cvs")
    cv_mtimes: dict = {}
    if cv_dir.exists():
        for pdf in cv_dir.glob("*.pdf"):
            stem = pdf.stem  # e.g. 04_MUHAMMAD_FARRUKH
            mtime = pdf.stat().st_mtime
            cv_mtimes[stem] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%S")

    df = pd.read_csv(comp_file, dtype=str)
    res = []
    for _, r in df.iterrows():
        cid = r.get("candidate_id", "")
        uploaded_at = cv_mtimes.get(cid) or comp_file.stat().st_mtime
        if not isinstance(uploaded_at, str):
            uploaded_at = datetime.datetime.fromtimestamp(uploaded_at).strftime("%Y-%m-%dT%H:%M:%S")
        res.append(CandidateSummary(
            candidate_id=cid,
            candidate_name=r.get("candidate_name", cid),
            overall_composite_score=float(r.get("overall_composite_score") or 0.0),
            candidate_tier=r.get("candidate_tier", "Unclassified"),
            education_score=float(r.get("education_score") or 0.0),
            research_score=float(r.get("research_score") or 0.0),
            supervision_score=float(r.get("supervision_score") or 0.0),
            innovation_score=float(r.get("innovation_score") or 0.0),
            breadth_score=float(r.get("breadth_score") or 0.0),
            collaboration_score=float(r.get("collaboration_score") or 0.0),
            experience_score=float(r.get("experience_score") or 0.0),
            total_experience_years=float(r.get("experience_score") or 0.0),
            status="done",
            missing_info_count=min(missing_counts.get(cid, 0), 9),
            uploaded_at=uploaded_at,
        ))
    return res


@router.get("/candidates/{candidate_id}")
def get_candidate_details(candidate_id: str):
    """
    Returns detailed multi-module evaluation breakdown for a specific candidate.
    """
    comp_file = DATA_DIR / "composite_evaluations.csv"
    if not comp_file.exists():
        raise HTTPException(status_code=404, detail="Evaluations dataset not found.")

    df = pd.read_csv(comp_file, dtype=str).fillna("")
    row = df[df["candidate_id"] == candidate_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found.")

    cand_data = row.iloc[0].to_dict()

    # Load module specific records if available
    for module_file in ["educational_profiles.csv", "research_aggregates.csv", "supervision_profiles.csv", "book_aggregates.csv", "patent_aggregates.csv", "research_breadth_profiles.csv", "collaboration_profiles.csv", "experience_profiles.csv"]:
        mf = DATA_DIR / module_file
        if mf.exists():
            try:
                mdf = pd.read_csv(mf, dtype=str).fillna("")
                mrow = mdf[mdf["candidate_id"] == candidate_id]
                if not mrow.empty:
                    key_name = module_file.replace(".csv", "")
                    cand_data[key_name] = mrow.iloc[0].to_dict()
            except Exception:
                pass

    return cand_data


@router.post("/compare", response_model=CompareResponse)
def compare_candidates(req: CompareRequest):
    """
    Compares two or more candidates side-by-side across all 10 modules.
    """
    comp_file = DATA_DIR / "composite_evaluations.csv"
    if not comp_file.exists():
        raise HTTPException(status_code=404, detail="Evaluations dataset not found.")

    df = pd.read_csv(comp_file, dtype=str)
    matched_rows = df[df["candidate_id"].isin(req.candidate_ids)]

    if matched_rows.empty:
        raise HTTPException(status_code=404, detail="None of the specified candidate IDs were found.")

    candidates_detail = matched_rows.to_dict(orient="records")

    # Ranking
    sorted_cand = sorted(candidates_detail, key=lambda x: float(x.get("overall_composite_score") or 0.0), reverse=True)
    ranking = [
        {"rank": i + 1, "candidate_id": c["candidate_id"], "overall_score": float(c.get("overall_composite_score") or 0.0), "tier": c.get("candidate_tier")}
        for i, c in enumerate(sorted_cand)
    ]

    return CompareResponse(
        comparison_count=len(candidates_detail),
        candidates=candidates_detail,
        ranking=ranking,
    )


@router.get("/email/{candidate_id}", response_model=EmailDraftResponse)
def get_candidate_email_draft(candidate_id: str):
    """
    Extracts missing candidate information and drafts a personalized follow-up email.
    """
    cand_df = _load_csv(EXTRACTED_DIR / "candidates.csv")
    cand_name = candidate_id
    if cand_df is not None and not cand_df.empty:
        r = cand_df[cand_df["candidate_id"] == candidate_id]
        if not r.empty:
            cand_name = r.iloc[0].get("name") or candidate_id

    missing_items = extract_missing_candidate_info(candidate_id, str(DATA_DIR))
    draft = draft_followup_email(candidate_id, cand_name, missing_items, skip_llm=True)

    return EmailDraftResponse(
        candidate_id=candidate_id,
        subject=draft["subject"],
        body=draft["body"],
        has_missing_info=draft["has_missing_info"],
    )


@router.get("/reports/{candidate_id}")
def get_candidate_report(candidate_id: str):
    """
    Returns full text report summary for a candidate.
    """
    rep_file = DATA_DIR / "experience_report.txt"
    if not rep_file.exists():
        raise HTTPException(status_code=404, detail="Reports not found.")

    content = rep_file.read_text(encoding="utf-8")
    return {"candidate_id": candidate_id, "report": content}


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            pass
    return None
