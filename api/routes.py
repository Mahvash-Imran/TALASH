"""
routes.py  –  FastAPI Endpoint Controllers
==========================================
"""

import os
import shutil
import logging
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["TALASH Candidates & Evaluation API"])

DATA_DIR = Path("data/analysis")
EXTRACTED_DIR = Path("data/extracted")
UPLOADS_DIR = Path("data/uploads")
CV_DIR = Path("data/cvs")
BULK_SPLIT_DIR = Path("data/cvs/split")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CV_DIR.mkdir(parents=True, exist_ok=True)
BULK_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Marker text that appears at the top of each candidate page in a bulk PDF
_BULK_MARKER = "Candidate for the Post"

# In-memory status tracker
# For single: candidate_id -> {"status", "message"}
# For bulk:   upload_id   -> {"status", "message", "total", "completed", "candidates", "errors"}
_processing_status: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm_config():
    """Load LLM config from environment."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    model    = os.environ.get("OPENAI_MODEL", "groq/compound-mini")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    is_valid = bool(api_key and not str(api_key).startswith("your_") and len(str(api_key).strip()) > 20)
    return api_key, model, base_url, is_valid


def _detect_bulk_pdf(pdf_path: str) -> int:
    """
    Detect if a PDF is a bulk (multi-candidate) file.
    Returns the number of candidates found (0 = not bulk / detection failed).
    """
    try:
        import pdfplumber
        count = 0
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if _BULK_MARKER in text:
                    count += 1
        return count
    except Exception as e:
        logger.warning("[Upload] Bulk detection failed: %s", e)
        return 0


def _extract_one_cv(pdf_path: Path, api_key: str, model: str, base_url: str) -> bool:
    """
    Run Module 1 (PDF read + LLM extraction + export) for a single CV file.
    Returns True on success, False on failure.
    """
    from preprocessing import PDFReader, LLMExtractor, Normalizer, Exporter
    import pdfplumber

    # Read this specific PDF directly (not via folder scan)
    try:
        pages_text = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text()
                    if t:
                        pages_text.append(t)
                except Exception:
                    pass
        cv_text = "\n\n".join(pages_text)
    except Exception as e:
        logger.error("[Extract] PDF read failed for %s: %s", pdf_path.name, e)
        return False

    if not cv_text.strip():
        logger.error("[Extract] Empty text from %s", pdf_path.name)
        return False

    extractor  = LLMExtractor(api_key=api_key, model=model, base_url=base_url or None)
    normalizer = Normalizer()
    exporter   = Exporter(output_dir=str(EXTRACTED_DIR))

    extraction = extractor.extract(
        candidate_filename=pdf_path.stem,
        cv_text=cv_text,
    )

    if not extraction.success:
        logger.error("[Extract] LLM failed for %s: %s", pdf_path.stem, extraction.error_message)
        return False

    clean_data = normalizer.normalize(extraction.data)
    exporter.add_candidate(
        candidate_filename=pdf_path.stem,
        normalized_data=clean_data,
        validation=extraction.validation,
        pdf_result=None,
    )
    exporter.export()
    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", version="1.0.0", service="TALASH Backend API")


@router.post("/upload", response_model=UploadResponse)
async def upload_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a CV PDF (single candidate) or a bulk PDF (multiple candidates).
    Auto-detects bulk PDFs by looking for the 'Candidate for the Post' marker.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    dest_path = UPLOADS_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    upload_id = file.filename.replace(".pdf", "").replace(" ", "_")

    # Detect bulk vs single
    candidate_count = _detect_bulk_pdf(str(dest_path))
    is_bulk = candidate_count > 1

    if is_bulk:
        _processing_status[upload_id] = {
            "status": "processing",
            "message": f"Bulk PDF detected — splitting {candidate_count} CVs...",
            "total": candidate_count,
            "completed": 0,
            "candidates": [],
            "errors": [],
            "is_bulk": True,
        }
        background_tasks.add_task(_process_bulk_cv_background, str(dest_path), upload_id, candidate_count)
        return UploadResponse(
            filename=file.filename,
            candidate_id=upload_id,
            status="processing",
            message=f"Bulk PDF detected with {candidate_count} candidates. Splitting and processing in background."
        )
    else:
        _processing_status[upload_id] = {
            "status": "processing",
            "message": "Queued for extraction and analysis.",
            "is_bulk": False,
        }
        background_tasks.add_task(_process_uploaded_cv_background, str(dest_path), upload_id)
        return UploadResponse(
            filename=file.filename,
            candidate_id=upload_id,
            status="processing",
            message="Resume uploaded. Full pipeline (extraction + analysis) started in background."
        )


@router.get("/status/{candidate_id}")
def get_processing_status(candidate_id: str):
    """Returns the current processing status of an uploaded CV or bulk upload."""
    if candidate_id in _processing_status:
        return {"candidate_id": candidate_id, **_processing_status[candidate_id]}
    # Pre-existing candidate — check composite evaluations
    comp_file = DATA_DIR / "composite_evaluations.csv"
    if comp_file.exists():
        df = pd.read_csv(comp_file, dtype=str)
        if candidate_id in df.get("candidate_id", pd.Series()).values:
            return {"candidate_id": candidate_id, "status": "done", "message": "Evaluation complete.", "is_bulk": False}
    return {"candidate_id": candidate_id, "status": "unknown", "message": "Candidate not found.", "is_bulk": False}


# ---------------------------------------------------------------------------
# Background processing: single CV
# ---------------------------------------------------------------------------

def _process_uploaded_cv_background(pdf_path: str, candidate_id: str):
    """
    Full pipeline for a newly uploaded single CV:
      Step 1 – Copy PDF into data/cvs/
      Step 2 – Run Module 1 (LLM extraction) with retry/chunking
      Step 3 – Run Modules 2-10 to compute composite scores
    """
    try:
        _processing_status[candidate_id] = {"status": "processing", "message": "Step 1/3: Copying PDF to CV folder.", "is_bulk": False}

        src = Path(pdf_path)
        cv_dest = CV_DIR / src.name
        shutil.copy2(str(src), str(cv_dest))
        logger.info("[Upload] Copied %s -> %s", src.name, cv_dest)

        _processing_status[candidate_id] = {"status": "processing", "message": "Step 2/3: Extracting CV data (LLM + normalization).", "is_bulk": False}
        logger.info("[Upload] Running Module 1 extraction for: %s", candidate_id)

        api_key, model, base_url, is_valid = _get_llm_config()

        success = _extract_one_cv(cv_dest, api_key, model, base_url)
        if not success:
            raise ValueError("LLM extraction failed after all retries. Check server logs for details.")

        logger.info("[Upload] Module 1 complete for %s", candidate_id)

        _processing_status[candidate_id] = {"status": "processing", "message": "Step 3/3: Running analysis pipeline (Modules 2-10).", "is_bulk": False}
        logger.info("[Upload] Running Modules 2-10 for %s", candidate_id)

        pipeline = MasterPipeline(
            api_key=api_key, model=model,
            base_url=base_url if base_url else None,
            skip_llm=True
        )
        pipeline.run_full_pipeline()

        _processing_status[candidate_id] = {"status": "done", "message": "Processing complete. Candidate added to evaluations.", "is_bulk": False}
        logger.info("[Upload] Full pipeline complete for %s", candidate_id)

    except Exception as e:
        error_msg = str(e)
        _processing_status[candidate_id] = {"status": "error", "message": f"Processing failed: {error_msg}", "is_bulk": False}
        logger.error("[Upload] Error for %s: %s", candidate_id, error_msg, exc_info=True)


# ---------------------------------------------------------------------------
# Background processing: bulk PDF
# ---------------------------------------------------------------------------

def _process_bulk_cv_background(pdf_path: str, upload_id: str, expected_count: int):
    """
    Full pipeline for a bulk (multi-candidate) PDF:
      Step 1 – Split PDF into individual candidate PDFs
      Step 2 – For each candidate: run Module 1 extraction
      Step 3 – Run Modules 2-10 once for all candidates
    """
    try:
        _processing_status[upload_id].update({"status": "processing", "message": f"Step 1/3: Splitting bulk PDF into individual CVs..."})
        logger.info("[Bulk] Starting bulk split for %s", upload_id)

        # ── Step 1: Split PDF ────────────────────────────────────────────
        from split_dataset import split_pdf
        written = split_pdf(
            input_path=pdf_path,
            output_dir=str(BULK_SPLIT_DIR),
        )

        # Collect the split PDF files
        split_pdfs = sorted(BULK_SPLIT_DIR.glob("*.pdf"))
        if not split_pdfs:
            raise ValueError("PDF splitting produced no output files.")

        _processing_status[upload_id].update({
            "message": f"Split complete: {len(split_pdfs)} individual CVs found. Extracting...",
            "total": len(split_pdfs),
        })
        logger.info("[Bulk] Split produced %d PDFs", len(split_pdfs))

        # ── Step 2: Extract each candidate ──────────────────────────────
        api_key, model, base_url, is_valid = _get_llm_config()
        completed = 0
        candidates_done = []
        errors = []

        for i, pdf_file in enumerate(split_pdfs, 1):
            cid = pdf_file.stem
            _processing_status[upload_id].update({
                "message": f"Step 2/3: Extracting CV {i}/{len(split_pdfs)}: {cid}",
                "completed": completed,
            })
            logger.info("[Bulk] Extracting %d/%d: %s", i, len(split_pdfs), cid)

            # Copy to main cvs folder
            cv_dest = CV_DIR / pdf_file.name
            shutil.copy2(str(pdf_file), str(cv_dest))

            success = _extract_one_cv(cv_dest, api_key, model, base_url)
            if success:
                completed += 1
                candidates_done.append(cid)
                logger.info("[Bulk] Extracted OK: %s (%d/%d)", cid, completed, len(split_pdfs))
            else:
                errors.append(cid)
                logger.warning("[Bulk] Extraction failed: %s", cid)

            _processing_status[upload_id].update({
                "completed": completed,
                "candidates": candidates_done,
                "errors": errors,
            })

        # ── Step 3: Run analysis pipeline ───────────────────────────────
        _processing_status[upload_id].update({
            "message": f"Step 3/3: Running analysis pipeline for all {completed} candidates...",
        })
        logger.info("[Bulk] Running Modules 2-10 for %d candidates", completed)

        pipeline = MasterPipeline(
            api_key=api_key, model=model,
            base_url=base_url if base_url else None,
            skip_llm=True
        )
        pipeline.run_full_pipeline()

        final_msg = f"Done! {completed}/{len(split_pdfs)} candidates added to evaluations."
        if errors:
            final_msg += f" ({len(errors)} failed: {', '.join(errors[:3])}{'...' if len(errors)>3 else ''})"

        _processing_status[upload_id].update({
            "status": "done",
            "message": final_msg,
            "completed": completed,
            "candidates": candidates_done,
            "errors": errors,
        })
        logger.info("[Bulk] Complete: %s", final_msg)

    except Exception as e:
        error_msg = str(e)
        _processing_status[upload_id].update({"status": "error", "message": f"Bulk processing failed: {error_msg}"})
        logger.error("[Bulk] Error for %s: %s", upload_id, error_msg, exc_info=True)


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
        from .main import ensure_seed_data
        ensure_seed_data()

    if not comp_file.exists():
        return []

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

    df = pd.read_csv(comp_file, dtype=str).fillna("")
    res = []
    def _safe_float(v):
        try:
            val = float(v)
            return 0.0 if (val != val) else val
        except (ValueError, TypeError):
            return 0.0

    for _, r in df.iterrows():
        cid = str(r.get("candidate_id", "")).strip()
        if not cid:
            continue
        cname = str(r.get("candidate_name", "")).strip()
        if not cname or cname.lower() in ("nan", "none", "null"):
            cname = cid.replace("_", " ").title()

        uploaded_at = cv_mtimes.get(cid) or comp_file.stat().st_mtime
        if not isinstance(uploaded_at, str):
            uploaded_at = datetime.datetime.fromtimestamp(uploaded_at).strftime("%Y-%m-%dT%H:%M:%S")

        res.append(CandidateSummary(
            candidate_id=cid,
            candidate_name=cname,
            overall_composite_score=_safe_float(r.get("overall_composite_score")),
            candidate_tier=str(r.get("candidate_tier") or "Unclassified"),
            education_score=_safe_float(r.get("education_score")),
            research_score=_safe_float(r.get("research_score")),
            supervision_score=_safe_float(r.get("supervision_score")),
            innovation_score=_safe_float(r.get("innovation_score")),
            breadth_score=_safe_float(r.get("breadth_score")),
            collaboration_score=_safe_float(r.get("collaboration_score")),
            experience_score=_safe_float(r.get("experience_score")),
            total_experience_years=_safe_float(r.get("total_experience_years")),
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
