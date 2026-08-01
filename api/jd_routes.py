"""
jd_routes.py  –  FastAPI Controller for Module 11 Job Description Matching
===========================================================================
"""

import json
import os
import re
import shutil
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import pandas as pd

from analysis.jd_parser import parse_job_description
from analysis.jd_matcher import compute_match_score
from run_jd_matching import evaluate_jd_against_candidates, generate_jd_id, load_candidate_profiles

router = APIRouter(prefix="/api/v1/jd", tags=["Job Description Matching API"])

DATA_DIR = Path("data/analysis")
JD_MATCHES_DIR = DATA_DIR / "jd_matches"
JD_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_experience_from_cv(cv_text: str) -> float:
    """
    Estimates total experience years from date ranges found in CV text.
    Section-aware: only counts years from the PROFESSIONAL EXPERIENCE section
    to avoid double-counting education date ranges.
    Falls back to explicit 'X years of experience' mentions.
    """
    month_pat = (
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
        r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
        r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    )
    year_range_re = re.compile(
        rf'{month_pat}?\s*(\d{{4}})\s*[-\u2013\u2014to\s]+{month_pat}?\s*(\d{{4}}|[Pp]resent|[Cc]urrent|[Nn]ow)',
    )

    # Try explicit mention first (e.g. "12+ years of experience")
    m = re.search(r'(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b.{0,40}?(?:experience|exp)', cv_text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 40:
            return float(val)

    # Section-aware extraction: find experience section lines
    lines = cv_text.splitlines()
    in_exp = False
    exp_lines = []
    for line in lines:
        low = line.lower().strip()
        if re.search(
            r'\b(professional experience|work experience|employment|career history|positions held|teaching experience)\b',
            low
        ):
            in_exp = True
        elif re.search(r'\b(education|academic background|qualifications|publications|skills|awards|references)\b', low):
            # Stop collecting when a new major section starts
            if in_exp:
                break
        if in_exp:
            exp_lines.append(line)

    search_text = '\n'.join(exp_lines) if exp_lines else cv_text

    year_pairs = year_range_re.findall(search_text)
    if not year_pairs:
        return 0.0

    current_year = datetime.datetime.now().year
    total = 0.0
    for start, end in year_pairs:
        s = int(start)
        e = current_year if end.lower() in ('present', 'current', 'now') else int(end)
        diff = max(0, e - s)
        if diff <= 45:  # Sanity: ignore crazy-long ranges
            total += diff

    return round(min(total, 50.0), 1)


def _extract_degree_from_cv(cv_text: str) -> str:
    """Extracts the highest degree from CV text."""
    text_lower = cv_text.lower()
    if re.search(r'\bph\.?d\b|\bdoctorate\b|\bdoctor of philosophy\b', text_lower):
        return "PhD"
    if re.search(r'\bm\.?s\b|\bmaster\b|\bmphil\b|\bm\.?phil\b|\bm\.?e\b|\bm\.?eng\b', text_lower):
        return "MS"
    return "BS"


def _extract_specializations_from_cv(cv_text: str) -> List[str]:
    """Extracts degree specializations/fields from CV education section."""
    specs = []
    # Look for education-section lines with common field names
    edu_lines = re.findall(
        r'(?:specializ(?:ation|ed|ing)|major(?:ing)?|field of study|discipline|in)\s*[:–-]?\s*([A-Za-z ,&/]+)',
        cv_text, re.IGNORECASE
    )
    for line in edu_lines:
        chunk = line.strip()[:80]
        if chunk and len(chunk) > 3:
            specs.append(chunk)
    return specs[:5]


def _extract_skills_from_cv(cv_text: str, jd_required_skills: List[str]) -> List[str]:
    """Finds which JD-required skills appear in the CV, plus common technical keywords."""
    found = set()
    # Check JD required skills first
    for sk in jd_required_skills:
        if sk.lower() in cv_text.lower():
            found.add(sk)

    # Check broad technical vocabulary
    tech_keywords = [
        "Python", "Machine Learning", "Deep Learning", "Computer Vision", "PyTorch",
        "TensorFlow", "OpenCV", "C++", "Java", "SQL", "Linux", "MATLAB", "R",
        "Cybersecurity", "IoT", "Signal Processing", "Image Processing", "NLP",
        "Natural Language Processing", "Reinforcement Learning", "Docker", "Kubernetes",
        "Microelectronics", "VLSI", "Verilog", "FPGA", "Embedded Systems",
        "AutoCAD", "PLC", "Wireless Communication", "Data Science", "Cloud Computing",
        "Artificial Intelligence", "Object Detection", "Robotics", "5G", "LTE",
    ]
    for kw in tech_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', cv_text, re.IGNORECASE):
            found.add(kw)

    return list(found)


@router.post("/upload")
async def upload_job_description(
    jd_text: Optional[str] = Form(None),
    jd_file: Optional[UploadFile] = File(None),
):
    """
    Accepts raw JD text or uploaded file (PDF/DOCX/TXT). Parses requirements and saves JD record.
    """
    raw_text = ""
    if jd_file and jd_file.filename:
        content_bytes = await jd_file.read()
        filename = jd_file.filename.lower()
        if filename.endswith(".pdf"):
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(content_bytes))
                raw_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception as e:
                # Fallback simple text decode
                raw_text = content_bytes.decode("utf-8", errors="ignore")
        else:
            raw_text = content_bytes.decode("utf-8", errors="ignore")
    elif jd_text and jd_text.strip():
        raw_text = jd_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Please provide either jd_text or a jd_file.")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Job description text is empty.")

    # Parse JD
    parsed_jd = parse_job_description(raw_text, skip_llm=True)
    jd_id = generate_jd_id(parsed_jd.get("title", "Position"))

    jd_dir = JD_MATCHES_DIR / jd_id
    jd_dir.mkdir(parents=True, exist_ok=True)

    # Save to disk for persistence
    (jd_dir / "jd_original.txt").write_text(raw_text, encoding="utf-8")
    (jd_dir / "jd_parsed.json").write_text(json.dumps(parsed_jd, indent=2), encoding="utf-8")

    meta = {
        "jd_id": jd_id,
        "title": parsed_jd.get("title", "Position"),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "required_degree": parsed_jd.get("required_degree_level", "BS"),
        "min_experience_years": parsed_jd.get("min_experience_years", 0),
        "skill_count": len(parsed_jd.get("required_skills", [])),
    }
    (jd_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "jd_id": jd_id,
        "title": meta["title"],
        "meta": meta,
        "parsed_jd": parsed_jd,
        "raw_text": raw_text,
    }


@router.get("/list")
def list_job_descriptions():
    """
    Returns list of all historically uploaded and saved JDs.
    """
    jds = []
    if JD_MATCHES_DIR.exists():
        for d in JD_MATCHES_DIR.iterdir():
            if d.is_dir():
                meta_file = d / "meta.json"
                parsed_file = d / "jd_parsed.json"
                results_file = d / "results.csv"

                if meta_file.exists():
                    try:
                        m = json.loads(meta_file.read_text(encoding="utf-8"))
                        # Count results
                        cand_count = 0
                        if results_file.exists():
                            try:
                                df = pd.read_csv(results_file)
                                cand_count = len(df)
                            except Exception:
                                pass
                        m["candidate_count"] = cand_count
                        jds.append(m)
                    except Exception:
                        pass
                elif parsed_file.exists():
                    try:
                        p = json.loads(parsed_file.read_text(encoding="utf-8"))
                        jds.append({
                            "jd_id": d.name,
                            "title": p.get("title", d.name),
                            "created_at": "Earlier",
                            "candidate_count": 0
                        })
                    except Exception:
                        pass

    jds.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jds": jds}


@router.post("/{jd_id}/evaluate")
async def evaluate_candidates_against_jd(
    jd_id: str,
    candidate_ids: Optional[str] = Form(None),
    cv_file: Optional[UploadFile] = File(None),
):
    """
    Evaluates candidate(s) against a saved JD ID.
    Supports Workflow B (existing candidates by candidate_ids) and Workflow A (brand-new CV PDF file).
    """
    jd_dir = JD_MATCHES_DIR / jd_id
    if not jd_dir.exists() or not (jd_dir / "jd_parsed.json").exists():
        raise HTTPException(status_code=404, detail=f"JD '{jd_id}' not found.")

    parsed_jd = json.loads((jd_dir / "jd_parsed.json").read_text(encoding="utf-8"))
    jd_text = ""
    if (jd_dir / "jd_original.txt").exists():
        jd_text = (jd_dir / "jd_original.txt").read_text(encoding="utf-8")

    # Workflow A: New CV File Upload
    if cv_file and cv_file.filename:
        if not cv_file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

        dest_path = UPLOADS_DIR / cv_file.filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(cv_file.file, buffer)

        cid = cv_file.filename.replace(".pdf", "").replace(" ", "_")

        # Fast text extraction
        cv_text = ""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(dest_path))
            cv_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        except Exception:
            pass

        # Robustly extract candidate profile from CV text
        exp_years = _extract_experience_from_cv(cv_text)
        highest_degree = _extract_degree_from_cv(cv_text)
        specializations = _extract_specializations_from_cv(cv_text)
        skills_found = _extract_skills_from_cv(cv_text, parsed_jd.get("required_skills", []))
        cand_display_name = cv_file.filename.replace(".pdf", "").replace("_", " ")

        custom_cand = [{
            "candidate_id": cid,
            "candidate_name": cand_display_name,
            "highest_degree": highest_degree,
            "total_experience_years": exp_years,
            "skills": skills_found + specializations,
            "specialization": ", ".join(specializations),
            "summary": (
                f"Degree: {highest_degree}. Experience: {exp_years} years. "
                f"Specializations: {', '.join(specializations)}. "
                f"Skills: {', '.join(skills_found)}. "
                + (cv_text[:300] if cv_text else "")
            ),
        }]

        match_res = compute_match_score(custom_cand[0], parsed_jd, jd_text=jd_text, skip_llm=True)
        match_res["evaluated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Load existing results to append
        existing_results = []
        res_csv = jd_dir / "results.csv"
        if res_csv.exists():
            try:
                existing_results = pd.read_csv(res_csv).to_dict(orient="records")
            except Exception:
                pass

        # Remove duplicate candidate ID if already present
        existing_results = [r for r in existing_results if str(r.get("candidate_id")) != cid]
        existing_results.insert(0, match_res)

        # Save updated results
        pd.DataFrame(existing_results).to_csv(res_csv, index=False)

        return {
            "status": "success",
            "jd_id": jd_id,
            "candidate_id": cid,
            "match_result": match_res,
            "all_results": existing_results,
            "can_trigger_full_pipeline": True,
            "message": "New CV evaluated against JD successfully. You can optionally trigger full 10-module rubric evaluation."
        }

    # Workflow B: Existing Candidate Selection
    cids_list = None
    if candidate_ids:
        cids_list = [c.strip() for c in candidate_ids.split(",") if c.strip()]

    eval_out = evaluate_jd_against_candidates(
        jd_text=jd_text,
        jd_id=jd_id,
        candidate_ids=cids_list,
        skip_llm=True
    )

    return {
        "status": "success",
        "jd_id": jd_id,
        "candidate_count": eval_out["candidate_count"],
        "results": eval_out["results"],
    }


@router.get("/{jd_id}/results")
def get_jd_results(jd_id: str):
    """
    Returns saved match results and parsed details for a specific JD.
    """
    jd_dir = JD_MATCHES_DIR / jd_id
    if not jd_dir.exists():
        raise HTTPException(status_code=404, detail=f"JD '{jd_id}' not found.")

    parsed_jd = {}
    if (jd_dir / "jd_parsed.json").exists():
        parsed_jd = json.loads((jd_dir / "jd_parsed.json").read_text(encoding="utf-8"))

    results = []
    if (jd_dir / "results.csv").exists():
        try:
            df = pd.read_csv(jd_dir / "results.csv").fillna("")
            results = df.to_dict(orient="records")
            for r in results:
                # Convert string fields back to lists for frontend
                if isinstance(r.get("matched_skills"), str):
                    r["matched_skills"] = [s.strip() for s in r["matched_skills"].split(",") if s.strip()]
                if isinstance(r.get("missing_skills"), str):
                    r["missing_skills"] = [s.strip() for s in r["missing_skills"].split(",") if s.strip()]
        except Exception:
            pass

    return {
        "jd_id": jd_id,
        "title": parsed_jd.get("title", jd_id),
        "parsed_jd": parsed_jd,
        "candidate_count": len(results),
        "results": results,
    }
