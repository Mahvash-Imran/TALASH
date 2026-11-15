"""
uploads.py — safe handling of user-uploaded files.

* Strips any directory component from the client filename (no path traversal).
* Restricts to [A-Za-z0-9._-] and forces a .pdf extension.
* Verifies the %PDF magic bytes.
* Enforces a maximum size (TALASH_MAX_UPLOAD_MB, default 100 MB) while streaming.
"""

import os
import re
from pathlib import Path
from typing import Tuple

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = int(float(os.environ.get("TALASH_MAX_UPLOAD_MB", "100")) * 1024 * 1024)
MAX_TEXT_UPLOAD_BYTES = 5 * 1024 * 1024
_CHUNK = 1024 * 1024
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")


def safe_pdf_name(filename: str) -> str:
    """Return a filesystem-safe PDF file name derived from a client-supplied one."""
    base = (filename or "").replace("\\", "/").split("/")[-1]
    stem = base[:-4] if base.lower().endswith(".pdf") else base
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:100]
    if not stem:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    return f"{stem}.pdf"


def validate_identifier(value: str, what: str = "identifier") -> str:
    """Validate ids that are later used as path components (e.g. JD ids)."""
    if not _SAFE_ID.match(value or ""):
        raise HTTPException(status_code=400, detail=f"Invalid {what}.")
    return value


async def save_pdf_upload(file: UploadFile, dest_dir: Path) -> Tuple[Path, str]:
    """
    Validate and store an uploaded PDF under ``dest_dir``.
    Returns (saved_path, upload_id) where upload_id is the sanitised stem (spaces -> '_').
    """
    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    name = safe_pdf_name(file.filename)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name

    first = await file.read(5)
    if not first.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File is not a valid PDF.")

    size = len(first)
    try:
        with open(dest, "wb") as out:
            out.write(first)
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    return dest, name[:-4].replace(" ", "_")
