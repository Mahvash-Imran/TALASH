"""
pdf_reader.py  –  Task 1.1: PDF Ingestion
==========================================

WHY THIS FILE EXISTS
--------------------
The plan (Section 1.1) requires:
  • Accept a folder path as input; scan for all .pdf files
  • Read each PDF using pdfplumber or pypdf
  • Handle multi-page CVs gracefully
  • Log files that fail to parse

DESIGN DECISIONS
----------------
- pdfplumber is the primary reader because it preserves whitespace/layout
  better than pypdf for CV-style documents.
- pypdf is a fallback for files that pdfplumber cannot open (encrypted,
  malformed, or scanned-only PDFs).
- Each file is wrapped in try/except so one bad PDF never crashes the batch.
- A structured log entry is returned for every file (success or failure)
  so Task 1.5's parsing report can be populated.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PDFReadResult:
    """Holds the result of attempting to read one PDF file."""
    file_path: str          # Absolute path to the PDF
    candidate_filename: str  # Basename (used as a temporary candidate_id)
    success: bool           # True if text was extracted
    text: Optional[str]     # Full concatenated text of all pages (or None)
    page_count: int         # Number of pages detected
    error_message: Optional[str] = None   # Populated on failure
    warnings: List[str] = field(default_factory=list)


class PDFReader:
    """
    Scans a folder for PDF CVs and extracts their text content.

    Usage
    -----
    reader = PDFReader(cv_folder="data/cvs")
    results = reader.read_all()
    for r in results:
        if r.success:
            print(r.text[:200])
        else:
            print(f"FAILED: {r.file_path} – {r.error_message}")
    """

    def __init__(self, cv_folder: str):
        """
        Parameters
        ----------
        cv_folder : str
            Path to the directory containing PDF CVs.
            Sub-directories are NOT scanned (only top-level .pdf files).
        """
        self.cv_folder = Path(cv_folder)
        if not self.cv_folder.exists():
            raise FileNotFoundError(
                f"CV folder not found: {self.cv_folder.resolve()}\n"
                "Create the folder and place PDF files inside it."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read_all(self) -> List[PDFReadResult]:
        """
        Scan cv_folder for all .pdf files (case-insensitive) and
        attempt to extract text from each one.

        Returns
        -------
        list of PDFReadResult
            One entry per PDF file found. Always returns a result even on
            failure so the calling pipeline can generate a complete report.
        """
        pdf_files = sorted(self.cv_folder.glob("*.pdf"))
        pdf_files += sorted(self.cv_folder.glob("*.PDF"))
        # Remove duplicates (Windows glob may match same file twice)
        seen = set()
        unique_files = []
        for p in pdf_files:
            key = str(p).lower()
            if key not in seen:
                seen.add(key)
                unique_files.append(p)

        if not unique_files:
            logger.warning(
                "No PDF files found in '%s'. "
                "Place CV PDFs in that folder and re-run.",
                self.cv_folder
            )

        results = []
        for pdf_path in unique_files:
            result = self._read_single(pdf_path)
            results.append(result)

        logger.info(
            "PDF ingestion complete: %d/%d files read successfully.",
            sum(1 for r in results if r.success),
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_single(self, pdf_path: Path) -> PDFReadResult:
        """
        Try pdfplumber first; fall back to pypdf if it fails.
        Returns a PDFReadResult regardless of outcome.
        """
        candidate_filename = pdf_path.stem  # filename without extension

        # --- Attempt 1: pdfplumber (better layout preservation) ---
        try:
            result = self._read_with_pdfplumber(pdf_path, candidate_filename)
            if result.success and result.text and result.text.strip():
                return result
            # pdfplumber opened the file but yielded no text (likely scanned)
            result.warnings.append(
                "pdfplumber extracted no text (possibly scanned PDF); "
                "trying pypdf as fallback."
            )
        except Exception as e:
            logger.debug("pdfplumber failed for '%s': %s", pdf_path.name, e)
            result = PDFReadResult(
                file_path=str(pdf_path),
                candidate_filename=candidate_filename,
                success=False,
                text=None,
                page_count=0,
                warnings=[f"pdfplumber error: {e}"]
            )

        # --- Attempt 2: pypdf fallback ---
        try:
            pypdf_result = self._read_with_pypdf(pdf_path, candidate_filename)
            if pypdf_result.success and pypdf_result.text and pypdf_result.text.strip():
                pypdf_result.warnings += result.warnings  # carry over warnings
                return pypdf_result
        except Exception as e:
            logger.debug("pypdf also failed for '%s': %s", pdf_path.name, e)

        # --- Both failed ---
        logger.error("Could not extract text from '%s'.", pdf_path.name)
        return PDFReadResult(
            file_path=str(pdf_path),
            candidate_filename=candidate_filename,
            success=False,
            text=None,
            page_count=result.page_count,
            error_message=(
                "Both pdfplumber and pypdf failed to extract text. "
                "The file may be scanned or encrypted."
            ),
            warnings=result.warnings,
        )

    # ------------------------------------------------------------------

    def _read_with_pdfplumber(
        self, pdf_path: Path, candidate_filename: str
    ) -> PDFReadResult:
        """Extract text using pdfplumber, concatenating all pages."""
        import pdfplumber  # import here so missing lib gives clear error

        pages_text: List[str] = []
        warnings: List[str] = []

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                    else:
                        warnings.append(f"Page {i + 1} yielded no text.")
                except Exception as e:
                    warnings.append(f"Page {i + 1} extraction error: {e}")

        full_text = "\n\n".join(pages_text)
        success = bool(full_text.strip())

        return PDFReadResult(
            file_path=str(pdf_path),
            candidate_filename=candidate_filename,
            success=success,
            text=full_text if success else None,
            page_count=page_count,
            warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _read_with_pypdf(
        self, pdf_path: Path, candidate_filename: str
    ) -> PDFReadResult:
        """Extract text using pypdf as a fallback."""
        import pypdf  # import here so missing lib gives a clear error

        pages_text: List[str] = []
        warnings: List[str] = []

        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            page_count = len(reader.pages)

            # Warn if the PDF is encrypted (pypdf may still read it if no password needed)
            if reader.is_encrypted:
                warnings.append(
                    "PDF is encrypted; extraction may be incomplete.")

            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                    else:
                        warnings.append(
                            f"pypdf: Page {i + 1} yielded no text.")
                except Exception as e:
                    warnings.append(f"pypdf: Page {i + 1} error: {e}")

        full_text = "\n\n".join(pages_text)
        success = bool(full_text.strip())

        return PDFReadResult(
            file_path=str(pdf_path),
            candidate_filename=candidate_filename,
            success=success,
            text=full_text if success else None,
            page_count=page_count,
            warnings=warnings,
        )
