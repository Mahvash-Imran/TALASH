"""
split_dataset.py  –  Utility: Split merged dataset PDF into per-candidate PDFs
===============================================================================

WHY THIS FILE EXISTS
--------------------
The talash_dataset.pdf contains all 43 candidate CVs merged into a single 251-page
PDF. Each candidate's CV begins with the header line:
  "Candidate for the Post of..."

This script detects those boundaries and writes each candidate's pages as a
separate PDF file into data/cvs/split/  so run_preprocessing.py can process
them one at a time.

USAGE
-----
    python split_dataset.py
    python split_dataset.py --input data/cvs/talash_dataset.pdf --output data/cvs/split
    python split_dataset.py --limit 3   # only split first 3 candidates (for testing)
"""

import argparse
import logging
import sys
import re
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("talash.split_dataset")

# The marker text that appears at the top of every candidate's first page
_CANDIDATE_HEADER = "Candidate for the Post"


def _extract_candidate_name(text: str) -> str:
    """
    Try to extract the candidate name from the first page text.
    Falls back to empty string if name cannot be found.
    """
    # Pattern 1: 'Name FIRST LAST  Father...' common format in TALASH dataset
    match = re.search(r"\bName\s+([A-Z][A-Z\s]{2,60}?)(?:\s{2,}|\s+Father|\s+Date|\n)", text)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"[^\w\s-]", "", name).strip()
        name = re.sub(r"\s+", "_", name)
        return name[:60]
    # Pattern 2: 'Name:  First Last'
    match = re.search(r"Name[:\s]+([A-Z][a-zA-Z\s]{2,50}?)\n", text)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"[^\w\s-]", "", name).strip()
        name = re.sub(r"\s+", "_", name)
        return name[:60]
    return ""


def split_pdf(input_path: str, output_dir: str, limit: int = 0):
    """
    Split a merged candidate PDF into per-candidate PDFs.

    Parameters
    ----------
    input_path : str
        Path to the merged PDF file.
    output_dir : str
        Folder to write individual candidate PDFs.
    limit : int
        If > 0, only extract the first N candidates (for testing).
    """
    try:
        import pdfplumber
        import pypdf
    except ImportError:
        logger.error("Missing dependencies. Run: pip install pdfplumber pypdf")
        sys.exit(1)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Scanning '%s' for candidate boundaries …", input_path.name)

    # ── Step 1: Find page indices where each candidate starts ───────────────
    candidate_start_pages = []  # list of (page_index, candidate_name)

    with pdfplumber.open(input_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info("Total pages: %d", total_pages)

        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            if _CANDIDATE_HEADER in text:
                name = _extract_candidate_name(text)
                candidate_start_pages.append((i, name))

    if not candidate_start_pages:
        logger.error(
            "No candidate boundaries found. Make sure the PDF contains "
            "'%s' at the start of each candidate's section.", _CANDIDATE_HEADER
        )
        sys.exit(1)

    logger.info("Found %d candidates.", len(candidate_start_pages))

    if limit > 0:
        candidate_start_pages = candidate_start_pages[:limit]
        logger.info("Limiting to first %d candidates.", limit)

    # ── Step 2: Write each candidate's pages to a separate PDF ─────────────
    reader = pypdf.PdfReader(str(input_path))
    written = 0

    for idx, (start_page, name) in enumerate(candidate_start_pages):
        # Determine end page (exclusive): next candidate's start, or EOF
        if idx + 1 < len(candidate_start_pages):
            end_page = candidate_start_pages[idx + 1][0]
        else:
            end_page = total_pages

        # Build output filename: prefer extracted name, fall back to index
        if name:
            out_stem = f"{(idx + 1):02d}_{name}"
        else:
            out_stem = f"candidate_{(idx + 1):02d}"

        out_path = output_dir / f"{out_stem}.pdf"

        # Skip if already exists
        if out_path.exists():
            logger.info("  [%d/%d] Already exists, skipping: %s",
                        idx + 1, len(candidate_start_pages), out_path.name)
            written += 1
            continue

        writer = pypdf.PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        with open(out_path, "wb") as f:
            writer.write(f)

        logger.info(
            "  [%d/%d] Written: %s  (pages %d–%d, %d pages)",
            idx + 1, len(candidate_start_pages),
            out_path.name,
            start_page + 1, end_page,
            end_page - start_page,
        )
        written += 1

    logger.info("Done! %d candidate PDFs written to: %s", written, output_dir)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a merged TALASH dataset PDF into per-candidate PDFs."
    )
    parser.add_argument(
        "--input",
        default="data/cvs/talash_dataset.pdf",
        help="Path to the merged PDF (default: data/cvs/talash_dataset.pdf)",
    )
    parser.add_argument(
        "--output",
        default="data/cvs/split",
        help="Output folder for individual PDFs (default: data/cvs/split)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only split first N candidates (0 = all). Use 3 for quick testing.",
    )
    args = parser.parse_args()
    split_pdf(input_path=args.input, output_dir=args.output, limit=args.limit)
