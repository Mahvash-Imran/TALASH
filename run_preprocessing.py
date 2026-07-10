"""
run_preprocessing.py  –  Main Entry Point for Module 1
=======================================================

HOW TO USE
----------
1. Place PDF CVs in:  talash/data/cvs/
2. Set your OpenAI API key:
       Windows:  $env:OPENAI_API_KEY = "sk-..."
       Linux:    export OPENAI_API_KEY="sk-..."
3. Install dependencies:
       pip install -r requirements.txt
4. Run:
       python run_preprocessing.py
       python run_preprocessing.py --cv-folder data/cvs --output-dir data/extracted --model gpt-4o-mini

WHAT THIS DOES (per the plan)
------------------------------
  Task 1.1: Scans data/cvs/ for all PDF files, reads & extracts text
  Task 1.2: Sends each CV text to GPT-4o-mini for structured JSON extraction
  Task 1.3: Cleans/normalizes all extracted data (dates, CGPA, encoding, dedup)
  Task 1.4: Organizes into 8 relational tables (one per entity type)
  Task 1.5: Exports CSVs + Excel workbook + prints parsing report

OUTPUT
------
  data/extracted/candidates.csv
  data/extracted/education.csv
  data/extracted/experience.csv
  data/extracted/skills.csv
  data/extracted/publications.csv
  data/extracted/supervision.csv
  data/extracted/books.csv
  data/extracted/patents.csv
  data/extracted/parsing_report.csv
  data/extracted/talash_extracted.xlsx   ← multi-sheet workbook
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Allow running from the talash/ package root ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import PDFReader, LLMExtractor, Normalizer, Exporter


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Force stdout to UTF-8 on Windows so special characters in log messages
# don't cause UnicodeEncodeError with the default cp1252 encoding.
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            if hasattr(sys.stdout, "buffer") else sys.stdout
        ),
        logging.FileHandler("data/logs/preprocessing.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("talash.preprocessing.main")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cv_folder: str, output_dir: str, model: str, api_key: str = "", base_url: str = ""):
    """
    Execute the full pre-processing pipeline.

    Parameters
    ----------
    cv_folder  : Path to the folder containing raw PDF CVs
    output_dir : Path to write CSVs and Excel workbook
    model      : OpenAI model name (e.g. gpt-4o-mini, gpt-4o)
    api_key    : OpenAI API key (or set OPENAI_API_KEY env var)
    base_url   : Custom base URL for alternative providers (e.g. Groq, xAI)
    """
    logger.info("=" * 55)
    logger.info("  TALASH Pre-Processing Module – Starting")
    logger.info("  CV folder  : %s", cv_folder)
    logger.info("  Output dir : %s", output_dir)
    logger.info("  LLM model  : %s", model)
    logger.info("=" * 55)

    # ── Task 1.1: PDF Ingestion ──────────────────────────────────────────────
    logger.info("[1.1] Reading PDF files …")
    reader   = PDFReader(cv_folder=cv_folder)
    pdf_results = reader.read_all()

    if not pdf_results:
        logger.error(
            "No PDF files found in '%s'.\n"
            "[!] DATASET NEEDED: Place candidate CV PDFs in that folder and re-run.",
            cv_folder
        )
        sys.exit(1)

    # ── Initialize downstream components ────────────────────────────────────
    extractor = LLMExtractor(api_key=api_key, model=model, base_url=base_url if base_url else None)
    normalizer = Normalizer()
    exporter   = Exporter(output_dir=output_dir)

    # ── Process each CV ──────────────────────────────────────────────────────
    for i, pdf in enumerate(pdf_results, 1):
        logger.info(
            "[%d/%d] Processing: %s", i, len(pdf_results), pdf.candidate_filename
        )

        if not pdf.success:
            logger.error(
                "  [FAIL] PDF read failed: %s - %s", pdf.candidate_filename, pdf.error_message
            )
            exporter.add_failure(
                pdf.candidate_filename,
                pdf_result=pdf,
                error=pdf.error_message or "PDF read failure",
            )
            continue

        # ── Task 1.2: LLM Extraction ─────────────────────────────────────
        logger.info("  [1.2] Extracting structured data via LLM …")
        extraction = extractor.extract(
            candidate_filename=pdf.candidate_filename,
            cv_text=pdf.text,
        )

        if not extraction.success:
            logger.error(
                "  [FAIL] LLM extraction failed: %s - %s",
                pdf.candidate_filename, extraction.error_message
            )
            exporter.add_failure(
                pdf.candidate_filename,
                pdf_result=pdf,
                error=extraction.error_message or "LLM extraction failure",
            )
            continue

        # ── Task 1.3: Normalization ───────────────────────────────────────
        logger.info("  [1.3] Normalizing data …")
        clean_data = normalizer.normalize(extraction.data)

        # ── Task 1.4 / 1.5: Accumulate in exporter ───────────────────────
        exporter.add_candidate(
            candidate_filename=pdf.candidate_filename,
            normalized_data=clean_data,
            validation=extraction.validation,
            pdf_result=pdf,
        )
        logger.info("  [OK] Done: %s", pdf.candidate_filename)

    # ── Task 1.5: Export CSV + Excel + Report ────────────────────────────────
    logger.info("[1.5] Exporting relational tables …")
    workbook_path = exporter.export()
    logger.info("[DONE] All done! Workbook: %s", workbook_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TALASH Module 1 - Pre-Processing: CV PDF -> structured CSV/Excel"
    )
    parser.add_argument(
        "--cv-folder",
        default="data/cvs",
        help="Folder containing PDF CV files (default: data/cvs)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/extracted",
        help="Output folder for CSVs and Excel (default: data/extracted)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model to use (default: gpt-4o-mini). Use gpt-4o for higher accuracy.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="Custom base URL for alternative providers (like Groq or xAI)",
    )

    args = parser.parse_args()
    run(
        cv_folder=args.cv_folder,
        output_dir=args.output_dir,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
