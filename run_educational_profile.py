"""
run_educational_profile.py  –  CLI Entry Point for Module 2
============================================================

Usage
-----
  # With LLM summaries (Task 2.9 — uses Groq API):
  python run_educational_profile.py

  # Skip LLM (tasks 2.1-2.8 only — no API calls, instant results):
  python run_educational_profile.py --skip-llm

  # Custom paths:
  python run_educational_profile.py \\
      --education-csv  data/extracted/education.csv \\
      --experience-csv data/extracted/experience.csv \\
      --output-dir     data/analysis

Environment variables (from .env):
  OPENAI_API_KEY   – Groq / OpenAI API key
  OPENAI_BASE_URL  – Base URL (https://api.groq.com/openai/v1 for Groq)
  OPENAI_MODEL     – Model name (default: llama-3.3-70b-versatile)
"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the talash/ package root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging setup  (UTF-8 stdout for Windows compatibility)
# ---------------------------------------------------------------------------
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.makedirs("data/logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            if hasattr(sys.stdout, "buffer") else sys.stdout
        ),
        logging.FileHandler("data/logs/educational_profile.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("talash.analysis.main")

from analysis.educational_profile import EducationalProfileAnalyser


def run(
    education_csv:  str,
    experience_csv: str,
    output_dir:     str,
    skip_llm:       bool,
    model:          str,
    api_key:        str,
    base_url:       str,
):
    analyser = EducationalProfileAnalyser(
        education_csv  = education_csv,
        experience_csv = experience_csv,
        output_dir     = output_dir,
        api_key        = api_key,
        model          = model,
        base_url       = base_url,
        skip_llm       = skip_llm,
    )
    paths = analyser.run()

    if paths:
        logger.info("")
        logger.info("Output files:")
        for key, path in paths.items():
            logger.info("  %-10s: %s", key, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TALASH Module 2 - Educational Profile Analysis"
    )
    parser.add_argument(
        "--education-csv",
        default=os.environ.get("EDUCATION_CSV", "data/extracted/education.csv"),
        help="Path to education.csv from Module 1",
    )
    parser.add_argument(
        "--experience-csv",
        default=os.environ.get("EXPERIENCE_CSV", "data/extracted/experience.csv"),
        help="Path to experience.csv from Module 1",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Directory for Module 2 output files",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip Task 2.9 LLM summaries (run pure deterministic pipeline only)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "groq/compound-mini"),
        help="LLM model name",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="API base URL (defaults to OPENAI_BASE_URL env var)",
    )
    args = parser.parse_args()

    run(
        education_csv  = args.education_csv,
        experience_csv = args.experience_csv,
        output_dir     = args.output_dir,
        skip_llm       = args.skip_llm,
        model          = args.model,
        api_key        = args.api_key,
        base_url       = args.base_url or None,
    )
