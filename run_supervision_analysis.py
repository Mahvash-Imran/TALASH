"""
run_supervision_analysis.py  –  CLI Entry Point for Module 4
=============================================================

Usage
-----
  # Full run with LLM summaries:
  python run_supervision_analysis.py

  # Skip LLM (rule-based summaries only, instant, no API cost):
  python run_supervision_analysis.py --skip-llm

  # Custom paths:
  python run_supervision_analysis.py \
      --supervision-csv  data/extracted/supervision.csv \
      --publications-csv data/extracted/publications.csv \
      --candidates-csv   data/extracted/candidates.csv \
      --output-dir       data/analysis

  # Use a specific model:
  python run_supervision_analysis.py --model meta-llama/llama-4-scout-17b-16e-instruct

Environment variables (from .env):
  OPENAI_API_KEY   – Groq / OpenAI API key
  OPENAI_BASE_URL  – Base URL (https://api.groq.com/openai/v1 for Groq)
  OPENAI_MODEL     – Default model name
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
# Logging  (UTF-8 for Windows)
# ---------------------------------------------------------------------------
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
        logging.FileHandler("data/logs/supervision_analysis.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("talash.supervision.main")

from analysis.supervision_analyser import SupervisionAnalyser


def run(
    supervision_csv:  str,
    publications_csv: str,
    candidates_csv:   str,
    output_dir:       str,
    skip_llm:         bool,
    model:            str,
    api_key:          str,
    base_url:         str,
):
    analyser = SupervisionAnalyser(
        supervision_csv  = supervision_csv,
        publications_csv = publications_csv,
        candidates_csv   = candidates_csv,
        output_dir       = output_dir,
        api_key          = api_key,
        model            = model,
        base_url         = base_url or None,
        skip_llm         = skip_llm,
    )
    paths = analyser.run()

    if paths:
        logger.info("")
        logger.info("Output files:")
        for key, path in paths.items():
            logger.info("  %-15s: %s", key, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TALASH Module 4 – Student Supervision Analysis"
    )
    parser.add_argument(
        "--supervision-csv",
        default=os.environ.get("SUPERVISION_CSV", "data/extracted/supervision.csv"),
        help="Path to supervision.csv from Module 1",
    )
    parser.add_argument(
        "--publications-csv",
        default=os.environ.get("PUBLICATIONS_CSV", "data/extracted/publications.csv"),
        help="Path to publications.csv from Module 1",
    )
    parser.add_argument(
        "--candidates-csv",
        default=os.environ.get("CANDIDATES_CSV", "data/extracted/candidates.csv"),
        help="Path to candidates.csv from Module 1",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analysis",
        help="Directory for Module 4 output files",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip LLM calls — use rule-based summaries only (no API cost)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        help="LLM model name",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
    )
    args = parser.parse_args()

    run(
        supervision_csv  = args.supervision_csv,
        publications_csv = args.publications_csv,
        candidates_csv   = args.candidates_csv,
        output_dir       = args.output_dir,
        skip_llm         = args.skip_llm,
        model            = args.model,
        api_key          = args.api_key,
        base_url         = args.base_url,
    )
