"""
run_research_profile.py  –  CLI Entry Point for Module 3
=========================================================

Usage
-----
  # Full run with LLM (venue reconstruction + indexing lookup + summaries):
  python run_research_profile.py

  # Skip all LLM calls (pure deterministic: authorship + indexing keywords only):
  python run_research_profile.py --skip-llm

  # Skip venue reconstruction only (saves tokens when venues are already clean):
  python run_research_profile.py --no-reconstruct-venues

  # Custom paths:
  python run_research_profile.py \\
      --publications-csv  data/extracted/publications.csv \\
      --candidates-csv    data/extracted/candidates.csv \\
      --output-dir        data/analysis

  # Use a specific model:
  python run_research_profile.py --model meta-llama/llama-4-scout-17b-16e-instruct

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
# Logging setup  (UTF-8 stdout for Windows compatibility)
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
        logging.FileHandler("data/logs/research_profile.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("talash.research.main")

from analysis.research_profile import ResearchProfileAnalyser


def run(
    publications_csv:   str,
    candidates_csv:     str,
    output_dir:         str,
    skip_llm:           bool,
    reconstruct_venues: bool,
    model:              str,
    api_key:            str,
    base_url:           str,
):
    analyser = ResearchProfileAnalyser(
        publications_csv   = publications_csv,
        candidates_csv     = candidates_csv,
        output_dir         = output_dir,
        api_key            = api_key,
        model              = model,
        base_url           = base_url or None,
        skip_llm           = skip_llm,
        reconstruct_venues = reconstruct_venues,
    )
    paths = analyser.run()

    if paths:
        logger.info("")
        logger.info("Output files:")
        for key, path in paths.items():
            logger.info("  %-15s: %s", key, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TALASH Module 3 - Research Profile Analysis"
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
        help="Directory for Module 3 output files",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        default=False,
        help="Skip all LLM calls (authorship + keyword-based indexing only, no API cost)",
    )
    parser.add_argument(
        "--no-reconstruct-venues",
        action="store_true",
        default=False,
        help="Disable LLM-based venue name reconstruction for truncated entries",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        help="LLM model name (default: meta-llama/llama-4-scout-17b-16e-instruct)",
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
        publications_csv   = args.publications_csv,
        candidates_csv     = args.candidates_csv,
        output_dir         = args.output_dir,
        skip_llm           = args.skip_llm,
        reconstruct_venues = not args.no_reconstruct_venues,
        model              = args.model,
        api_key            = args.api_key,
        base_url           = args.base_url,
    )
