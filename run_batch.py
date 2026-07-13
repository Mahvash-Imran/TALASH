"""
run_batch.py  –  Batch processor for TALASH candidates
======================================================
This script runs pre-processing on remaining candidates in order,
automatically detecting rate limits and running educational profile
analysis on all successfully processed candidates.
"""

import sys
import io
import os
import subprocess
from pathlib import Path
import pandas as pd

# Force UTF-8 logging/stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load_processed_candidates() -> set:
    report_path = Path("data/extracted/parsing_report.csv")
    if not report_path.exists():
        return set()
    try:
        df = pd.read_csv(report_path)
        # Succeeded candidate IDs
        succeeded = df[df["status"] == "success"]["candidate_id"].dropna().tolist()
        return set(succeeded)
    except Exception as e:
        print(f"Warning: Could not read parsing report: {e}")
        return set()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TALASH Batch Processor")
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="LLM model name to use"
    )
    args = parser.parse_args()
    model = args.model

    split_dir = Path("data/cvs/split")
    if not split_dir.exists():
        print(f"Error: Split directory does not exist at {split_dir}")
        sys.exit(1)

    pdfs = sorted(split_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Error: No PDF files found in {split_dir}")
        sys.exit(1)

    processed = load_processed_candidates()
    print(f"Found {len(pdfs)} total candidates.")
    print(f"Already successfully processed: {len(processed)}")

    to_process = []
    for pdf in pdfs:
        cid = pdf.stem
        if cid not in processed:
            to_process.append(pdf)

    if not to_process:
        print("All candidates have been successfully processed!")
        # Run module 2 to be safe
        subprocess.run(["python", "run_educational_profile.py", "--model", model])
        return

    print(f"Remaining to process: {len(to_process)}")
    print("Starting batch processing...")

    success_count = 0
    rate_limited = False

    for i, pdf in enumerate(to_process, 1):
        cid = pdf.stem
        print(f"\n==================================================")
        print(f"[{i}/{len(to_process)}] Processing: {cid}")
        print(f"==================================================")

        # Run preprocessing for this single candidate
        cmd = [
            "python", "run_preprocessing.py",
            "--cv-folder", str(split_dir),
            "--output-dir", "data/extracted",
            "--candidate", cid,
            "--model", model
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        # Verify if successfully added
        new_processed = load_processed_candidates()
        if cid in new_processed:
            print(f"[OK] Successfully processed candidate: {cid}")
            success_count += 1
        else:
            print(f"[FAIL] Candidate {cid} was not successfully processed.")
            # Check if rate limited only when this candidate failed
            output_lower = (result.stdout + "\n" + result.stderr).lower()
            fail_prefix = f"[fail] llm extraction failed: {cid.lower()}"
            is_rate_limit = False
            for line in output_lower.split("\n"):
                if fail_prefix in line:
                    if "429" in line or "rate_limit" in line or "rate limit" in line:
                        is_rate_limit = True
                        break
            if is_rate_limit:
                print(f"\n[!] Rate limit reached at candidate {cid}. Stopping batch.")
                rate_limited = True
                break

    print(f"\nBatch finished. Successfully processed {success_count} new candidate(s).")
    
    # Run Module 2 analysis on all successful candidates
    print("\nRunning Module 2 (Educational Profile Analysis)...")
    subprocess.run(["python", "run_educational_profile.py", "--model", model])

    if rate_limited:
        print("\n[!] Processing stopped because the Groq API rate limit was reached.")
        print("Please wait for the rate limit to reset (usually a few hours/day) and run 'python run_batch.py' again.")

if __name__ == "__main__":
    main()
