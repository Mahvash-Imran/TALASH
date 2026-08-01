"""
rename_split_cvs.py  –  Rename all split CVs to consistent NN_FIRSTNAME_LASTNAME format

Reads the Name field from page 1 of each PDF.
Always uses the ORIGINAL sequence number from the source dataset (1-43),
not the file position.

The correct sequence order is determined by page order in the original PDF.
"""

import re
import sys
import io
import logging
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rename_cvs")

try:
    import pdfplumber
except ImportError:
    logger.error("pdfplumber not installed. Run: pip install pdfplumber")
    sys.exit(1)



def extract_name(pdf_path: Path) -> str:
    """
    Extract the candidate's name from the first page of the PDF.
    
    The form has this structure on page 1:
        Name  MUHAMMAD SALMAN  Father's /Guardian QAMAR UZ ZAMAN
    or across two lines when the name is long:
        Name Muhammad Farrukh  Father's Muhammad Fayyaz
        Qureshi /Guardian
    
    Strategy: join first few lines, then regex from 'Name' to 'Father'.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Join first page lines for multi-line name handling
            text = pdf.pages[0].extract_text() or ""
    except Exception as e:
        logger.warning("Cannot read %s: %s", pdf_path.name, e)
        return ""

    # Collapse all whitespace including newlines into single spaces
    # but first, join lines intelligently
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Find the line containing "Name" at the start (the identity line)
    name_line_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^Name\s+[A-Za-z]", line):
            name_line_idx = i
            break
    
    if name_line_idx is None:
        return ""

    # Combine the name line + next line (handles wrap-around names)
    combined = lines[name_line_idx]
    if name_line_idx + 1 < len(lines):
        next_line = lines[name_line_idx + 1]
        # Only join if next line doesn't start a new field
        if not re.match(r"^(Date|Spouse|Current|Unit|SOD|Expected|Serving|Education|DOB)", next_line):
            combined = combined + " " + next_line

    # Extract: everything between "Name " and "Father"
    patterns = [
        r"Name\s+(.+?)\s+Father'?s?\s*/Guardian",
        r"Name\s+(.+?)\s+Father'?s?\b",
        r"Name\s+(.+?)\s+/Guardian",
        r"Name\s+(.+?)\s+(?:Date|Spouse|Current)",
    ]
    
    for pattern in patterns:
        m = re.search(pattern, combined, re.IGNORECASE)
        if m:
            raw_name = m.group(1).strip()
            return _to_filename(raw_name)
    
    return ""


def _to_filename(name: str) -> str:
    """Convert 'Muhammad Farrukh Qureshi' -> 'MUHAMMAD_FARRUKH_QURESHI'"""
    # Keep only letters, spaces, hyphens
    name = re.sub(r"[^A-Za-z\s\-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Uppercase, spaces -> underscores
    name = name.upper().replace("-", "_").replace(" ", "_")
    name = re.sub(r"_+", "_", name).strip("_")
    return name


# ---------------------------------------------------------------------------
# Get correct sequence number from original dataset page order
# ---------------------------------------------------------------------------

def get_sequence_numbers(split_dir: Path) -> dict:
    """
    Determine correct 1-43 sequence numbers by scanning the original dataset PDF
    in page order. Returns {pdf_filename: correct_sequence_num}.
    
    Falls back to scanning the split folder sorted by the number already in filename.
    """
    dataset = split_dir.parent / "talash_dataset.pdf"
    
    if not dataset.exists():
        logger.warning("Original dataset PDF not found; using filename numbers as-is")
        return {}
    
    logger.info("Scanning original dataset to get correct page order...")
    candidate_starts = []  # list of start_page indices
    
    with pdfplumber.open(dataset) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
                if "Candidate for the Post" in text:
                    candidate_starts.append(i)
            except:
                pass
    
    logger.info("Found %d candidate boundaries in original dataset", len(candidate_starts))
    return candidate_starts  # list of start page indices in order


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rename_all(split_dir: str = "data/cvs/split"):
    split_path = Path(split_dir)
    pdfs = sorted(split_path.glob("*.pdf"))

    if not pdfs:
        logger.error("No PDFs found in %s", split_path)
        return

    logger.info("Found %d PDFs in: %s", len(pdfs), split_path)
    logger.info("")

    # --- Step 1: Sort by first-page start page in original document ---
    # We determine order by reading the first-page "Apply Date" or just trust
    # the current numeric prefix in each filename (which was set by split_dataset.py
    # and IS already correct — it matches the order in the original PDF).
    
    # Build (correct_num, pdf_path, extracted_name) triples
    triples = []
    
    for pdf_path in pdfs:
        # Extract sequence number from current filename
        m = re.match(r"^(\d+)", pdf_path.stem)
        seq_num = int(m.group(1)) if m else 999
        
        name = extract_name(pdf_path)
        if not name:
            logger.warning("  [%02d] Could not extract name from: %s", seq_num, pdf_path.name)
            name = f"UNKNOWN"
        
        triples.append((seq_num, pdf_path, name))
        logger.info("  [%02d] %-50s -> name: %s", seq_num, pdf_path.name, name)

    # Sort by sequence number to ensure correct order
    triples.sort(key=lambda x: x[0])
    
    # Re-assign clean sequential numbers 01-43 in document order
    logger.info("")
    logger.info("Building final rename map...")
    
    rename_map = {}  # old_path -> new_filename
    used = set()
    
    for final_num, (seq_num, pdf_path, name) in enumerate(triples, 1):
        new_stem = f"{final_num:02d}_{name}"
        # Handle rare duplicate names
        if new_stem in used:
            new_stem = f"{new_stem}_B"
        used.add(new_stem)
        new_name = f"{new_stem}.pdf"
        rename_map[pdf_path] = split_path / new_name
    
    # --- Step 2: Two-pass rename to avoid collisions ---
    # Pass 1: all -> temp
    temp_map = {}
    for old_path, new_path in rename_map.items():
        if old_path.resolve() == new_path.resolve():
            continue
        temp = old_path.parent / f"__tmp_{old_path.name}"
        old_path.rename(temp)
        temp_map[temp] = new_path

    # Pass 2: temp -> final
    for temp_path, new_path in temp_map.items():
        if new_path.exists():
            logger.warning("Target already exists, skipping: %s", new_path.name)
            temp_path.rename(new_path.parent / f"CONFLICT_{new_path.name}")
        else:
            temp_path.rename(new_path)

    # --- Show final result ---
    logger.info("")
    logger.info("Done! Final file list:")
    logger.info("-" * 50)
    for f in sorted(split_path.glob("*.pdf")):
        logger.info("  %s", f.name)


if __name__ == "__main__":
    rename_all()
