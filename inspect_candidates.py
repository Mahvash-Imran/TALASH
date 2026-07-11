"""
inspect_candidates.py - Show first page text of all split CVs to diagnose name extraction
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pdfplumber
from pathlib import Path

split_dir = Path('data/cvs/split')
pdfs = sorted(split_dir.glob('*.pdf'))

for pdf_path in pdfs:
    print(f"\n{'='*70}")
    print(f"FILE: {pdf_path.name}")
    print('='*70)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text() or ''
            # Show first 20 non-empty lines
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in lines[:20]:
                print(repr(line))
    except Exception as e:
        print(f"ERROR: {e}")
