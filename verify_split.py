import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pdfplumber
from pathlib import Path

split_dir = Path('data/cvs/split')
pdfs = sorted(split_dir.glob('*.pdf'))
print(f'Total split files: {len(pdfs)}')
print()
print(f"{'Filename':<52} {'Pages':>5}  First line of content")
print("-" * 120)
for pdf_path in pdfs:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            npages = len(pdf.pages)
            first_text = pdf.pages[0].extract_text() or ''
            first_lines = [l.strip() for l in first_text.split('\n') if l.strip()]
            preview = first_lines[0][:70] if first_lines else 'empty'
            print(f"{pdf_path.name:<52} {npages:>5}  {preview}")
    except Exception as e:
        print(f"{pdf_path.name}: ERROR {e}")
