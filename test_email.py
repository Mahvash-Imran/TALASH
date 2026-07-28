import sys
sys.path.insert(0, '.')
from analysis.email_drafter import extract_missing_candidate_info, draft_followup_email
import pandas as pd

comp = pd.read_csv('data/analysis/composite_evaluations.csv', dtype=str)

# Test candidates with different profiles
test_ids = [
    '13_MANZOOR_ELLAHI',
    '04_MUHAMMAD_FARRUKH',
    '23_WAQAS_TARIQ_TOOR',
    '11_WAQAS_AMIN',          # has unverified venues
    '36_SAMANA_BATOOL',       # has 144-month real career gap
    '02_MUHAMMAD_SHAHWAZ',    # multiple gaps
    '27_SHAHEER',
]

for cid in test_ids:
    row = comp[comp['candidate_id'] == cid]
    name = row.iloc[0]['candidate_name'] if not row.empty else cid
    items = extract_missing_candidate_info(cid, 'data/analysis')
    draft = draft_followup_email(cid, name, items, skip_llm=True)
    print(f"=== {name} ===")
    print(f"Items flagged: {len(items)}")
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item[:100]}...")
    print(f"\nSubject: {draft['subject']}")
    print(f"Body:\n{draft['body']}")
    print("\n" + "="*70 + "\n")
