import sys
sys.path.insert(0, '.')
from analysis.email_drafter import extract_missing_candidate_info, draft_followup_email
import pandas as pd

comp = pd.read_csv('data/analysis/composite_evaluations.csv', dtype=str)
test_ids = ['04_MUHAMMAD_FARRUKH', '27_SHAHEER', '05_MUHAMMAD_MAJID', '11_WAQAS_AMIN', '21_SAMAN_FATIMA']
for cid in test_ids:
    row = comp[comp['candidate_id'] == cid]
    name = row.iloc[0]['candidate_name'] if not row.empty else cid
    items = extract_missing_candidate_info(cid, 'data/analysis')
    draft = draft_followup_email(cid, name, items, skip_llm=True)
    print(f"=== {cid} ({name}) ===")
    print(f"Missing items ({len(items)}): {items}")
    print(f"Subject: {draft['subject']}")
    print(f"Body preview: {draft['body'][:250]}")
    print()
