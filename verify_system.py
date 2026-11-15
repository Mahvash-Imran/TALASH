import os, sys
# Add repo root to path and set working directory
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, repo_root)
os.chdir(repo_root)

from api import routes
from api.schemas import CompareRequest

ADMIN = 'admin@talash.pk'

def main():
    print('--- Health Check ---')
    health = routes.health_check()
    print('Health:', health)

    print('\n--- List Departments ---')
    depts = routes.list_departments()
    print('Count:', depts['count'])
    print('Slugs:', [d['slug'] for d in depts['departments']])

    print('\n--- Department Profile (computer_science) ---')
    cs = routes.get_department_profile('computer_science')
    print('Slug:', cs['slug'], 'Weight sum:', sum(cs['base_weights'].values()))

    print('\n--- List Candidates ---')
    cand_list = routes.list_candidates(user_email=ADMIN)
    print('Candidates retrieved:', len(cand_list))
    if cand_list:
        first = cand_list[0]
        print('First candidate id:', first.candidate_id, 'name:', first.candidate_name)

    print('\n--- Universal Evaluation for 01_MUHAMMAD_SALMAN ---')
    uni = routes.get_candidate_universal_evaluation(
        candidate_id='01_MUHAMMAD_SALMAN',
        department='electrical_engineering',
        role_preset='faculty',
        user_email=ADMIN,
    )
    print('Candidate ID:', uni['candidate_id'])
    print('Dimension scores count:', len(uni['dimension_scores']))
    print('Gate status:', uni['gate_status'])
    print('Final score:', uni['final_score'], 'Tier:', uni['tier'])

    print('\n--- Compare Candidates ---')
    comp = routes.compare_candidates(
        CompareRequest(candidate_ids=['01_MUHAMMAD_SALMAN', '04_MUHAMMAD_FARRUKH']),
        user_email=ADMIN,
    )
    print('Comparison count:', comp.comparison_count)
    print('Ranking first entry:', comp.ranking[0])

    print('\n--- Batch Universal Evaluation ---')
    batch = routes.run_universal_batch_evaluation(
        department='computer_science',
        role_preset='faculty',
        user_email=ADMIN,
    )
    print('Batch status:', batch['status'], 'Evaluated count:', batch['evaluated_count'])

if __name__ == '__main__':
    main()
