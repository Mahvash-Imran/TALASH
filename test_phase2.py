import sys
sys.path.insert(0, 'talash')

import profile_loader
import profile_generator
from models import RolePreset

print('=== 1. TEST LISTING SEED PROFILES ===')
profs = profile_loader.list_profiles()
print(f'Total profiles found: {len(profs)}')
assert len(profs) >= 6, 'Must have at least 6 seed profiles'
for p in profs:
    print(f"  • {p['slug']} ({p['department_name']}): status={p['status']}, artifacts={p['artifact_types']}")

print('\n=== 2. TEST LOADING PROFILES ===')
cs = profile_loader.get_profile('computer_science')
assert cs is not None
print(f"Loaded CS: {cs.department_name}, Version: {cs.version}")
print(f"  Base weights sum: {round(sum(cs.base_weights.values()), 4)}")
faculty_weights = cs.get_effective_weights(RolePreset.FACULTY)
print(f"  Faculty effective weights sum: {round(sum(faculty_weights.values()), 4)}")

print('\n=== 3. TEST FUZZY / KEYWORD RESOLUTION ===')
test_queries = [
    ('Clinical Medicine', 'medicine'),
    ('Faculty of Law', 'law'),
    ('Electrical & Electronics Engineering', 'electrical_engineering'),
    ('Department of Business Management', 'business_administration'),
    ('English Literature', 'english_literature'),
]
for query, expected_slug in test_queries:
    matched = profile_loader.get_profile(query)
    assert matched is not None, f"Failed to match query '{query}'"
    print(f"  Query '{query}' => {matched.slug} (Expected: {expected_slug})")
    assert matched.slug == expected_slug

print('\n=== 4. TEST DYNAMIC PROFILE GENERATOR ===')
new_dept = 'Civil Engineering'
sample_jd = (
    'We are seeking an Assistant Professor of Civil Engineering. '
    'Requirements: PhD in Civil Engineering, PEC registration as professional engineer, '
    'at least 3 years teaching/research experience, and publications in HEC recognized journals.'
)
new_profile = profile_generator.generate_department_profile(
    department_name=new_dept,
    sample_jd_text=sample_jd,
    save=True
)
print(f"Generated Profile: {new_profile.slug}")
print(f"  Department: {new_profile.department_name}")
print(f"  Status: {new_profile.status} (Verified strictly 'draft')")
assert new_profile.status == 'draft', 'Generated profile must be in draft status'
print(f"  Base weights sum: {round(sum(new_profile.base_weights.values()), 4)}")
print(f"  Hard gates: {[g.name for g in new_profile.hard_gates]}")
print(f"  Artifact types: {list(new_profile.artifact_taxonomy.keys())}")

print('\n=== 5. VERIFY GENERATED PROFILE RETRIEVAL ===')
retrieved = profile_loader.get_profile('civil_engineering')
assert retrieved is not None
assert retrieved.status == 'draft'
print(f"Successfully retrieved generated profile from disk cache: {retrieved.slug} (status={retrieved.status})")

# Clean up generated draft profile to preserve the 6 canonical seed profiles
profile_loader.delete_profile('civil_engineering')
assert profile_loader.get_profile('civil_engineering', allow_fuzzy=False) is None
print("Cleaned up temporary test profile; 6 canonical seed profiles intact.")

print('\nAll Phase 2 tests passed successfully with 100% assertions satisfied!')
