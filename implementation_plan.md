# TALASH — Job Description (JD) Matching Feature (Module 11)

Implementation plan for adding parallel, JD-aware evaluation capabilities to TALASH without altering existing rubric-based scoring (Modules 1–10).

## User Decisions & Approved Architecture

1. **Dedicated Tab**: JD Match exists as a standalone main tab ("JD Match") in the sidebar.
2. **Candidate Profile Integration**: Candidate detail view / action panel includes a secondary action ("Evaluate against JD") that navigates to the JD Match page with that candidate pre-selected.
3. **Persistence & Historical Reuse**: Uploaded JDs and their matching score cards are saved persistently in `data/analysis/jd_matches/{jd_id}/` (`jd_parsed.json`, `results.csv`, `jd_original.txt`), allowing instant loading and historical re-matching.
4. **New CV Workflow**: For a brand-new CV, the system runs fast JD matching directly (Modules 1, 2, 11). Once complete, a prompt gives the user the option to trigger the full 10-module composite rubric evaluation.

---

## Technical Components

### 1. Core JD Analysis Engine (`analysis/`)

#### [NEW] [jd_parser.py](file:///c:/Users/Hp/OneDrive%20-%20Higher%20Education%20Commission/Desktop/talash/talash/analysis/jd_parser.py)
- Parses raw Job Description text (or uploaded PDF/DOCX/TXT file) using the Groq LLM client (`llama-3.3-70b-versatile`).
- Extracts structured requirements JSON: `title`, `required_degree_level`, `required_discipline`, `min_experience_years`, `required_skills`, `preferred_skills`, `research_areas`, `publication_requirement`, `other_requirements`.

#### [NEW] [jd_matcher.py](file:///c:/Users/Hp/OneDrive%20-%20Higher%20Education%20Commission/Desktop/talash/talash/analysis/jd_matcher.py)
- Evaluates candidate profiles against parsed JD requirements using a 2-tier matching strategy:
  1. **Structured Overlap**: Fuzzy skill matching (`rapidfuzz`), degree level comparison, and experience year threshold check.
  2. **Semantic Alignment**: LLM-based fit score + concise rationale text for nuanced experience/domain alignment.
- Computes weighted overall JD match score (0–100%) and returns detailed breakdowns (matched skills, missing skills, degree match, experience match, semantic rationale).

#### [NEW] [run_jd_matching.py](file:///c:/Users/Hp/OneDrive%20-%20Higher%20Education%20Commission/Desktop/talash/talash/run_jd_matching.py)
- CLI entry point to run JD matching for single candidate IDs or batch evaluate all extracted candidates against a specified JD file.
- Saves results isolated in `data/analysis/jd_matches/{jd_id}/` (`jd_parsed.json`, `results.csv`, `jd_original.txt`).

---

### 2. Backend REST API (`api/`)

#### [NEW] [jd_routes.py](file:///c:/Users/Hp/OneDrive%20-%20Higher%20Education%20Commission/Desktop/talash/talash/api/jd_routes.py)
- Dedicated APIRouter under `/api/v1/jd`:
  - `POST /api/v1/jd/upload`: Accepts pasted text or uploaded JD files; parses and returns structured JD requirements.
  - `POST /api/v1/jd/{jd_id}/evaluate`: Runs matching against a newly uploaded CV (Workflow A) or a batch of existing candidate IDs (Workflow B).
  - `GET /api/v1/jd/{jd_id}/results`: Retrieves saved match results for a specific JD.
  - `GET /api/v1/jd/list`: Returns history of uploaded JDs.

#### [MODIFY] [main.py](file:///c:/Users/Hp/OneDrive%20-%20Higher%20Education%20Commission/Desktop/talash/talash/api/main.py)
- Mounts `jd_router` with `app.include_router(jd_router)`.

---

### 3. Frontend Dashboard (`frontend/`)

#### [MODIFY] [index.html](file:///c:/Users/Hp/OneDrive%20-%20Higher%20Education%20Commission/Desktop/talash/talash/frontend/index.html)
- Adds **"JD Match"** navigation item in sidebar.
- Adds `#page-jd` section with JD upload/paste, JD history selector, candidate multi-select, match score leaderboard, and breakdown modal.
- Adds "Evaluate against JD" shortcut button in candidate drawer/table.
- Adds prompt upon new CV JD evaluation offering full 10-module pipeline execution.

---

## Verification Plan

### Automated Tests
- Run unit tests for `jd_parser.py` and `jd_matcher.py`.
- Execute API test script for `/api/v1/jd/*` endpoints.
- Confirm `pipeline_orchestrator.py` outputs remain untouched.

### Manual Verification
- Test JD upload, historical selection, candidate matching, and optional full pipeline trigger.
