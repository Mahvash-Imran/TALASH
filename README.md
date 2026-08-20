# TALASH

**Talent Acquisition and Longitudinal Academic Scoring for Higher Education**

TALASH is an automated faculty candidate evaluation system developed for the Higher Education Commission of Pakistan. It ingests PDF curricula vitae, parses and normalises structured data across nine domain modules, computes a composite suitability score across seven academic dimensions, matches candidates against job descriptions using AI, and exposes results through a REST API and an interactive single-page dashboard.

The system is designed to eliminate manual screening bottlenecks in large-scale faculty recruitment cycles, where evaluation committees are required to assess dozens to hundreds of CVs against a standardised rubric of academic merit.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Pipeline](#module-pipeline)
3. [Module 11: Job Description Matching](#module-11-job-description-matching)
4. [Scoring Dimensions](#scoring-dimensions)
5. [Repository Structure](#repository-structure)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [Running the Pipeline](#running-the-pipeline)
9. [API Reference](#api-reference)
10. [Frontend Dashboard](#frontend-dashboard)
11. [Output Artefacts](#output-artefacts)
12. [Technology Stack](#technology-stack)
13. [Deployment](#deployment)
14. [Development Status](#development-status)

---

## Architecture Overview

The system follows a sequential batch-processing architecture. An orchestrator script invokes each module in order, with each module reading from the shared data directory and writing its output back to it. The REST API layer reads these outputs at request time. The frontend communicates with the API when the backend is running and falls back to embedded data when it is not.

```
PDF CVs
  |
  v
Module 1: PDF Splitting and Preprocessing
  |
  v
Module 2: Structured Data Extraction (LLM-assisted)
  |
  v
Modules 3-10: Domain Analysis
  |            |
  |            +-- Module 3:  Educational Profile Analysis
  |            +-- Module 4:  Research Output Analysis
  |            +-- Module 5:  Supervision Record Analysis
  |            +-- Module 6:  Experience and Tenure Analysis
  |            +-- Module 7:  Innovation and Patent Analysis
  |            +-- Module 8:  Research Breadth and Topic Modelling
  |            +-- Module 9:  Collaboration Network Analysis
  |            +-- Module 10: Composite Scoring and Ranking
  |
  v
Module 11: Job Description Matching (independent, on-demand)
  |
  v
data/analysis/   <-- CSVs, XLSX, text reports, JD match results
  |
  v
FastAPI REST API  <-- /api/v1/candidates, /api/v1/jd/*, /api/v1/email/{id}
  |
  v
Frontend Dashboard  <-- Single-page application (vanilla JS/CSS)
```

---

## Module Pipeline

### Module 1: PDF Preprocessing

Ingests the combined CV dataset PDF, splits it into individual candidate files using page-boundary detection, and writes them to `data/cvs/split/`. Performs text normalisation, Unicode sanitisation, and encoding standardisation.

Entry point: `run_preprocessing.py`

### Module 2: Structured Data Extraction

Passes each candidate raw text through an LLM (Groq-hosted, configurable) using structured prompt templates to extract entities including personal metadata, educational qualifications, publications, patents, supervision records, employment history, and skills. Outputs are written as normalised CSVs to `data/extracted/`.

Entry point: `preprocessing/llm_extractor.py`

### Module 3: Educational Profile Analysis

Evaluates degree credentials against a ranked university database (`data/rankings/universities.csv`). Assigns education scores based on institution tier, degree level, and discipline relevance. Flags inconsistencies such as missing graduation years or unverifiable institutions.

Entry point: `run_educational_profile.py`

### Module 4: Research Output Analysis

Analyses publication records against a journal quality list (`data/rankings/beall_list_journals.csv`) and a live lookup cache for impact factor retrieval. Computes a research score weighted by publication count, journal tier, citation proxies, and recency. Caches external API responses in `data/research_cache/`.

Entry point: `run_research_profile.py`

### Module 5: Supervision Record Analysis

Parses thesis supervision entries and cross-references them against declared employment periods. Flags supervision claims with no corresponding employment period. Computes a supervision score based on PhD and MS student counts and completion status.

Entry point: `run_supervision_analysis.py`

### Module 6: Experience and Tenure Analysis

Constructs a unified employment timeline per candidate, identifies overlapping periods, and computes total verified academic and industry experience in years. Assigns experience scores according to a configurable tenure-to-score mapping table.

Entry point: `run_experience_analysis.py`

### Module 7: Innovation and Patent Analysis

Identifies patent records from extracted data and validates them against standard formats. Computes an innovation score based on granted versus filed patents, jurisdictional scope, and co-inventor relationships.

Entry point: `run_patent_analysis.py`

### Module 8: Research Breadth and Topic Modelling

Applies keyword clustering across publication titles and abstracts to identify distinct research themes per candidate. Computes a breadth score reflecting interdisciplinary coverage and thematic diversity rather than volume alone.

Entry point: `run_topic_analysis.py`

### Module 9: Collaboration Network Analysis

Extracts co-authorship relationships from publications and generates collaboration profiles indicating the extent of national and international research partnerships. Assigns a collaboration score based on network diversity and co-authored output.

Entry point: `run_collaboration_analysis.py`

### Module 10: Composite Scoring and Ranking

Aggregates dimension scores from Modules 3 through 9 using a configurable weighted formula to produce a single composite score per candidate on a 0 to 100 scale. Classifies candidates into four suitability tiers and generates the final ranked output.

Entry point: `pipeline_orchestrator.py` (full pipeline) or `run_batch.py` (scoring only)

---

## Module 11: Job Description Matching

Module 11 is an independent, on-demand feature added to complement the core evaluation pipeline. It allows a hiring committee to paste or upload a Job Description and instantly receive ranked match scores for any selection of candidates, without needing to re-run the full evaluation pipeline.

### How It Works

A Job Description is submitted as plain text or a PDF through the dashboard or API. The system parses it to extract the required degree level, minimum years of experience, and required technical skills. Each candidate is then scored against the JD across three dimensions: degree eligibility, experience adequacy, and skill overlap. A weighted composite match score is computed and each candidate is assigned a fit tier.

**Fit tiers**

- Strong Fit: 75 percent and above
- Moderate Fit: 50 to 74 percent
- Weak Fit: below 50 percent

### Candidate-Specific AI Rationale

Every match result includes a dynamically generated natural-language rationale unique to each candidate. The rationale states the candidate's degree, years of experience, which required skills were matched, which were missing, and whether the experience threshold was met. This replaces generic or identical explanations with meaningful, candidate-specific reasoning.

### New CV Upload for JD Screening

A new CV PDF can be uploaded directly on the JD Match page without running the full pipeline. The system extracts experience from date ranges in the employment section, identifies the highest degree, and detects skills from a broad technical keyword vocabulary aligned with the JD requirements. The candidate is immediately scored against the active JD and the result is stored historically.

After a new CV is screened, the dashboard offers an option to trigger the full 10-module rubric evaluation for that candidate.

### Historical JD Storage

Each uploaded Job Description is saved with a unique identifier in `data/analysis/jd_matches/`. Match results are persisted as CSV files alongside the parsed JD and original text. Previously uploaded JDs can be loaded from a history dropdown without re-uploading, and their stored results are retrieved instantly.

### Experience Extraction Accuracy

The experience parser is section-aware. It locates the professional experience section of the CV and sums only the date ranges found there, avoiding double-counting of education period date ranges. It also recognises explicit experience statements such as "12 years of experience" and uses them as a direct input when present.

### JD Match API Endpoints

All JD endpoints are prefixed with `/api/v1/jd`.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/jd/upload` | Upload a new JD as text or PDF. Returns parsed JD metadata and a unique JD ID. |
| GET | `/api/v1/jd/list` | Returns a list of all historically stored JDs with candidate counts. |
| POST | `/api/v1/jd/{jd_id}/evaluate` | Evaluates selected existing candidates or a new uploaded CV against the specified JD. |
| GET | `/api/v1/jd/{jd_id}/results` | Returns stored match results for a previously evaluated JD. |

---

## Scoring Dimensions

The composite score is a weighted sum across seven dimensions. Default weights are indicated below.

| Dimension | Max Points | Default Weight | Source Module |
|---|---|---|---|
| Education | 20 | 20% | Module 3 |
| Research Output | 25 | 25% | Module 4 |
| Experience and Tenure | 15 | 15% | Module 6 |
| Supervision | 10 | 10% | Module 5 |
| Innovation and Patents | 10 | 10% | Module 7 |
| Research Breadth | 10 | 10% | Module 8 |
| Collaboration | 10 | 10% | Module 9 |

**Tier Classification**

| Tier | Score Range | Label |
|---|---|---|
| Tier 1 | 80 and above | Exceptional |
| Tier 2 | 65 to 79 | Strong |
| Tier 3 | 50 to 64 | Moderate |
| Tier 4 | Below 50 | Needs Clarification |

Dimension weights are adjustable at runtime through the frontend dashboard Smart Weighting interface, which re-ranks candidates without modifying stored scores.

---

## Repository Structure

```
talash/
|
|-- analysis/                   Domain analysis modules (M3-M10) and JD matching (M11)
|   |-- educational_profile.py
|   |-- research_profile.py
|   |-- supervision_analysis.py
|   |-- experience_analysis.py
|   |-- patent_analysis.py
|   |-- topic_analysis.py
|   |-- collaboration_analysis.py
|   |-- composite_scorer.py
|   |-- jd_parser.py            Module 11: JD text parsing and skill/experience extraction
|   |-- jd_matcher.py           Module 11: candidate-JD scoring and AI rationale generation
|
|-- api/                        FastAPI application
|   |-- main.py                 Application entry point and static mounts
|   |-- routes.py               Core candidate API route handlers
|   |-- jd_routes.py            Module 11 JD match API route handlers
|   |-- schemas.py              Pydantic request/response models
|
|-- data/
|   |-- cvs/                    Raw and split PDF files
|   |-- extracted/              Module 2 structured CSV outputs
|   |-- analysis/               Module 3-10 analysis outputs (CSV, XLSX, TXT)
|   |   |-- jd_matches/         Module 11 JD match results, one subdirectory per JD
|   |-- rankings/               Reference data (universities, journal lists)
|   |-- research_cache/         Cached external API responses
|   |-- uploads/                Uploaded CV PDFs from the JD new-CV workflow
|   |-- logs/                   Per-module execution logs
|
|-- frontend/
|   |-- index.html              Single-page dashboard (vanilla JS/CSS)
|
|-- preprocessing/              Module 1 and 2 utilities
|   |-- pdf_reader.py
|   |-- llm_extractor.py
|   |-- normalizer.py
|   |-- exporter.py
|
|-- pipeline_orchestrator.py    Full end-to-end pipeline runner
|-- run_jd_matching.py          Module 11 standalone runner and evaluate helper
|-- run_backend.py              Uvicorn server launcher
|-- run_batch.py                Scoring-only batch runner
|-- run_preprocessing.py        Module 1 runner
|-- run_*.py                    Individual module runners (M3-M9)
|-- Dockerfile                  Production container definition
|-- railway.toml                Railway deployment configuration
|-- requirements.txt
|-- .env.example
```

---

## Installation

Prerequisites: Python 3.10 or later. A virtual environment is strongly recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and populate the required values.

```
OPENAI_API_KEY=your_groq_api_key_here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=groq/compound-mini
```

The system uses the Groq API for LLM-assisted structured extraction in Module 2 and for AI-generated candidate rationale in Module 11. All other modules operate without external API dependencies. Journal quality lookups in Module 4 use cached responses stored in `data/research_cache/` and will only make live requests on cache misses. Module 11 scoring is fully heuristic and does not require an LLM call; the LLM is used only to enrich the rationale text when available.

---

## Running the Pipeline

Full pipeline, all modules end to end:

```bash
python pipeline_orchestrator.py
```

Individual modules (each requires upstream outputs to be present):

```bash
python run_preprocessing.py          # Module 1: PDF splitting
python run_educational_profile.py    # Module 3
python run_research_profile.py       # Module 4
python run_supervision_analysis.py   # Module 5
python run_experience_analysis.py    # Module 6
python run_patent_analysis.py        # Module 7
python run_topic_analysis.py         # Module 8
python run_collaboration_analysis.py # Module 9
python pipeline_orchestrator.py      # Module 10: scoring and ranking
```

Module 11 standalone (requires Module 10 output to be present):

```bash
python run_jd_matching.py
```

Starting the API server:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080 --reload
```

The server starts on `http://127.0.0.1:8080` by default. Interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### GET /api/v1/health

Returns the operational status of the API server.

```json
{ "status": "ok", "version": "1.0.0" }
```

### GET /api/v1/candidates

Returns the ranked list of all evaluated candidates with composite scores, dimension scores, suitability tier, missing information count, and upload timestamp.

### GET /api/v1/candidates/{candidate_id}

Returns the full evaluation profile for a single candidate.

### GET /api/v1/email/{candidate_id}

Returns an AI-drafted follow-up email template for candidates with missing information flags.

### POST /api/v1/upload

Accepts a PDF file upload and queues it for processing through the full pipeline.

Content-Type: `multipart/form-data` — Form field: `file` (PDF)

### POST /api/v1/jd/upload

Uploads a new Job Description as plain text or PDF. Returns parsed metadata including extracted title, required degree, minimum experience, required skills, and a unique JD ID for subsequent evaluation calls.

### GET /api/v1/jd/list

Returns a summary list of all historically stored Job Descriptions with titles, creation timestamps, and candidate match counts.

### POST /api/v1/jd/{jd_id}/evaluate

Evaluates one or more candidates against the specified JD. Accepts either a comma-separated list of existing candidate IDs or a new CV PDF file upload. Returns ranked match results with scores, skill match details, fit tiers, and AI-generated rationale for each candidate.

### GET /api/v1/jd/{jd_id}/results

Returns stored match results for a previously evaluated JD without re-running the evaluation.

---

## Frontend Dashboard

The dashboard is a self-contained single-page application written in vanilla JavaScript and CSS with no build step or external framework dependency. It loads embedded candidate data by default and attempts to fetch live data from the API on startup. The API base URL is detected automatically from `window.location.origin`, making the dashboard work correctly in both local and cloud-hosted environments without code changes.

**Pages**

| Page | Description |
|---|---|
| Dashboard | Hero banner with live stats, KPI summary cards, pipeline stage bar, recent candidate table, score distribution chart, and top-ranked candidate highlights |
| AI Recommendations | Full ranked candidate list with real-time re-ranking via adjustable dimension weight sliders |
| Candidates | Searchable, sortable, filterable table of all candidates with score visualisations and a JD Match shortcut button per row |
| Compare | Side-by-side dimensional score comparison for any two or three selected candidates |
| Email Center | Personalised follow-up email generation for flagged candidates |
| JD Match | Job description upload, candidate selection, match evaluation, historical JD results, and new CV upload for instant JD screening |
| Upload CVs | Drag-and-drop PDF upload interface with processing queue |

**Smart Weighting**

The AI Recommendations page includes a weight control panel with seven range sliders corresponding to each scoring dimension. Adjusting any slider recomputes the weighted composite score and re-ranks all candidates in real time without modifying stored database values.

**JD Match Tab**

The JD Match page is a dedicated output tab for Module 11. It supports two workflows. In the existing-candidates workflow, a committee selects any subset of evaluated candidates and runs them against a loaded JD. In the new-CV workflow, a fresh PDF resume is uploaded and screened against the JD immediately, with an option to trigger the full 10-module evaluation afterwards. All JD sessions are stored and reloadable from a history dropdown.

**Backend Connectivity**

The API status indicator in the sidebar polls `/api/v1/health` every fifteen seconds and reflects the connection state with a live or offline dot indicator. When the API is reachable, the dashboard replaces embedded data with live API data transparently.

---

## Output Artefacts

All module outputs are written to `data/analysis/`.

| File | Description |
|---|---|
| `candidates.csv` | Master candidate index with composite scores and tier classifications |
| `edu_gaps.csv` | Educational credential gap flags per candidate |
| `research_aggregates.csv` | Aggregated publication and journal tier statistics |
| `supervision_profiles.csv` | Thesis supervision records with verification flags |
| `experience_profiles.csv` | Verified employment timelines with computed tenure |
| `research_breadth_profiles.csv` | Topic cluster assignments and breadth scores |
| `patent_aggregates.csv` | Patent records with granted and filed classification |
| `journal_profiles.csv` | Per-publication journal tier and impact metadata |
| `jd_matches/{jd_id}/meta.json` | Parsed JD metadata: title, required degree, min experience, skills |
| `jd_matches/{jd_id}/jd_parsed.json` | Full structured JD parse output |
| `jd_matches/{jd_id}/jd_original.txt` | Original JD text as submitted |
| `jd_matches/{jd_id}/results.csv` | Ranked candidate match results for this JD |

---

## Technology Stack

**Backend and Pipeline**

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| API framework | FastAPI 0.100+ |
| ASGI server | Uvicorn |
| PDF extraction | pdfplumber, pypdf |
| Data processing | pandas, openpyxl |
| LLM integration | Groq API via OpenAI-compatible client |
| String matching | RapidFuzz |
| ISBN resolution | isbnlib |
| Environment config | python-dotenv |

**Frontend**

| Component | Technology |
|---|---|
| Structure | HTML5 |
| Logic | Vanilla JavaScript (ES2020+) |
| Styling | Vanilla CSS with custom properties |
| Charts | Chart.js 4.4 |
| Typography | Inter, JetBrains Mono (Google Fonts) |
| Build tooling | None |

**Deployment**

| Component | Technology |
|---|---|
| Containerisation | Docker |
| Cloud platform | Railway |
| Version control | GitHub |

---

## Development Status

The following additions are planned for subsequent development cycles: WebSocket-based real-time pipeline progress streaming, persistent candidate storage using a relational database backend, export functionality for ranked reports in PDF and Excel formats, role-based access control for multi-user institutional deployment, and an email delivery integration for the Email Center page.

---

## License

This project was developed for internal use. Distribution and reproduction outside the commissioning institution require explicit written authorisation.
