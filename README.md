# TALASH

**Talent Acquisition and Longitudinal Academic Scoring for Higher Education**

TALASH is an automated faculty candidate evaluation system developed for the Higher Education Commission of Pakistan. It ingests PDF curricula vitae, parses and normalises structured data across nine domain modules, computes a composite suitability score across seven academic dimensions, and exposes results through a REST API and an interactive single-page dashboard.

The system is designed to eliminate manual screening bottlenecks in large-scale faculty recruitment cycles, where evaluation committees are required to assess dozens to hundreds of CVs against a standardised rubric of academic merit.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Pipeline](#module-pipeline)
3. [Scoring Dimensions](#scoring-dimensions)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)
8. [API Reference](#api-reference)
9. [Frontend Dashboard](#frontend-dashboard)
10. [Output Artefacts](#output-artefacts)
11. [Technology Stack](#technology-stack)
12. [Development Status](#development-status)

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
data/analysis/   <-- CSVs, XLSX, text reports
  |
  v
FastAPI REST API  <-- /api/v1/candidates, /api/v1/email/{id}, /api/v1/upload
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
|-- analysis/                   Domain analysis modules (M3-M10)
|   |-- educational_profile.py
|   |-- research_profile.py
|   |-- supervision_analysis.py
|   |-- experience_analysis.py
|   |-- patent_analysis.py
|   |-- topic_analysis.py
|   |-- collaboration_analysis.py
|   |-- composite_scorer.py
|
|-- api/                        FastAPI application
|   |-- main.py                 Application entry point and static mounts
|   |-- routes.py               API route handlers
|   |-- schemas.py              Pydantic request/response models
|
|-- data/
|   |-- cvs/                    Raw and split PDF files
|   |-- extracted/              Module 2 structured CSV outputs
|   |-- analysis/               Module 3-10 analysis outputs (CSV, XLSX, TXT)
|   |-- rankings/               Reference data (universities, journal lists)
|   |-- research_cache/         Cached external API responses
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
|-- run_backend.py              Uvicorn server launcher
|-- run_batch.py                Scoring-only batch runner
|-- run_preprocessing.py        Module 1 runner
|-- run_*.py                    Individual module runners (M3-M9)
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

Install core pipeline dependencies:

```bash
pip install -r requirements.txt
```

Install API server dependencies:

```bash
pip install fastapi uvicorn python-multipart
```

---

## Configuration

Copy `.env.example` to `.env` and populate the required values.

```
GROQ_API_KEY=your_groq_api_key_here
```

The system uses the Groq API for LLM-assisted structured extraction in Module 2. All other modules operate without external API dependencies. Journal quality lookups in Module 4 use cached responses stored in `data/research_cache/` and will only make live requests on cache misses.

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

Starting the API server:

```bash
python run_backend.py
```

The server starts on `http://127.0.0.1:8000` by default. Interactive API documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### GET /api/v1/health

Returns the operational status of the API server.

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### GET /api/v1/candidates

Returns the ranked list of all evaluated candidates with composite scores, dimension scores, suitability tier, missing information count, and upload timestamp.

```json
[
  {
    "candidate_id": "04_MUHAMMAD_FARRUKH",
    "candidate_name": "Muhammad Farrukh Qureshi",
    "overall_composite_score": 53.0,
    "candidate_tier": "Tier 3: Moderate Candidate",
    "education_score": 10,
    "research_score": 5,
    "experience_score": 15,
    "supervision_score": 2,
    "innovation_score": 2,
    "breadth_score": 3,
    "collaboration_score": 10,
    "status": "done",
    "missing_info_count": 2,
    "uploaded_at": "2026-07-19T10:00:00"
  }
]
```

### GET /api/v1/candidates/{candidate_id}

Returns the full evaluation profile for a single candidate.

### GET /api/v1/email/{candidate_id}

Returns an AI-drafted follow-up email template for candidates with missing information flags.

```json
{
  "candidate_id": "04_MUHAMMAD_FARRUKH",
  "subject": "Faculty Position Application — Additional Information Required",
  "body": "..."
}
```

### POST /api/v1/upload

Accepts a PDF file upload and returns a processing acknowledgement. The uploaded file is queued for processing through the full pipeline.

Content-Type: `multipart/form-data`  
Form field: `file` (PDF)

---

## Frontend Dashboard

The dashboard is a self-contained single-page application written in vanilla JavaScript and CSS with no build step or external framework dependency. It loads embedded candidate data by default and attempts to fetch live data from the API on startup.

When the backend is running, the dashboard is served at `http://127.0.0.1:8000`. For offline use, open `frontend/index.html` directly in any modern browser.

**Pages**

| Page | Description |
|---|---|
| Dashboard | KPI summary cards, pipeline stage bar, recent candidate table, score distribution chart, top-ranked candidate highlights |
| AI Recommendations | Full ranked candidate list with real-time re-ranking via adjustable dimension weight sliders |
| Candidates | Searchable, sortable, filterable table of all candidates with score visualisations |
| Compare | Side-by-side dimensional score comparison for any two selected candidates |
| Email Center | Personalised follow-up email generation for flagged candidates |
| Upload CVs | Drag-and-drop PDF upload interface with processing queue |

**Smart Weighting**

The AI Recommendations page includes a weight control panel with seven range sliders corresponding to each scoring dimension. Adjusting any slider recomputes the weighted composite score and re-ranks all candidates in real time without modifying stored database values. This allows evaluation committees to explore alternative prioritisation scenarios interactively.

**Backend Connectivity**

The API status indicator in the sidebar polls `http://127.0.0.1:8000/api/v1/health` every fifteen seconds and reflects the connection state. When the API is reachable, the dashboard replaces embedded data with live API data transparently.

---

## Output Artefacts

All module outputs are written to `data/analysis/`.

| File | Description |
|---|---|
| `candidates.csv` | Master candidate index with composite scores and tier classifications |
| `edu_gaps.csv` | Educational credential gap flags per candidate |
| `research_aggregates.csv` | Aggregated publication and journal tier statistics |
| `research_aggregates.xlsx` | Excel version with formatted tables |
| `supervision_profiles.csv` | Thesis supervision records with verification flags |
| `supervision_profiles.xlsx` | Excel version |
| `experience_profiles.csv` | Verified employment timelines with computed tenure |
| `experience_profiles.xlsx` | Excel version |
| `research_breadth_profiles.csv` | Topic cluster assignments and breadth scores |
| `patent_aggregates.csv` | Patent records with granted and filed classification |
| `journal_profiles.csv` | Per-publication journal tier and impact metadata |
| `edu_report.txt` | Human-readable education analysis narrative |
| `research_report.txt` | Human-readable research analysis narrative |
| `supervision_report.txt` | Human-readable supervision analysis narrative |
| `experience_report.txt` | Human-readable experience analysis narrative |

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

---

## Development Status

The system has completed its initial evaluation cycle against a cohort of 43 faculty applicants. All ten pipeline modules are operational. The REST API exposes candidate data, email generation, and file upload endpoints. The frontend dashboard is functional in both offline (embedded data) and online (live API) modes.

Planned additions for subsequent development cycles include WebSocket-based real-time pipeline progress streaming, persistent candidate storage using a relational database backend, export functionality for ranked reports in PDF and Excel formats, and role-based access control for multi-user institutional deployment.

---

## License

This project was developed for internal use by the Higher Education Commission of Pakistan. Distribution and reproduction outside the commissioning institution require explicit written authorisation.

---

*TALASH is maintained by the TALASH Development Team. For queries related to deployment or integration, raise an issue in this repository.*
