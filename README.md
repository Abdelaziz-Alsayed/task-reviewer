# Task Reviewer

An AI-assisted assignment evaluation workbench for reviewing and scoring student submissions using deterministic checks, evidence extraction, LLM-assisted evaluation, and human review.

The project was built around a **Week 05 Regression** assignment based on the Kaggle House Prices dataset.

---

## Table of Contents

- [Overview](#overview)
- [Evaluation Workflow](#evaluation-workflow)
- [Main Features](#main-features)
  - [Submission Extraction](#submission-extraction)
  - [Rubric-Based Evaluation](#rubric-based-evaluation)
  - [Deterministic Checks](#deterministic-checks)
  - [Evidence-Based LLM Evaluation](#evidence-based-llm-evaluation)
  - [LLM Evaluation Providers](#llm-evaluation-providers)
  - [Structured Output & Validation](#structured-output--validation)
  - [Human Review](#human-review)
  - [Database & History](#database--history)
  - [CSV Export](#csv-export)
- [Streamlit Interface](#streamlit-interface)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Clone & Setup](#clone--setup)
  - [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Design Principles](#design-principles)
- [Current Limitations](#current-limitations)
- [Future Extensions](#future-extensions)
- [Project Status](#project-status)
- [License](#license)
- [Author](#author)

---

## Overview

**Task Reviewer** is a local Streamlit application designed to reduce the manual effort required to review programming and data science assignments.

Instead of sending an entire student submission blindly to an LLM, the system follows an evidence-based evaluation pipeline. The LLM acts strictly as an **evaluation assistant**, not the final grading authority—the instructor remains responsible for reviewing and approving the final score.

---

## Evaluation Workflow

```text
Student Submission
        ↓
Submission Extraction
        ↓
Deterministic Analysis
        ↓
Evidence Selection
        ↓
LLM Evaluation
        ↓
Criterion-Level Suggestions
        ↓
AI Suggested Score
        ↓
Human Review
        ↓
Final Score
        ↓
History / Export
```

---

## Main Features

### Submission Extraction
Extracts content from heterogeneous student submission files into a unified format for downstream analysis:
- **Python Scripts** (`.py`)
- **Jupyter Notebooks** (`.ipynb`)
- **PDF Documents** (`.pdf`)
- **CSV Data Files** (`.csv`)

### Rubric-Based Evaluation
Evaluation criteria are completely decoupled from application logic and configured via `config/rubric.yaml`. 

The default **Week 05 Regression** rubric includes eight criteria:

| Criterion | Maximum Score |
| :--- | :---: |
| **Data Understanding** | 3 |
| **Data Preparation** | 3 |
| **Regression Models** | 4 |
| **Evaluation** | 4 |
| **Cross-Validation** | 3 |
| **RandomizedSearchCV** | 4 |
| **Visualization** | 2 |
| **Final Analysis** | 2 |
| **Total** | **25** |

### Deterministic Checks
Performs rule-based static analysis and heuristic checks prior to invoking the LLM, including:
- **Data Pipeline:** Loading, exploration, target identification, $X/y$ split, target analysis, missing value handling, categorical encoding, preprocessing leakage prevention.
- **Model Training:** Baseline model, linear regression, flexible regression, predictions.
- **Metrics:** MAE, RMSE, $R^2$, and metric interpretation.
- **Validation & Tuning:** Cross-validation implementation, mean/standard deviation reporting, `RandomizedSearchCV` usage, search space sizing, best parameter extraction.
- **Diagnostics:** Actual vs. predicted plots, residual plots, model comparison, and justification.

### Evidence-Based LLM Evaluation
Instead of passing raw source files to an LLM, Task Reviewer constructs an isolated **Evidence Package**:

```text
Submission
    ↓
Extracted Content
    ↓
Keyword / Criterion Matching
    ↓
Relevant Evidence
    ↓
Rubric + Evidence
    ↓
LLM
```

This reduces token usage, eliminates irrelevant noise, and keeps reasoning transparent and verifiable.

### LLM Evaluation Providers
Supports decoupled provider backends configured via `llm/factory.py`:
- **Mock Evaluator:** Completely local; requires no API key. Ideal for CI/CD, development, pipeline validation, and UI testing.
- **Gemini Evaluator:** Uses the Google Gemini API. Executes a single unified evaluation request per submission to minimize API quota usage.

### Structured Output & Validation
Responses are strictly validated against **Pydantic** schemas:

```json
{
  "criterion_id": "randomized_search",
  "criterion_name": "RandomizedSearchCV",
  "score": 4,
  "max_score": 4,
  "evidence": [
    "RandomizedSearchCV is used with a defined parameter distribution."
  ],
  "missing_requirements": [],
  "reasoning": "The supplied evidence supports the criterion requirements.",
  "confidence": 0.92
}
```

The scoring engine enforces:
- All rubric criteria are present exactly once without unknown IDs.
- Criterion scores stay bounded ($0 \le \text{Score} \le \text{Max Score}$).
- **The application—not the LLM—calculates the cumulative total score.**

### Human Review
The AI score is treated purely as a baseline suggestion. Instructors can:
- Inspect granular evidence, identified gaps, confidence scores, and reasoning.
- Override individual criterion scores.
- Add feedback notes and finalize the authoritative grade.

```text
AI Suggested Score ──> Human Review ──> Final Score
```

### Database & History
Persistent local storage powered by **SQLite**:
- **Students & Evaluations:** Tracks student profiles, submission runs, statuses, timestamps, suggested scores, and finalized scores.
- **Criterion Records:** Stores granular criteria breakdowns, feedback, confidence metrics, and raw reasoning.
- **Audit History:** Full retrospective view of past grading runs with schema migration support.

### CSV Export
One-click export utility that compiles historical evaluation summaries (student metadata, suggested vs. final scores, timestamps) into standardized `.csv` reports.

---

## Streamlit Interface

The web interface is organized into five functional views:

| View | Purpose |
| :--- | :--- |
| **Dashboard** | High-level metrics: total submissions, graded counts, pending queue, average score, and status table. |
| **Submissions** | Discovered local submissions, evaluator selection (Mock vs. Gemini), and evaluation execution. |
| **Evaluation** | Human-in-the-loop review workbench for criterion-by-criterion inspection and score overrides. |
| **History** | Historical submission logs and drill-down review of previous evaluation records. |
| **Export** | Generates and downloads evaluation summary reports as CSV files. |

---

## System Architecture

The application adopts a modular, layered architecture:

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                │
└───────────┬────────────────────────────────┬───────────┘
            │                                │
┌───────────▼───────────┐        ┌───────────▼───────────┐
│ Ingestion & Extraction │        │   Database & Export   │
│  (local.py, extractor) │        │ (db.py, reports.py)   │
└───────────┬───────────┘        └───────────▲───────────┘
            │                                │
┌───────────▼────────────────────────────────┴───────────┐
│                  Evaluation Engine                     │
│  - deterministic.py (Static analysis)                 │
│  - evidence_selector.py (Relevance filtering)          │
│  - scoring.py (Validation & bounds checking)           │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│                    LLM Provider Layer                  │
│       factory.py  ──>  [ MockProvider | GeminiProvider ]│
└────────────────────────────────────────────────────────┘
```

1. **Configuration (`config/`):** Manages externalized YAML rubrics and environment configurations.
2. **Ingestion (`ingestion/`):** Scans and discovers candidate submission files in local directories.
3. **Extraction (`extraction/`):** Normalizes multiple formats (`.py`, `.ipynb`, `.pdf`, `.csv`) into plain text payloads.
4. **Deterministic Evaluation (`evaluation/deterministic.py`):** Runs static heuristic checks.
5. **Evidence Selection (`evaluation/evidence*.py`):** Filters extracted text into targeted context snippets.
6. **LLM Provider Layer (`llm/`):** Provider-agnostic client managing structured prompts and Pydantic validation.
7. **Scoring Engine (`evaluation/scoring.py`):** Enforces rubric bounds and computes total scores.
8. **Persistence (`database/`):** Manages SQLite connections, migrations, and queries.
9. **Export (`export/`):** Serializes database records to CSV.

---

## Project Structure

```text
task-reviewer/
│
├── app.py                      # Main Streamlit web application entry point
│
├── config/                     # Configuration and rubric specifications
│   ├── __init__.py
│   ├── rubric.py
│   ├── rubric.yaml
│   └── settings.py
│
├── database/                   # SQLite persistence and schema handling
│   ├── __init__.py
│   └── db.py
│
├── evaluation/                 # Core scoring and evaluation orchestration
│   ├── __init__.py
│   ├── deterministic.py
│   ├── evidence.py
│   ├── evidence_selector.py
│   ├── history.py
│   ├── pipeline.py
│   ├── runner.py
│   └── scoring.py
│
├── export/                     # Export utilities (CSV generation)
│   ├── __init__.py
│   └── reports.py
│
├── extraction/                 # File parsers (.py, .ipynb, .pdf, .csv)
│   ├── __init__.py
│   └── extractor.py
│
├── ingestion/                  # Local directory submission discovery
│   ├── __init__.py
│   └── local.py
│
├── llm/                        # Evaluator abstractions, schemas, and clients
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── gemini.py
│   ├── mock.py
│   ├── prompts.py
│   └── schemas.py
│
├── scripts/                    # Maintenance and diagnostic scripts
│   ├── __init__.py
│   └── gemini_smoke_test.py
│
├── tests/                      # Automated test suite (Pytest)
│   ├── test_checks.py
│   ├── test_database.py
│   ├── test_evidence.py
│   ├── test_evidence_package.py
│   ├── test_evidence_selector.py
│   ├── test_extraction.py
│   ├── test_gemini.py
│   ├── test_history.py
│   ├── test_human_review.py
│   ├── test_llm_factory.py
│   ├── test_llm_pipeline.py
│   ├── test_pipeline.py
│   ├── test_runner.py
│   └── test_scoring.py
│
├── data/                       # Local data directories
│   ├── submissions/
│   └── exports/
│
├── requirements.txt            # Project dependencies
├── README.md                   # Repository documentation
└── .gitignore                  # Git ignore rules
```

---

## Installation & Setup

### Prerequisites
- **Python:** 3.10 or higher
- **Git**
- Tested on **Ubuntu / WSL2** and **macOS / Linux** environments.

### Clone & Setup
```bash
# Clone the repository
git clone https://github.com/Abdelaziz-Alsayed/task-reviewer.git
cd task-reviewer

# Create and activate a conda environment (recommended)
conda create -n task-reviewer python=3.10 -y
conda activate task-reviewer

# Install dependencies
pip install -r requirements.txt
```

### Configuration
By default, the application runs out of the box using the **Mock Evaluator**.

To enable Gemini-powered evaluations, configure a `.env` file in the root directory:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```
*(Ensure `.env` remains ignored by version control).*

---

## Running the Application

Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Access the application locally at:
```text
http://localhost:8501
```

> **Running without an API key:** On the **Submissions** page, simply set the evaluator provider to **Mock** to validate the complete workflow offline.

---

## Running Tests

### Automated Test Suite
Run the full test suite via `pytest`:
```bash
python -m pytest -q
```
*Current benchmark: **43 passed**.*

### Gemini Smoke Test
Verify API connectivity, schema compliance, and end-to-end evaluation flow:
```bash
python scripts/gemini_smoke_test.py
```
*(Requires a valid `GEMINI_API_KEY`).*

---

## Design Principles

1. **Evidence Before LLM:** The model evaluates structured, targeted evidence snippets rather than an unconstrained source repository.
2. **Deterministic Checks Where Possible:** Rules that can be tested deterministically are resolved statically without non-deterministic model calls.
3. **LLM as an Assistant:** The model drafts criterion suggestions; it never possesses autonomous grading authority.
4. **Application-Owned Scoring:** The application layer validates score boundaries and computes aggregate scores.
5. **Human-in-the-Loop:** All automated evaluations require instructor review, adjustment, and sign-off.
6. **Provider Abstraction:** The evaluator layer supports interchangeable backends (Mock, Gemini, etc.) through a unified interface.

---

## Current Limitations

Designed strictly as a local evaluation workbench:
- **Storage & Ingestion:** Local submission discovery only (no cloud storage or Google Drive sync).
- **Security & Multi-Tenancy:** Single-user workbench without role-based access control (RBAC).
- **Execution:** Synchronous execution without distributed background queues (e.g., Celery/Redis).
- **Parsing:** Keyword/heuristic evidence filtering rather than deep semantic/AST embedding retrieval.
- **Domain Scope:** Tuned specifically for the Week 05 Regression rubric.

---

## Future Extensions

- Google Drive & Canvas/Blackboard LMS integrations.
- Multi-user authentication and instructor collaboration.
- AST-based code analysis and semantic vector-indexed evidence retrieval.
- Multi-assignment rubric configurator UI.
- Batch asynchronous evaluation queues.
- Inter-rater reliability (human vs. AI agreement analytics).

---

## Project Status

- [x] Project architecture & configuration
- [x] Modular assignment rubric (`rubric.yaml`)
- [x] Multi-format extraction (`.py`, `.ipynb`, `.pdf`, `.csv`)
- [x] Deterministic heuristic checks
- [x] Evidence selection and packaging pipeline
- [x] Provider abstractions (Mock & Gemini)
- [x] Pydantic schema validation & scoring boundaries
- [x] SQLite persistence & schema migration handling
- [x] Human-in-the-loop review workflow
- [x] Historical review viewer & CSV report generation
- [x] Comprehensive test suite (43 unit/integration tests)
- [x] Gemini smoke test validation

---

## License

This project is open-source and distributed under the [MIT License](LICENSE).

---

## Author

**Abdelaziz Alsayed Mohamed**  
- **GitHub:** [@Abdelaziz-Alsayed](https://github.com/Abdelaziz-Alsayed)  
- **LinkedIn:** [Abdelaziz Alsayed Mohamed](https://www.linkedin.com/in/abdelaziz-alsayed/)