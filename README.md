# Task Reviewer

Local AI-assisted assignment evaluation workbench. Phase 1 works entirely locally: no Google Drive and no LLM.

## Phase 1

- Local submission ingestion
- `.py`, `.ipynb`, `.pdf`, `.csv` extraction
- Week 05 Regression rubric (25 marks)
- Deterministic checks for:
  - RandomizedSearchCV
  - MAE
  - RMSE
  - R²
  - cross-validation
  - potential preprocessing leakage
- SQLite result storage
- Streamlit dashboard

## Setup

```bash
conda create -n task-reviewer python=3.11 -y
conda activate task-reviewer
pip install -r requirements.txt
streamlit run app.py
```

The app defaults to the local `data/submissions/` directory.

## Submission layout

Each direct child directory is treated as one student submission:

```text
data/submissions/
├── Ahmed_Ali/
│   ├── solution.py
│   ├── report.pdf
│   └── predictions.csv
└── Sara_Hassan/
    └── solution.ipynb
```

The student directory name is used as the student name.
