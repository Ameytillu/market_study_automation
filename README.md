# Market Study Agent — Streamlit-only (Base)

This project is a Streamlit-only Python application for analyzing market datasets (CoStar Excel files).

Key points
- UI and execution run entirely inside Streamlit; there is no separate backend or API.
- Data is processed in-memory using pandas; no database is used.
- Upload one or more Excel files; the app will automatically inspect sheets, normalize tables to a canonical format, compute analytics, and render charts.

Run locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:
```bash
streamlit run ui/app.py
```

Notes on deployment
- This repository is designed to run entirely as a Streamlit app. If you deploy, pick a host that runs Python processes (Streamlit Community Cloud, Render, Railway, etc.). Netlify alone cannot run a Streamlit process — that would require a separate backend, which is out of scope.

Constraints
- No Flask/FastAPI or REST APIs are used.
- No databases or authentication.
- All logic executes in-process in Streamlit.
# Market Study AI Agent (Base)

This project is a base AI agent that ingests CoStar Excel files,
automatically analyzes their structure, and prepares them for
market study analytics with charts and commentary.

## How to Run
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Phase 1 Scope
- No PostgreSQL
- In-memory analytics
- Automated dataset understanding