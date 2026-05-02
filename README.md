# EHR Summarization Pipeline

A multi-stage, LLM-backed clinical documentation pipeline that ingests **FHIR R4 Bundles** and free-text clinical notes, producing verified summaries and patient-facing documents. Includes a **FastAPI** backend with SSE streaming and a **React + Vite** reviewer UI.

## Pipeline Stages

| # | Stage | Output |
|---|-------|--------|
| 1 | **Evidence Assembly** | `evidence_store.json` |
| 2 | **Claim Extraction** | `claims.json` |
| 3 | **Claim Verification** | `verifications.json` |
| 4 | **Context Agent** (optional) | `context.json` |
| 5 | **Fact Sheet** | `fact_sheet.json` |
| 6 | **Summary Generation** | `summary.md` |
| 7 | **Deterministic Check** | `check_report.json` |
| 8 | **Clinical Review** | `review.json` |
| 9 | **Patient-Facing Summary** | `patient_summary.md` |

Each stage writes its artifact to `outputs/<case_id>/`. Runs are resumable — stages whose outputs are already newer than their inputs can be skipped with `--resume`.

## Project Structure

```
├── ehr_pipeline/          # Core Python package
│   ├── pipeline.py        # Orchestration & resume logic
│   ├── cli.py             # CLI entry point
│   ├── config.py          # Model assignments, thresholds, env loading
│   ├── extraction.py      # FHIR resource extraction
│   ├── verification.py    # Claim verification helpers
│   ├── evidence_store.py  # Evidence assembly
│   ├── ollama_client.py   # Ollama API client
│   ├── prompts.py         # Prompt templates
│   ├── schemas.py         # Pydantic models
│   └── stages/            # s1_evidence … s9_patient_summary
├── frontend/              # React + Vite reviewer UI
├── server.py              # FastAPI backend (REST + SSE)
├── benchmarks/            # ROUGE, BERTScore, entity F1 metrics; MIMIC adapters
├── benchmark.ipynb        # End-to-end MIMIC-III benchmark notebook
├── datasets/              # PDSQI-9 research materials, MIMIC-III-Ext-Notes
├── outputs/               # Runtime artifacts & benchmark bundles
├── infer_ollama.py        # Standalone Ollama inference CLI
├── requirements.txt       # Server dependencies
└── requirements-bench.txt # Benchmark dependencies
```

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for the frontend)
- An **Ollama** instance (local or cloud) — see [ollama.com](https://ollama.com)

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-org>/LLMS.git
cd LLMS

# Install Python dependencies
pip install -r requirements.txt

# (Optional) Install benchmark dependencies
pip install -r requirements-bench.txt

# Build the frontend
cd frontend
npm install
npm run build
cd ..
```

### Configuration

Create a `.env` file in the project root:

```
OLLAMA_HOST=http://localhost:11434   # or https://ollama.com for cloud
OLLAMA_API_KEY=your-api-key         # required for cloud
```

Default model assignments are defined in `ehr_pipeline/config.py` and can be updated there:

| Role | Default Model |
|------|---------------|
| Claim Extraction | `gemma3:27b` |
| Claim Verification | `gemma3:27b` |
| Context Agent | `gemma3:27b` |
| Summary Generation | `gemma4:31b` |
| Final Review | `gemma3:27b` |

### Running the Pipeline (CLI)

```bash
python -m ehr_pipeline.cli \
    --bundle data/sample_bundle.json \
    --notes data/sample_notes \
    --case-id sample
```

Key flags:

| Flag | Description |
|------|-------------|
| `--resume` | Skip stages whose outputs are up to date |
| `--no-context-agent` | Skip stage 4 (useful for benchmarking) |
| `--allow-violations` | Emit summary even if the deterministic check fails |
| `--summary-min-ratio` | Minimum summary compression ratio (default 0.20) |
| `--summary-max-ratio` | Maximum summary compression ratio (default 0.30) |

### Running the Web UI

```bash
# Start the backend (serves the built frontend in production)
uvicorn server:app --reload

# Or for frontend development with hot reload
cd frontend && npm run dev
```

**API Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/cases` | List existing output cases |
| `GET` | `/api/cases/{case_id}/artifacts` | Fetch all pipeline artifacts for a case |
| `GET` | `/api/bench-cases` | List available benchmark cases |
| `POST` | `/api/run/{case_id}` | Stream pipeline progress via SSE |

## Benchmarking

The `benchmark.ipynb` notebook runs the pipeline against **MIMIC-III-Ext-Notes** and evaluates outputs using:

- **ROUGE** (1/2/L)
- **BERTScore**
- **Entity-level F1**

Install benchmark dependencies first:

```bash
pip install -r requirements-bench.txt
```

## Datasets

- **`datasets/pdsqi-9-main/`** — [PDSQI-9](datasets/pdsqi-9-main/README.md) (Provider Documentation Summarization Quality Instrument): LLM-as-judge notebooks, summary generation scripts, study data, and R/Rmd analyses.
- **`datasets/mimic-iii-ext-notes-1.0.0/`** — MIMIC-III Extended Notes schema and labels. The full `notes.csv` must be obtained separately per the dataset license.

## License

See individual dataset directories for their respective licenses.
