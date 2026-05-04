# EHR Summarization Pipeline

A multi-stage, LLM-backed clinical documentation pipeline that ingests **FHIR R4 Bundles** and free-text clinical notes, producing verified summaries and patient-facing documents. Includes a **FastAPI** backend with SSE streaming and a **React + Vite** reviewer UI.

## Pipeline Stages

| # | Stage | Output |
| --- | --- | --- |
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

```text
├── ehr_pipeline/              # Core Python package
│   ├── pipeline.py            # Orchestration & resume logic
│   ├── cli.py                 # CLI entry point
│   ├── config.py              # Model assignments, thresholds, env loading
│   ├── extraction.py          # FHIR resource extraction (stages 1-2)
│   ├── verification.py        # Claim verification orchestration (stages 3-4)
│   ├── evidence_store.py      # Evidence assembly & candidate retrieval
│   ├── ollama_client.py       # Ollama API client with corrective retry
│   ├── prompts.py             # All LLM prompt templates (S2–S4, S6, S8, S9)
│   ├── schemas.py             # Pydantic models for all stage I/O
│   ├── runtime.py             # Stage timing & cache-freshness helpers
│   └── stages/                # s1_evidence … s9_patient_summary
├── frontend/                  # React + Vite reviewer UI
│   └── src/components/        # CaseList, PipelineStepper, ArtifactTabs,
│                              #   EvidenceDrawer, FactSheet, SummaryViewer
├── server.py                  # FastAPI backend (REST + SSE)
├── benchmarks/                # ROUGE, BERTScore, entity F1 metrics; MIMIC adapters
├── benchmark.ipynb            # End-to-end MIMIC-III pipeline benchmark
├── baseline_benchmark.ipynb   # Single-shot Anthropic baseline benchmark
├── single_note_inference.ipynb # Interactive single-note inference notebook
├── datasets/                  # PDSQI-9 research materials, MIMIC-III-Ext-Notes
├── outputs/                   # Runtime artifacts & benchmark bundles
├── infer_ollama.py            # Standalone Ollama inference CLI
├── run_sample.sh              # Quick-start shell script for a sample case
├── requirements.txt           # Server dependencies
└── requirements-bench.txt     # Benchmark dependencies
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

```dotenv
OLLAMA_HOST=http://localhost:11434   # or https://ollama.com for cloud
OLLAMA_API_KEY=your-api-key         # required for cloud
ANTHROPIC_API_KEY=your-api-key      # optional; only needed for baseline_benchmark.ipynb
```

Default model assignments are defined in `ehr_pipeline/config.py` and can be updated there:

| Role | Default Model |
| --- | --- |
| Claim Extraction | `gemma4:31b` |
| Claim Verification | `gemma4:31b` |
| Context Agent | `gemma4:31b` |
| Summary Generation | `gemma4:31b` |
| Final Review | `gemma4:31b` |

### Running the Pipeline (CLI)

```bash
python -m ehr_pipeline.cli \
    --bundle data/sample_bundle.json \
    --notes data/sample_notes \
    --case-id sample
```

Key flags:

| Flag | Description |
| --- | --- |
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

Open `http://localhost:8000` in your browser. The UI is a case reviewer built around the pipeline output structure:

- **Sidebar** — lists all completed cases from `outputs/`. Select a case to load its artifacts, or click **Run** to kick off the pipeline for a pre-materialized benchmark case directly from the UI (progress streams in real time via SSE).
- **Pipeline Stepper** — a progress bar in the top bar that lights up each stage (S1–S9) as it completes during a live run, or shows which stages are already present for a cached case.
- **Artifact Tabs** — tabbed view of everything the pipeline wrote: the input note, evidence store, claims, verifications, fact sheet, clinician summary, check report, review, and patient-facing summary. Markdown artifacts (summaries) are rendered as formatted text.
- **Evidence Drawer** — click any inline citation (`[E:cond:3]`) in a summary to open a side drawer showing the underlying evidence item from the structured EHR (display name, value, unit, date, FHIR resource type).

**API Endpoints:**

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/cases` | List existing output cases |
| `GET` | `/api/cases/{case_id}/artifacts` | Fetch all pipeline artifacts for a case |
| `GET` | `/api/bench-cases` | List available benchmark cases |
| `POST` | `/api/run/{case_id}` | Stream pipeline progress via SSE |

## Benchmarking

The `benchmark.ipynb` notebook runs the pipeline against **MIMIC-III-Ext-Notes** and evaluates outputs using:

- **ROUGE** (1/2/L)
- **BERTScore**
- **Entity-level F1**
- **Compression ratio**

Pipeline summaries keep their inline evidence citations (for example, `[E:cond:3]`)
in the generated Markdown. For benchmark scoring, the notebook strips those
citations downstream before computing ROUGE, BERTScore, entity metrics, and
compression ratio. This lets existing generated outputs be recovered and
rescored without rerunning the pipeline.

The `baseline_benchmark.ipynb` notebook runs a single-shot Anthropic baseline
(`claude-opus-4-7`) on the same cases and writes results under
`outputs/_baseline/`.

Saved benchmark results in this checkout compare 100 matched MIMIC cases. Mean
metrics are:

- **Pipeline (`outputs/_bench/benchmark_metrics.csv`)**: ROUGE-1 F1 0.290,
  ROUGE-2 F1 0.127, ROUGE-L F1 0.187, BERTScore F1 0.813, entity recall
  0.377, entity F1 0.523, compression ratio 0.264, with 76/100 summaries in
  the 0.20-0.30 compression band.
- **Anthropic single-shot baseline (`outputs/_baseline/baseline_metrics.csv`)**:
  ROUGE-1 F1 0.130, ROUGE-2 F1 0.020, ROUGE-L F1 0.071, BERTScore F1 0.793,
  entity recall 0.151, entity F1 0.254, compression ratio 0.426, with 5/100
  summaries in the 0.20-0.30 compression band.
- **Mean paired delta (pipeline - Anthropic)**: +0.160 ROUGE-1 F1, +0.106
  ROUGE-2 F1, +0.116 ROUGE-L F1, +0.020 BERTScore F1, +0.226 entity recall,
  +0.269 entity F1, and -0.163 compression ratio.

Install benchmark dependencies first:

```bash
pip install -r requirements-bench.txt
# Only needed for the Anthropic baseline notebook if not already installed:
pip install anthropic python-dotenv
```

## Datasets

- **`datasets/pdsqi-9-main/`** — [PDSQI-9](datasets/pdsqi-9-main/README.md) (Provider Documentation Summarization Quality Instrument): LLM-as-judge notebooks, summary generation scripts, study data, and R/Rmd analyses.
- **`datasets/mimic-iii-ext-notes-1.0.0/`** — MIMIC-III Extended Notes schema and labels. The full `notes.csv` must be obtained separately per the dataset license.

## License

See individual dataset directories for their respective licenses.
