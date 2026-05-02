"""FastAPI backend for the EHR Pipeline Frontend.

Endpoints:
  GET  /api/cases                      – list existing output cases
  GET  /api/cases/{case_id}/artifacts  – return all pipeline artifacts
  GET  /api/bench-cases                – list _bench cases available to run
  POST /api/run/{case_id}              – stream SSE pipeline progress
  GET  /                               – serve React app (production)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
BENCH_DIR = OUTPUTS_DIR / "_bench"
FRONTEND_DIST = ROOT / "frontend" / "dist"

# Stage output files in appearance order (used for progress polling)
STAGE_FILES: list[tuple[str, str]] = [
    ("stage1_evidence", "evidence_store.json"),
    ("stage2_extract", "claims.json"),
    ("stage3_verify", "verifications.json"),
    ("stage4_context", "context.json"),
    ("stage5_fact_sheet", "fact_sheet.json"),
    ("stage6_summarize", "summary.md"),
    ("stage7_check", "check_report.json"),
    ("stage8_review", "review.json"),
    ("stage9_patient_summary", "patient_summary.md"),
]

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

app = FastAPI(title="EHR Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stages_present(case_dir: Path) -> list[str]:
    present = []
    for stage_name, filename in STAGE_FILES:
        if (case_dir / filename).exists():
            present.append(stage_name)
    return present


def _read_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text("utf-8"))
        except Exception:
            return None
    return None


def _read_text(path: Path) -> str | None:
    if path.exists():
        try:
            return path.read_text("utf-8")
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/cases")
def list_cases():
    """Return all completed/partial output cases."""
    if not OUTPUTS_DIR.exists():
        return []
    cases = []
    for d in sorted(OUTPUTS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        stages = _stages_present(d)
        cases.append(
            {
                "case_id": d.name,
                "stages_present": stages,
                "has_summary": (d / "summary.md").exists(),
                "has_revised_summary": (d / "summary_revised.md").exists(),
                "has_audit": (d / "audit.json").exists(),
                "check_passed": _read_json(d / "check_report.json") is not None
                and (_read_json(d / "check_report.json") or {}).get("passed"),
            }
        )
    return cases


@app.get("/api/cases/{case_id}/artifacts")
def get_artifacts(case_id: str):
    """Return all pipeline artifacts for a case as a single JSON blob."""
    case_dir = OUTPUTS_DIR / case_id
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    return {
        "case_id": case_id,
        "note": _read_text(case_dir / "input_note.txt"),
        "evidence_store": _read_json(case_dir / "evidence_store.json"),
        "claims": _read_json(case_dir / "claims.json"),
        "verifications": _read_json(case_dir / "verifications.json"),
        "verifications_augmented": _read_json(case_dir / "verifications_augmented.json"),
        "context": _read_json(case_dir / "context.json"),
        "fact_sheet": _read_json(case_dir / "fact_sheet.json"),
        "summary_md": _read_text(case_dir / "summary.md"),
        "summary_revised_md": _read_text(case_dir / "summary_revised.md"),
        "summary_meta": _read_json(case_dir / "summary_meta.json"),
        "patient_summary_md": _read_text(case_dir / "patient_summary.md"),
        "patient_summary_meta": _read_json(case_dir / "patient_summary_meta.json"),
        "check_report": _read_json(case_dir / "check_report.json"),
        "check_report_revised": _read_json(case_dir / "check_report_revised.json"),
        "review": _read_json(case_dir / "review.json"),
        "audit": _read_json(case_dir / "audit.json"),
        "stages_present": _stages_present(case_dir),
    }


@app.get("/api/bench-cases")
def list_bench_cases():
    """Return benchmark cases that have a bundle.json and can be re-run."""
    if not BENCH_DIR.exists():
        return []
    cases = []
    for d in sorted(BENCH_DIR.iterdir()):
        if not d.is_dir():
            continue
        bundle = d / "bundle.json"
        notes = d / "notes"
        if bundle.exists():
            cases.append(
                {
                    "case_id": d.name,
                    "bundle_path": str(bundle),
                    "notes_dir": str(notes) if notes.exists() else None,
                    "already_run": (OUTPUTS_DIR / d.name / "summary.md").exists(),
                }
            )
    return cases


# Run-state registry so we don't double-run
_running: dict[str, bool] = {}
_run_lock = threading.Lock()


def _run_pipeline_thread(case_id: str, bundle_path: Path, notes_dir: Path | None) -> None:
    """Run the pipeline in a background thread."""
    try:
        from ehr_pipeline.pipeline import run_pipeline

        run_pipeline(
            case_id=case_id,
            bundle_path=bundle_path,
            notes_dir=notes_dir,
            resume=False,
            allow_violations=True,
            enable_context_agent=False,  # faster for demo; toggle if desired
        )
    except Exception as exc:
        log.error("Pipeline run failed for %s: %s", case_id, exc)
    finally:
        with _run_lock:
            _running.pop(case_id, None)


async def _sse_progress(case_id: str) -> AsyncGenerator[str, None]:
    """Yield SSE events as stage output files appear in the output directory."""
    case_dir = OUTPUTS_DIR / case_id
    seen: set[str] = set()

    # Remove stale outputs so we watch fresh writes
    for _, filename in STAGE_FILES:
        fp = case_dir / filename
        if fp.exists():
            fp.unlink()

    def _event(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    yield _event({"type": "started", "case_id": case_id})

    max_wait = 900  # 15 minute safety timeout
    elapsed = 0.0
    poll_interval = 1.0

    while elapsed < max_wait:
        # Check if the thread finished (no longer in _running)
        still_running = _running.get(case_id, False)

        for stage_name, filename in STAGE_FILES:
            if stage_name in seen:
                continue
            if (case_dir / filename).exists():
                seen.add(stage_name)
                yield _event({"type": "stage_done", "stage": stage_name, "file": filename})

        if len(seen) == len(STAGE_FILES) or not still_running:
            break

        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

    # Final event
    artifacts = get_artifacts(case_id)
    yield _event({"type": "done", "case_id": case_id, "stages_present": artifacts["stages_present"]})


@app.post("/api/run/{case_id}")
async def run_case(case_id: str):
    """Start (or resume) a pipeline run and stream SSE progress."""
    bench_entry = None
    for d in sorted(BENCH_DIR.iterdir()) if BENCH_DIR.exists() else []:
        if d.name == case_id and (d / "bundle.json").exists():
            bench_entry = d
            break

    if bench_entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No bench bundle found for case '{case_id}'. "
            "Only pre-materialized benchmark cases can be run from the UI.",
        )

    with _run_lock:
        if _running.get(case_id):
            raise HTTPException(status_code=409, detail=f"Case '{case_id}' is already running.")
        _running[case_id] = True

    bundle_path = bench_entry / "bundle.json"
    notes_dir = bench_entry / "notes"
    if not notes_dir.exists():
        notes_dir = None

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(case_id, bundle_path, notes_dir),
        daemon=True,
    )
    thread.start()

    return StreamingResponse(
        _sse_progress(case_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Static file serving (production build)
# ---------------------------------------------------------------------------

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        raise HTTPException(status_code=404, detail="Frontend not built. Run: cd frontend && npm run build")
