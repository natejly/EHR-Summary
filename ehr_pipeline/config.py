"""Central configuration for the EHR summarization pipeline.

All model assignments live here so they can be swapped in one place.
Environment variables are loaded from a .env file 
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

LOCAL_HOST = "http://localhost:11434"
CLOUD_HOST = "https://ollama.com"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class ModelAssignments:
    """Models per pipeline stage"""

    claim_extraction: str = "gemma3:27b"
    claim_verification: str = "gemma3:27b"
    context_agent: str = "gemma3:27b"
    summary_generation: str = "gemma4:31b"
    final_review: str = "gemma3:27b"


MODELS = ModelAssignments()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", CLOUD_HOST)
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

VERIFY_DATE_WINDOW_DAYS = 365
VERIFY_TOPK_CANDIDATES = 8
VERIFY_MAX_CONCURRENCY = 4

LAB_OBSERVATION_WINDOW_DAYS = 180

SUMMARY_TARGET_COMPRESSION_MIN = 0.20
SUMMARY_TARGET_COMPRESSION_MAX = 0.30

# Stage 9: patient-facing summary settings
PATIENT_SUMMARY_GRADE_MIN = 7.0          # Flesch-Kincaid grade target (lower)
PATIENT_SUMMARY_GRADE_MAX = 8.0          # Flesch-Kincaid grade target (upper)
PATIENT_SUMMARY_MIN_CHARS = 600
PATIENT_SUMMARY_MAX_CHARS = 1800

DEFAULT_TEMPERATURE = 0.1
REQUEST_TIMEOUT_SECONDS = 600

# Retry policy for LLM calls (network errors, 5xx, bad JSON).
# Set LLM_MAX_RETRIES=0 to disable retries entirely.
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))  # seconds


def output_dir(case_id: str) -> Path:
    """Per-case output directory; creates it on demand."""
    path = OUTPUTS_DIR / case_id
    path.mkdir(parents=True, exist_ok=True)
    return path
