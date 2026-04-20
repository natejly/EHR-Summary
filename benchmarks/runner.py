"""Disk-cached summarization runner for the MIMIC benchmark."""

from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd


class Summarizer(Protocol):
    def summarize_text(self, note_text: str, *, patient_id: str | None = None) -> str: ...


@dataclass(slots=True)
class RunStats:
    n_notes: int
    n_generated: int
    n_cache_hits: int
    elapsed_seconds: float


def run(
    notes: pd.DataFrame,
    summarizer: Summarizer,
    *,
    cache_dir: str | Path,
    model_tag: str,
    force_regenerate: bool = False,
    progress: bool = True,
) -> tuple[dict[int, str], RunStats]:
    """Generate (or load from cache) a summary per note row.

    Cache layout: ``<cache_dir>/<sanitized_model_tag>__<row_id>.txt``. Cache
    hits are atomic; partial writes are written to ``.tmp`` then renamed so a
    Ctrl-C during generation never leaves a half-written summary that would be
    silently reused on the next run.
    """

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sanitized_tag = _sanitize_tag(model_tag)

    if force_regenerate:
        for stale in cache_dir.glob(f"{sanitized_tag}__*.txt"):
            stale.unlink()

    iterator: Iterable = notes.itertuples(index=False)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                iterator,
                total=len(notes),
                desc=f"summarize ({sanitized_tag})",
                unit="note",
            )
        except ImportError:
            pass  # tqdm is optional; fall back to a quiet loop

    summaries: dict[int, str] = {}
    n_generated = 0
    n_cache_hits = 0
    start = time.perf_counter()

    for record in iterator:
        row_id = int(getattr(record, "row_id"))
        text = str(getattr(record, "text"))
        cache_path = cache_dir / f"{sanitized_tag}__{row_id}.txt"

        cached = _read_if_present(cache_path)
        if cached is not None:
            summaries[row_id] = cached
            n_cache_hits += 1
            continue

        summary = summarizer.summarize_text(
            text,
            patient_id=str(row_id),
        )
        _atomic_write(cache_path, summary)
        summaries[row_id] = summary
        n_generated += 1

    elapsed = time.perf_counter() - start
    stats = RunStats(
        n_notes=len(notes),
        n_generated=n_generated,
        n_cache_hits=n_cache_hits,
        elapsed_seconds=elapsed,
    )
    return summaries, stats


def _read_if_present(path: Path) -> str | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return text


def _atomic_write(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _sanitize_tag(tag: str) -> str:
    """Make a model identifier safe for use as a filename prefix."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", tag.strip())
    return cleaned.strip("-") or "model"


def clear_cache(cache_dir: str | Path) -> None:
    """Remove every cached summary in `cache_dir` (preserves .gitkeep)."""

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return
    for entry in cache_dir.iterdir():
        if entry.name == ".gitkeep":
            continue
        if entry.is_file():
            entry.unlink()
        else:
            shutil.rmtree(entry)
