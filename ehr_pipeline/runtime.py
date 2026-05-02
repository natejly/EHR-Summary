"""Shared runtime helpers for pipeline orchestration.

These primitives let the extraction, verification, and top-level pipeline
modules share the same caching/timing semantics without circular imports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class StageTiming:
    name: str
    seconds: float
    skipped: bool = False


def is_fresh(output: Path, inputs: Iterable[Path]) -> bool:
    """Return True when ``output`` exists and is newer than every input."""
    if not output.exists():
        return False
    out_mtime = output.stat().st_mtime
    return all(inp.exists() and inp.stat().st_mtime <= out_mtime for inp in inputs)


def time_stage(
    label: str,
    fn: Callable[[], T],
    timings: list[StageTiming],
    *,
    skipped: bool = False,
) -> T:
    """Run ``fn`` while appending a `StageTiming` entry to ``timings``."""
    if skipped:
        log.info("Skipping %s (cached)", label)
        timings.append(StageTiming(name=label, seconds=0.0, skipped=True))
        return fn()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    timings.append(StageTiming(name=label, seconds=elapsed))
    log.info("  %s done in %.2fs", label, elapsed)
    return result
