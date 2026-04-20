"""Benchmark harness for clinical summarization on MIMIC-III-Ext-Notes."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PACKAGE_ROOT / "cache"
RESULTS_DIR = PACKAGE_ROOT / "results"

__all__ = ["PACKAGE_ROOT", "CACHE_DIR", "RESULTS_DIR"]
