"""Stage 1 wrapper: build the evidence store and persist it to disk."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..evidence_store import build_evidence_store
from ..schemas import EvidenceStore

log = logging.getLogger(__name__)


def run(
    *,
    case_id: str,
    bundle_path: Path,
    notes_dir: Path | None,
    output_dir: Path,
) -> EvidenceStore:
    log.info("Stage 1: building evidence store for case %s", case_id)
    store = build_evidence_store(
        case_id=case_id, bundle_path=bundle_path, notes_dir=notes_dir
    )
    out_path = output_dir / "evidence_store.json"
    out_path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "  wrote %d evidence items (%d structured, %d note sentences) to %s",
        len(store.evidence),
        sum(1 for ev in store.evidence if ev.kind != "note_sentence"),
        sum(1 for ev in store.evidence if ev.kind == "note_sentence"),
        out_path,
    )
    return store
