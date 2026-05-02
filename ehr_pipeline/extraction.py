"""Extraction phase: build the evidence store and extract atomic claims.

This module owns stages 1-2 of the pipeline. Inputs are the raw FHIR bundle
plus optional free-text notes; outputs are an :class:`EvidenceStore` and a
:class:`ClaimList` persisted under ``outputs/<case_id>/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .runtime import StageTiming, is_fresh, time_stage
from .schemas import ClaimList, EvidenceStore
from .stages import s1_evidence, s2_extract

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    store: EvidenceStore
    claims: ClaimList
    evidence_path: Path
    claims_path: Path


def run_extraction(
    *,
    case_id: str,
    bundle_path: Path,
    notes_dir: Path | None,
    output_dir: Path,
    timings: list[StageTiming],
    resume: bool = False,
) -> ExtractionResult:
    """Run the extraction phase (stages 1-2) and return its artifacts."""
    es_path = output_dir / "evidence_store.json"
    skip_s1 = resume and is_fresh(
        es_path,
        [bundle_path] + (list(notes_dir.glob("*")) if notes_dir else []),
    )
    if skip_s1:
        store = time_stage(
            "stage1_evidence",
            lambda: EvidenceStore.model_validate_json(es_path.read_text("utf-8")),
            timings,
            skipped=True,
        )
    else:
        store = time_stage(
            "stage1_evidence",
            lambda: s1_evidence.run(
                case_id=case_id,
                bundle_path=bundle_path,
                notes_dir=notes_dir,
                output_dir=output_dir,
            ),
            timings,
        )

    claims_path = output_dir / "claims.json"
    skip_s2 = resume and is_fresh(claims_path, [es_path])
    if skip_s2:
        claims = time_stage(
            "stage2_extract",
            lambda: ClaimList.model_validate_json(claims_path.read_text("utf-8")),
            timings,
            skipped=True,
        )
    else:
        claims = time_stage(
            "stage2_extract",
            lambda: s2_extract.run(notes_dir=notes_dir, output_dir=output_dir),
            timings,
        )

    return ExtractionResult(
        store=store,
        claims=claims,
        evidence_path=es_path,
        claims_path=claims_path,
    )
