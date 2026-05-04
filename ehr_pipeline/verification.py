"""Verification phase: validate claims and merge EHR-backed context (stage 4).

This module owns stages 3-4. It consumes extraction artifacts (`EvidenceStore`
+ `ClaimList`) and produces `VerificationResult`, optional `ContextReport`, and
`verifications_augmented.json`. Stage 4 proposes additional claims from
structured evidence; verified suggestions are merged on disk for the fact
sheet builder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .runtime import StageTiming, is_fresh, time_stage
from .schemas import ClaimList, ContextReport, EvidenceStore, VerificationResult
from .stages import s3_verify, s4_context

log = logging.getLogger(__name__)


@dataclass
class VerificationOutput:
    verifications: VerificationResult
    context_report: ContextReport
    augmented: VerificationResult
    verifications_path: Path
    context_path: Path
    augmented_path: Path
    context_agent_enabled: bool


def run_verification(
    *,
    claims: ClaimList,
    store: EvidenceStore,
    output_dir: Path,
    evidence_path: Path,
    claims_path: Path,
    timings: list[StageTiming],
    resume: bool = False,
    enable_context_agent: bool = True,
) -> VerificationOutput:
    """Run the verification phase (stages 3-4).

    When ``enable_context_agent`` is False we skip stage 4 entirely; the raw
    verifications double as the "augmented" result so downstream stages see
    only verified claims and no LLM-suggested additions. This mode is useful
    for benchmarking where determinism + cost matter more than the extra
    coverage the context agent provides.
    """
    verif_path = output_dir / "verifications.json"
    skip_s3 = resume and is_fresh(verif_path, [claims_path, evidence_path])
    if skip_s3:
        verifications = time_stage(
            "stage3_verify",
            lambda: VerificationResult.model_validate_json(
                verif_path.read_text("utf-8")
            ),
            timings,
            skipped=True,
        )
    else:
        verifications = time_stage(
            "stage3_verify",
            lambda: s3_verify.run(
                claims=claims, store=store, output_dir=output_dir
            ),
            timings,
        )

    context_path = output_dir / "context.json"
    aug_path = output_dir / "verifications_augmented.json"

    if not enable_context_agent:
        log.info("Stage 4 (context agent) disabled by caller; using raw verifications")
        timings.append(StageTiming(name="stage4_context", seconds=0.0, skipped=True))
        empty_context = ContextReport()
        context_path.write_text(empty_context.model_dump_json(indent=2), encoding="utf-8")
        aug_path.write_text(verifications.model_dump_json(indent=2), encoding="utf-8")
        return VerificationOutput(
            verifications=verifications,
            context_report=empty_context,
            augmented=verifications,
            verifications_path=verif_path,
            context_path=context_path,
            augmented_path=aug_path,
            context_agent_enabled=False,
        )

    skip_s4 = resume and is_fresh(aug_path, [verif_path, evidence_path])
    if skip_s4:
        context_report = ContextReport.model_validate_json(
            context_path.read_text("utf-8")
        )
        augmented = VerificationResult.model_validate_json(
            aug_path.read_text("utf-8")
        )
        timings.append(StageTiming(name="stage4_context", seconds=0.0, skipped=True))
        log.info("Skipping stage4_context (cached)")
    else:
        context_report, augmented = time_stage(
            "stage4_context",
            lambda: s4_context.run(
                claims=claims,
                verifications=verifications,
                store=store,
                output_dir=output_dir,
            ),
            timings,
        )

    return VerificationOutput(
        verifications=verifications,
        context_report=context_report,
        augmented=augmented,
        verifications_path=verif_path,
        context_path=context_path,
        augmented_path=aug_path,
        context_agent_enabled=True,
    )
