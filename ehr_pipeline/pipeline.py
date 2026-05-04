"""End-to-end pipeline orchestrator.

The heavy lifting now lives in :mod:`ehr_pipeline.extraction` (stages 1-2) and
:mod:`ehr_pipeline.verification` (stages 3-4). This module wires those phases
together and runs the remaining summary, check, and review stages, persisting
artifacts under ``outputs/<case_id>/``. With ``resume=True`` we skip stages
whose output is newer than its inputs. Stage 7 gates stage 8 and the final
emission unless ``allow_violations=True``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .extraction import run_extraction
from .runtime import StageTiming, is_fresh, time_stage
from .schemas import CheckReport, ClaimList, FactSheet, ReviewReport
from .stages import (
    s5_fact_sheet,
    s6_summarize,
    s7_check,
    s8_review,
    s9_patient_summary,
)
from .verification import run_verification

log = logging.getLogger(__name__)


@dataclass
class CompressionInfo:
    note_chars: int
    summary_chars: int
    target_min_chars: int
    target_max_chars: int
    achieved_ratio: float
    in_target_band: bool


@dataclass
class PatientSummaryInfo:
    path: Path
    fk_grade: float
    fk_words: int
    fk_sentences: int
    fk_syllables: int
    chars: int
    in_target_band: bool
    target_grade_min: float
    target_grade_max: float


@dataclass
class PipelineResult:
    case_id: str
    output_dir: Path
    summary_path: Path | None
    audit_path: Path
    check_passed: bool
    context_agent_enabled: bool = True
    compression: CompressionInfo | None = None
    timings: list[StageTiming] = field(default_factory=list)
    input_note_path: Path | None = None
    review_passed: bool = True          # False only when a high-severity concern is found
    review_concern_counts: dict = field(default_factory=dict)  # {"high":0,"medium":0,"low":0}
    revised_summary_path: Path | None = None   # summary_revised.md (None if no revisions applied)
    revised_check_passed: bool | None = None   # Stage 7 re-check on the revised summary
    patient_summary: PatientSummaryInfo | None = None


def _read_notes(notes_dir: Path | None) -> str:
    """Concatenate all note files and return the combined text."""
    if notes_dir is None or not notes_dir.exists():
        return ""
    parts: list[str] = []
    for path in sorted(notes_dir.iterdir()):
        if path.suffix.lower() in (".txt", ".md") and path.is_file():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def _count_note_chars(notes_dir: Path | None) -> int:
    return len(_read_notes(notes_dir))


def _claims_for_fact_sheet(out_dir: Path, base_claims: ClaimList) -> ClaimList:
    """Prefer merged claims when stage 4 suggested EHR-backed claims to verify."""
    aug_path = out_dir / "claims_augmented.json"
    if aug_path.exists():
        return ClaimList.model_validate_json(aug_path.read_text("utf-8"))
    return base_claims


def _apply_revisions(
    summary_md: str,
    review: ReviewReport,
    out_dir: Path,
    fact_sheet: FactSheet,
    store,
) -> tuple[Path | None, bool | None]:
    """Apply Stage 8 recommended_revisions to the summary and re-validate.

    Returns (revised_path, revised_check_passed).
    If there are no revisions, returns (None, None) — the original is already
    the best version.
    """
    if not review.recommended_revisions:
        log.info("No recommended revisions from Stage 8; original summary is final.")
        return None, None

    revised = summary_md
    applied = 0
    for rev in review.recommended_revisions:
        original = rev.original.strip()
        suggested = rev.suggested.strip()
        if original == suggested:
            continue
        if original in revised:
            revised = revised.replace(original, suggested, 1)
            applied += 1
            log.info("  applied revision: %r → %r", original[:60], suggested[:60])
        else:
            log.warning("  revision original not found verbatim in summary, skipping: %r", original[:60])

    if applied == 0:
        log.info("No revisions could be applied verbatim; original summary is final.")
        return None, None

    revised_path = out_dir / "summary_revised.md"
    revised_path.write_text(revised + "\n", encoding="utf-8")
    log.info("Wrote revised summary (%d revision(s) applied) -> %s", applied, revised_path)

    # Re-run the deterministic check on the revised text.
    revised_check = s7_check.run(
        summary_markdown=revised,
        fact_sheet=fact_sheet,
        store=store,
        output_dir=out_dir,
        report_filename="check_report_revised.json",
    )
    log.info(
        "  revised summary check: passed=%s violations=%d",
        revised_check.passed,
        len(revised_check.violations),
    )
    return revised_path, revised_check.passed


def run_pipeline(
    *,
    case_id: str,
    bundle_path: Path,
    notes_dir: Path | None,
    resume: bool = False,
    allow_violations: bool = False,
    enable_context_agent: bool = True,
    summary_min_ratio: float = config.SUMMARY_TARGET_COMPRESSION_MIN,
    summary_max_ratio: float = config.SUMMARY_TARGET_COMPRESSION_MAX,
) -> PipelineResult:
    out_dir = config.output_dir(case_id)
    timings: list[StageTiming] = []

    # Persist the raw input note alongside the summary so both are always
    # in the same directory (outputs/<case_id>/).
    note_text = _read_notes(notes_dir)
    input_note_path: Path | None = None
    if note_text:
        input_note_path = out_dir / "input_note.txt"
        if not (resume and input_note_path.exists()):
            input_note_path.write_text(note_text, encoding="utf-8")
            log.info("Saved input note (%d chars) -> %s", len(note_text), input_note_path)

    extraction = run_extraction(
        case_id=case_id,
        bundle_path=bundle_path,
        notes_dir=notes_dir,
        output_dir=out_dir,
        timings=timings,
        resume=resume,
    )

    verification = run_verification(
        claims=extraction.claims,
        store=extraction.store,
        output_dir=out_dir,
        evidence_path=extraction.evidence_path,
        claims_path=extraction.claims_path,
        timings=timings,
        resume=resume,
        enable_context_agent=enable_context_agent,
    )

    note_chars = len(note_text)

    fs_path = out_dir / "fact_sheet.json"
    s5_inputs = [
        verification.augmented_path,
        extraction.claims_path,
        extraction.evidence_path,
    ]
    aug_claims_path = out_dir / "claims_augmented.json"
    if aug_claims_path.exists():
        s5_inputs.append(aug_claims_path)
    skip_s5 = resume and is_fresh(fs_path, s5_inputs)
    if skip_s5:
        fact_sheet = time_stage(
            "stage5_fact_sheet",
            lambda: FactSheet.model_validate_json(fs_path.read_text("utf-8")),
            timings,
            skipped=True,
        )
    else:
        fact_sheet = time_stage(
            "stage5_fact_sheet",
            lambda: s5_fact_sheet.run(
                case_id=case_id,
                claims=_claims_for_fact_sheet(out_dir, extraction.claims),
                verifications=verification.augmented,
                store=extraction.store,
                output_dir=out_dir,
            ),
            timings,
        )

    # Stage 9 — patient-facing summary (independent of clinician summary).
    # Reads only the verified fact sheet, so it can run safely once we have one.
    patient_md_path = out_dir / "patient_summary.md"
    patient_meta_path = out_dir / "patient_summary_meta.json"
    skip_s9 = (
        resume
        and is_fresh(patient_md_path, [fs_path])
        and patient_meta_path.exists()
    )
    if skip_s9:
        log.info("Skipping stage9_patient_summary (cached)")
        meta = json.loads(patient_meta_path.read_text("utf-8"))
        patient_summary = PatientSummaryInfo(
            path=patient_md_path,
            fk_grade=meta.get("fk_grade", 0.0),
            fk_words=meta.get("fk_words", 0),
            fk_sentences=meta.get("fk_sentences", 0),
            fk_syllables=meta.get("fk_syllables", 0),
            chars=meta.get("chars", 0),
            in_target_band=meta.get("in_target_band", False),
            target_grade_min=meta.get("target_grade_min", config.PATIENT_SUMMARY_GRADE_MIN),
            target_grade_max=meta.get("target_grade_max", config.PATIENT_SUMMARY_GRADE_MAX),
        )
        timings.append(StageTiming(name="stage9_patient_summary", seconds=0.0, skipped=True))
    else:
        try:
            patient_result = time_stage(
                "stage9_patient_summary",
                lambda: s9_patient_summary.run(
                    fact_sheet=fact_sheet,
                    output_dir=out_dir,
                ),
                timings,
            )
            patient_summary = PatientSummaryInfo(
                path=patient_md_path,
                fk_grade=patient_result.fk_grade,
                fk_words=patient_result.fk_words,
                fk_sentences=patient_result.fk_sentences,
                fk_syllables=patient_result.fk_syllables,
                chars=patient_result.chars,
                in_target_band=patient_result.in_target_band,
                target_grade_min=patient_result.target_grade_min,
                target_grade_max=patient_result.target_grade_max,
            )
        except Exception as exc:
            log.warning("Stage 9 patient summary failed (non-fatal): %s", exc)
            patient_summary = None

    summary_path = out_dir / "summary.md"
    summary_meta_path = out_dir / "summary_meta.json"
    skip_s6 = resume and is_fresh(summary_path, [fs_path]) and summary_meta_path.exists()
    if skip_s6:
        meta = json.loads(summary_meta_path.read_text("utf-8"))
        summary_md = summary_path.read_text("utf-8")
        compression = CompressionInfo(**meta)
        timings.append(StageTiming(name="stage6_summarize", seconds=0.0, skipped=True))
        log.info("Skipping stage6_summarize (cached)")
    else:
        summary_result = time_stage(
            "stage6_summarize",
            lambda: s6_summarize.run(
                fact_sheet=fact_sheet,
                output_dir=out_dir,
                note_chars=note_chars,
                min_ratio=summary_min_ratio,
                max_ratio=summary_max_ratio,
            ),
            timings,
        )
        summary_md = summary_result.markdown
        compression = CompressionInfo(
            note_chars=summary_result.note_chars,
            summary_chars=summary_result.summary_chars,
            target_min_chars=summary_result.target_chars_min,
            target_max_chars=summary_result.target_chars_max,
            achieved_ratio=summary_result.achieved_ratio,
            in_target_band=summary_result.in_target_band,
        )
        summary_meta_path.write_text(
            json.dumps(compression.__dict__, indent=2), encoding="utf-8"
        )

    check_path = out_dir / "check_report.json"
    skip_s7 = resume and is_fresh(
        check_path, [summary_path, fs_path, extraction.evidence_path]
    )
    if skip_s7:
        check_report = time_stage(
            "stage7_check",
            lambda: CheckReport.model_validate_json(check_path.read_text("utf-8")),
            timings,
            skipped=True,
        )
    else:
        check_report = time_stage(
            "stage7_check",
            lambda: s7_check.run(
                summary_markdown=summary_md,
                fact_sheet=fact_sheet,
                store=extraction.store,
                output_dir=out_dir,
            ),
            timings,
        )

    if not check_report.passed and not allow_violations:
        audit = {
            "case_id": case_id,
            "stage_failed": "stage7_check",
            "violations": [v.model_dump() for v in check_report.violations],
            "compression": compression.__dict__,
            "context_agent_enabled": verification.context_agent_enabled,
            "timings": [t.__dict__ for t in timings],
        }
        audit_path = out_dir / "audit.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        log.error(
            "Deterministic check failed with %d violations; emission halted.",
            len(check_report.violations),
        )
        return PipelineResult(
            case_id=case_id,
            output_dir=out_dir,
            summary_path=None,
            audit_path=audit_path,
            check_passed=False,
            context_agent_enabled=verification.context_agent_enabled,
            compression=compression,
            timings=timings,
            input_note_path=input_note_path,
            patient_summary=patient_summary,
        )

    review_path = out_dir / "review.json"
    skip_s8 = resume and is_fresh(review_path, [summary_path, fs_path, check_path])
    if skip_s8:
        review = time_stage(
            "stage8_review",
            lambda: ReviewReport.model_validate_json(review_path.read_text("utf-8")),
            timings,
            skipped=True,
        )
    else:
        review = time_stage(
            "stage8_review",
            lambda: s8_review.run(
                summary_markdown=summary_md,
                fact_sheet=fact_sheet,
                check_report=check_report,
                output_dir=out_dir,
            ),
            timings,
        )

    # Apply Stage 8 recommended revisions deterministically, then re-check.
    revised_summary_path, revised_check_passed = _apply_revisions(
        summary_md=summary_md,
        review=review,
        out_dir=out_dir,
        fact_sheet=fact_sheet,
        store=extraction.store,
    )

    audit = {
        "case_id": case_id,
        "stage_failed": None,
        "context_agent_enabled": verification.context_agent_enabled,
        "context": verification.context_report.model_dump(),
        "compression": compression.__dict__,
        "check_report": check_report.model_dump(),
        "review": review.model_dump(),
        "review_passed": review.passed,
        "review_concern_counts": review.concern_counts,
        "revised_summary": str(revised_summary_path) if revised_summary_path else None,
        "revised_check_passed": revised_check_passed,
        "patient_summary": (
            {k: v for k, v in patient_summary.__dict__.items() if k != "path"}
            | {"path": str(patient_summary.path)}
            if patient_summary else None
        ),
        "timings": [t.__dict__ for t in timings],
        "models": {
            "claim_extraction": config.MODELS.claim_extraction,
            "claim_verification": config.MODELS.claim_verification,
            "context_agent": config.MODELS.context_agent,
            "summary_generation": config.MODELS.summary_generation,
            "final_review": config.MODELS.final_review,
        },
    }
    audit_path = out_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    return PipelineResult(
        case_id=case_id,
        output_dir=out_dir,
        summary_path=summary_path,
        audit_path=audit_path,
        check_passed=check_report.passed,
        context_agent_enabled=verification.context_agent_enabled,
        compression=compression,
        timings=timings,
        input_note_path=input_note_path,
        review_passed=review.passed,
        review_concern_counts=review.concern_counts,
        revised_summary_path=revised_summary_path,
        revised_check_passed=revised_check_passed,
        patient_summary=patient_summary,
    )
