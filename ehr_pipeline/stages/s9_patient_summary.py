"""Stage 9: patient-facing plain-language summary.

Sees ONLY the verified fact sheet (same input as stage 6) so it inherits the
same hallucination bound. Generates a discharge-style summary written in plain
English aimed at a 7th-8th grade reading level, while still keeping inline
[E:...] citations for traceability.

Outputs:
    patient_summary.md       – Markdown text with citations
    patient_summary_meta.json – {fk_grade, fk_words, ..., target_grade_min/max}
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import config
from ..ollama_client import chat_text
from ..prompts import S9_PATIENT_SUMMARY_TEMPLATE
from ..schemas import FactSheet

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatientSummaryResult:
    markdown: str
    fk_grade: float
    fk_words: int
    fk_sentences: int
    fk_syllables: int
    target_grade_min: float
    target_grade_max: float
    in_target_band: bool
    chars: int


def _format_fact_sheet(fs: FactSheet) -> str:
    """Same minimal serialization as s6 so the model has a uniform input."""
    lines: list[str] = [f"case_id: {fs.case_id}", ""]
    for section, entries in fs.sections.items():
        lines.append(f"[{section}]")
        if not entries:
            lines.append("  (none)")
        for entry in entries:
            ids = "".join(f"[E:{eid.removeprefix('E:')}]" for eid in entry.evidence_ids)
            lines.append(f"  - {entry.text} {ids}")
        lines.append("")
    return "\n".join(lines)


def run(
    *,
    fact_sheet: FactSheet,
    output_dir: Path,
    target_grade_min: float = config.PATIENT_SUMMARY_GRADE_MIN,
    target_grade_max: float = config.PATIENT_SUMMARY_GRADE_MAX,
    min_chars: int = config.PATIENT_SUMMARY_MIN_CHARS,
    max_chars: int = config.PATIENT_SUMMARY_MAX_CHARS,
) -> PatientSummaryResult:
    """Generate a patient-facing summary from the verified fact sheet."""
    log.info("Stage 9: generating patient summary (target FK grade %.0f-%.0f)",
             target_grade_min, target_grade_max)

    system_prompt = S9_PATIENT_SUMMARY_TEMPLATE.format(
        target_grade_min=int(target_grade_min),
        target_grade_max=int(target_grade_max),
        min_chars=min_chars,
        max_chars=max_chars,
    )
    user = (
        "Write the patient summary using only this fact sheet, "
        f"keeping it between {min_chars} and {max_chars} characters and at "
        f"a {int(target_grade_min)}-{int(target_grade_max)} grade reading level.\n\n"
        + _format_fact_sheet(fact_sheet)
    )

    markdown = chat_text(
        model=config.MODELS.summary_generation,
        system=system_prompt,
        user=user,
        temperature=0.2,
    ).strip()

    out_path = output_dir / "patient_summary.md"
    out_path.write_text(markdown + "\n", encoding="utf-8")

    # Score the result so the orchestrator/UI can flag it.
    # Imported lazily so a missing benchmarks dep never breaks the pipeline.
    try:
        from benchmarks.metrics import flesch_kincaid_grade
        fk = flesch_kincaid_grade(markdown)
    except Exception as exc:
        log.warning("Could not compute Flesch-Kincaid: %s", exc)
        fk = {"fk_grade": 0.0, "fk_words": 0, "fk_sentences": 0, "fk_syllables": 0}

    in_band = target_grade_min <= float(fk["fk_grade"]) <= target_grade_max
    result = PatientSummaryResult(
        markdown=markdown,
        fk_grade=float(fk["fk_grade"]),
        fk_words=int(fk["fk_words"]),
        fk_sentences=int(fk["fk_sentences"]),
        fk_syllables=int(fk["fk_syllables"]),
        target_grade_min=float(target_grade_min),
        target_grade_max=float(target_grade_max),
        in_target_band=in_band,
        chars=len(markdown),
    )

    meta_path = output_dir / "patient_summary_meta.json"
    meta = {k: v for k, v in asdict(result).items() if k != "markdown"}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("  wrote %s (%d chars, FK grade=%.2f, in_band=%s)",
             out_path, result.chars, result.fk_grade, in_band)
    return result
