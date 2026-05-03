"""Stage 8: final LLM review.

Advisory only -- the model reviews the gated summary, the fact sheet, and
the deterministic check report, and surfaces remaining concerns or
recommended revisions. Output is written to review.json (and folded into
audit.json by the orchestrator); the summary itself is never modified
automatically.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .. import config
from ..ollama_client import OllamaError, chat_json
from ..prompts import S8_REVIEW
from ..schemas import (
    CheckReport,
    FactSheet,
    ReviewReport,
    schema_for,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = S8_REVIEW

# Maximum entries per fact-sheet section sent to the review model.
# Large fact sheets (e.g. 39 active_problems) can push payloads past the
# cloud model's context limit and trigger HTTP 500 errors.
_MAX_ENTRIES_PER_SECTION = 15


def _trim_fact_sheet(fact_sheet: FactSheet) -> dict:
    """Return a compacted fact-sheet dict capped at _MAX_ENTRIES_PER_SECTION."""
    trimmed: dict = {"case_id": fact_sheet.case_id, "sections": {}}
    for section, entries in fact_sheet.sections.items():
        kept = entries[:_MAX_ENTRIES_PER_SECTION]
        trimmed["sections"][section] = [e.model_dump() for e in kept]
        if len(entries) > _MAX_ENTRIES_PER_SECTION:
            trimmed["sections"][section].append(
                {"text": f"… {len(entries) - _MAX_ENTRIES_PER_SECTION} more entries omitted",
                 "evidence_ids": []}
            )
    return trimmed


def run(
    *,
    summary_markdown: str,
    fact_sheet: FactSheet,
    check_report: CheckReport,
    output_dir: Path,
) -> ReviewReport:
    """Run the final advisory LLM review.

    Returns an empty ReviewReport (no concerns) if the cloud API fails with
    a server error — Stage 8 is advisory only and must never block emission.
    """
    log.info("Stage 8: final LLM review")
    payload = {
        "summary_markdown": summary_markdown,
        "fact_sheet": _trim_fact_sheet(fact_sheet),
        "check_report": check_report.model_dump(),
    }
    try:
        review = chat_json(
            model=config.MODELS.final_review,
            system=S8_REVIEW,
            user="Review this summary and report any remaining concerns.\n\n"
            + json.dumps(payload, indent=2, ensure_ascii=False),
            schema=schema_for(ReviewReport),
            temperature=0.1,
            validate=lambda raw: ReviewReport.model_validate(raw),
        )
    except OllamaError as exc:
        log.warning(
            "Stage 8 review skipped due to API error (%s); "
            "summary is still emitted — review is advisory only.",
            exc,
        )
        review = ReviewReport(concerns=[], recommended_revisions=[])

    out_path = output_dir / "review.json"
    out_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "  concerns=%d recommended_revisions=%d -> %s",
        len(review.concerns),
        len(review.recommended_revisions),
        out_path,
    )
    return review
