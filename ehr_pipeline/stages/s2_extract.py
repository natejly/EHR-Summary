"""Stage 2: claim extraction from notes only.

The structured EHR is intentionally NOT shown to the model; everything it
extracts will be cross-checked against the EHR by stage 3.
Prompt lives in ehr_pipeline/prompts.py (S2_EXTRACTION).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config
from ..evidence_store import load_notes_from_dir
from ..ollama_client import chat_json
from ..prompts import S2_EXTRACTION
from ..schemas import ClaimList, schema_for

log = logging.getLogger(__name__)

SYSTEM_PROMPT = S2_EXTRACTION  # kept for backward-compat imports


def _format_notes(notes: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for doc_id, text in notes:
        parts.append(f"=== NOTE doc_id={doc_id} ===\n{text.strip()}\n")
    return "\n".join(parts)


def run(
    *,
    notes_dir: Path | None,
    output_dir: Path,
) -> ClaimList:
    log.info("Stage 2: extracting claims from notes")
    notes = load_notes_from_dir(notes_dir) if notes_dir else []
    if not notes:
        log.warning("  no notes found; producing empty claim list")
        result = ClaimList(claims=[])
        (output_dir / "claims.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        return result

    user = (
        "Extract every clinical claim from the following notes. "
        "Respond with a single JSON object matching the schema in the system prompt.\n\n"
        + _format_notes(notes)
    )

    raw = chat_json(
        model=config.MODELS.claim_extraction,
        system=S2_EXTRACTION,
        user=user,
        schema=schema_for(ClaimList),
        temperature=0.0,
    )

    result = ClaimList.model_validate(raw)
    out_path = output_dir / "claims.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    log.info("  extracted %d claims, wrote %s", len(result.claims), out_path)
    return result
