"""Stage 4: context agent.

Looks at the full claim list, the verifications, and a compact evidence
dump to surface missing context, contradictions, and structured facts that
would meaningfully strengthen the summary if verified. Any newly suggested
claim is fed back through stage 3 so the verified fact sheet only ever
contains verified material.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .. import config
from ..ollama_client import chat_json
from ..prompts import S4_CONTEXT
from ..schemas import (
    Claim,
    ClaimList,
    ContextReport,
    EvidenceStore,
    VerificationResult,
    schema_for,
)
from . import s3_verify

log = logging.getLogger(__name__)

SYSTEM_PROMPT = S4_CONTEXT


def _compact_evidence(store: EvidenceStore) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in store.evidence:
        if ev.kind == "note_sentence":
            continue
        out.append(
            {
                "id": ev.id,
                "kind": ev.kind,
                "display": ev.display,
                "code": ev.code.model_dump() if ev.code else None,
                "value": ev.value,
                "unit": ev.unit,
                "effective": ev.effective,
            }
        )
    return out


def run(
    *,
    claims: ClaimList,
    verifications: VerificationResult,
    store: EvidenceStore,
    output_dir: Path,
) -> tuple[ContextReport, VerificationResult]:
    log.info("Stage 4: context agent")

    payload = {
        "claims": [c.model_dump() for c in claims.claims],
        "verifications": [v.model_dump() for v in verifications.verifications],
        "evidence": _compact_evidence(store),
    }

    report = chat_json(
        model=config.MODELS.context_agent,
        system=S4_CONTEXT,
        user="Audit this case and return the requested JSON.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False),
        schema=schema_for(ContextReport),
        temperature=0.1,
        validate=lambda raw: ContextReport.model_validate(raw),
    )

    augmented = VerificationResult(verifications=list(verifications.verifications))
    suggested_claims: list[Claim] = []
    for suggestion in report.suggested_supporting_facts:
        if suggestion.suggested_claim is not None:
            suggested_claims.append(suggestion.suggested_claim)

    if suggested_claims:
        log.info(
            "  context agent proposed %d new claims; re-verifying", len(suggested_claims)
        )
        reverify_dir = output_dir / "_context_reverify"
        reverify_dir.mkdir(parents=True, exist_ok=True)
        extra = s3_verify.run(
            claims=ClaimList(claims=suggested_claims),
            store=store,
            output_dir=reverify_dir,
        )
        augmented = VerificationResult(
            verifications=list(verifications.verifications) + list(extra.verifications)
        )

        merged_claims = ClaimList(claims=list(claims.claims) + suggested_claims)
        (output_dir / "claims_augmented.json").write_text(
            merged_claims.model_dump_json(indent=2), encoding="utf-8"
        )

    (output_dir / "context.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_dir / "verifications_augmented.json").write_text(
        augmented.model_dump_json(indent=2), encoding="utf-8"
    )
    log.info(
        "  missing_context=%d contradictions=%d suggestions=%d",
        len(report.missing_context),
        len(report.contradictions),
        len(report.suggested_supporting_facts),
    )
    return report, augmented
