"""Stage 3: per-claim verification against the structured evidence store.

For each extracted claim we deterministically pull a small candidate set
from the inverted indexes (top_k by code/display/date), then ask the
verification model whether the claim is supported by the candidates. We use
asyncio + a semaphore to bound in-flight requests against Ollama Cloud.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from pathlib import Path
from typing import Any

from .. import config
from ..evidence_store import candidates_for_claim
from ..ollama_client import chat_json
from ..prompts import S3_VERIFICATION
from ..schemas import (
    Claim,
    ClaimList,
    EvidenceStore,
    Verification,
    VerificationResult,
    schema_for,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = S3_VERIFICATION


def _claim_value_text(claim: Claim) -> str:
    parts = [claim.predicate]
    if claim.value:
        parts.append(claim.value)
    return " ".join(parts).strip()


def _candidate_payload(claim: Claim, store: EvidenceStore) -> list[dict[str, Any]]:
    candidates = candidates_for_claim(
        store,
        display_query=claim.value or claim.predicate,
        code=None,
        date_hint=claim.time_ref,
        top_k=config.VERIFY_TOPK_CANDIDATES,
    )
    out: list[dict[str, Any]] = []
    for ev in candidates:
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


def _verify_one_sync(claim: Claim, candidates: list[dict[str, Any]]) -> Verification:
    if not candidates:
        return Verification(
            claim_id=claim.claim_id,
            status="unsupported",
            evidence_ids=[],
            rationale="No candidate evidence found in EHR.",
        )

    user_payload = {
        "claim": claim.model_dump(),
        "candidates": candidates,
    }
    try:
        return chat_json(
            model=config.MODELS.claim_verification,
            system=S3_VERIFICATION,
            user="Verify this claim against the candidates.\n\n" + _to_json(user_payload),
            schema=schema_for(Verification),
            temperature=0.0,
            validate=lambda raw: Verification.model_validate(raw),
        )
    except Exception:
        log.exception("Verification failed for %s after retries", claim.claim_id)
        return Verification(
            claim_id=claim.claim_id,
            status="unsupported",
            evidence_ids=[],
            rationale="Verifier returned malformed output after retries.",
        )


def _to_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


async def _verify_all(claims: list[Claim], store: EvidenceStore) -> list[Verification]:
    sem = asyncio.Semaphore(config.VERIFY_MAX_CONCURRENCY)
    loop = asyncio.get_running_loop()

    async def _one(claim: Claim) -> Verification:
        candidates = _candidate_payload(claim, store)
        async with sem:
            return await loop.run_in_executor(
                None, _verify_one_sync, claim, candidates
            )

    return await asyncio.gather(*[_one(c) for c in claims])


def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running.

    ``asyncio.run()`` raises RuntimeError when called from inside Jupyter
    (which keeps its own event loop alive). In that case we spin up a
    fresh thread — which has no running loop — and call ``asyncio.run``
    there instead.
    """
    try:
        asyncio.get_running_loop()
        # A loop is already running (Jupyter / async context).
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — standard script / CLI path.
        return asyncio.run(coro)


def run(
    *,
    claims: ClaimList,
    store: EvidenceStore,
    output_dir: Path,
) -> VerificationResult:
    log.info("Stage 3: verifying %d claims", len(claims.claims))
    if not claims.claims:
        result = VerificationResult(verifications=[])
    else:
        verifications = _run_async(_verify_all(claims.claims, store))
        result = VerificationResult(verifications=verifications)

    out_path = output_dir / "verifications.json"
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    counts = {"verified": 0, "contradicted": 0, "unsupported": 0}
    for v in result.verifications:
        counts[v.status] = counts.get(v.status, 0) + 1
    log.info("  %s -> %s", counts, out_path)
    return result
