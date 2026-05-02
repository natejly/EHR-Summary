"""Stage 5: deterministic fact sheet compilation.

Takes only verified claims (with at least one evidence id) and merges them
with directly-structured EHR facts that should always appear in a clinician
summary regardless of whether the notes mention them.

Output groups facts into the sections used by the summary template:
hpi, active_problems, medications, labs, vitals, plan.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from .. import config
from ..schemas import (
    ClaimList,
    Evidence,
    EvidenceStore,
    FactSheet,
    FactSheetEntry,
    VerificationResult,
)

log = logging.getLogger(__name__)


CLAIM_TYPE_TO_SECTION = {
    "diagnosis": "active_problems",
    "medication": "medications",
    "lab": "labs",
    "vital": "vitals",
    "procedure": "active_problems",
    "allergy": "active_problems",
    "plan": "plan",
}


def _format_evidence_for_fact_sheet(ev: Evidence) -> str:
    parts = [ev.display]
    if ev.value is not None:
        if ev.unit:
            parts.append(f"{ev.value} {ev.unit}")
        else:
            parts.append(str(ev.value))
    if ev.effective:
        parts.append(f"({ev.effective})")
    return " ".join(parts)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None


def _structured_facts(store: EvidenceStore) -> dict[str, list[FactSheetEntry]]:
    today = date.today()
    cutoff = today - timedelta(days=config.LAB_OBSERVATION_WINDOW_DAYS)

    sections: dict[str, list[FactSheetEntry]] = {
        "hpi": [],
        "active_problems": [],
        "medications": [],
        "labs": [],
        "vitals": [],
        "plan": [],
    }

    for ev in store.evidence:
        text = _format_evidence_for_fact_sheet(ev)
        entry = FactSheetEntry(text=text, evidence_ids=[ev.id])
        if ev.kind == "condition":
            sections["active_problems"].append(entry)
        elif ev.kind == "medication":
            sections["medications"].append(entry)
        elif ev.kind == "observation":
            ev_date = _parse_iso_date(ev.effective)
            section = "labs"
            if ev.unit in {"bpm", "/min", "mmHg", "mm[Hg]", "cel", "C", "kg", "cm"}:
                section = "vitals"
            else:
                display_lc = (ev.display or "").lower()
                if any(
                    keyword in display_lc
                    for keyword in (
                        "blood pressure",
                        "heart rate",
                        "pulse",
                        "respiratory rate",
                        "temperature",
                        "oxygen saturation",
                        "spo2",
                        "weight",
                        "height",
                        "bmi",
                    )
                ):
                    section = "vitals"
            if ev_date is None or ev_date >= cutoff:
                sections[section].append(entry)
        elif ev.kind == "allergy":
            sections["active_problems"].append(
                FactSheetEntry(text=f"Allergy: {text}", evidence_ids=[ev.id])
            )
        elif ev.kind == "procedure":
            sections["active_problems"].append(
                FactSheetEntry(text=f"Procedure: {text}", evidence_ids=[ev.id])
            )

    return sections


def _claim_text(claim_value: str | None, claim_predicate: str) -> str:
    if claim_value:
        return f"{claim_predicate}: {claim_value}"
    return claim_predicate


def _merge(target: list[FactSheetEntry], new_entry: FactSheetEntry) -> None:
    for existing in target:
        if existing.text.strip().lower() == new_entry.text.strip().lower():
            for eid in new_entry.evidence_ids:
                if eid not in existing.evidence_ids:
                    existing.evidence_ids.append(eid)
            return
    target.append(new_entry)


def _verified_claim_entries(
    claims: ClaimList,
    verifications: VerificationResult,
) -> dict[str, list[FactSheetEntry]]:
    by_id = {c.claim_id: c for c in claims.claims}
    sections: dict[str, list[FactSheetEntry]] = {
        "hpi": [],
        "active_problems": [],
        "medications": [],
        "labs": [],
        "vitals": [],
        "plan": [],
    }

    for v in verifications.verifications:
        if v.status != "verified" or not v.evidence_ids:
            continue
        claim = by_id.get(v.claim_id)
        if claim is None:
            continue
        section = CLAIM_TYPE_TO_SECTION.get(claim.type, "hpi")
        text = _claim_text(claim.value, claim.predicate)
        entry = FactSheetEntry(text=text, evidence_ids=list(v.evidence_ids))
        _merge(sections[section], entry)

    return sections


def run(
    *,
    case_id: str,
    claims: ClaimList,
    verifications: VerificationResult,
    store: EvidenceStore,
    output_dir: Path,
) -> FactSheet:
    log.info("Stage 5: compiling verified fact sheet")
    structured = _structured_facts(store)
    from_claims = _verified_claim_entries(claims, verifications)

    merged: dict[str, list[FactSheetEntry]] = {key: [] for key in structured}
    for section in merged:
        for entry in structured.get(section, []):
            _merge(merged[section], entry)
        for entry in from_claims.get(section, []):
            _merge(merged[section], entry)

    fact_sheet = FactSheet(case_id=case_id, sections=merged)
    out_path = output_dir / "fact_sheet.json"
    out_path.write_text(fact_sheet.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "  fact sheet sections: %s",
        {k: len(v) for k, v in merged.items()},
    )
    return fact_sheet
