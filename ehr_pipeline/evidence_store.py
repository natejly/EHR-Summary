"""FHIR bundle parsing, note sentence splitting, and deterministic indexing.

Builds an EvidenceStore that downstream LLM stages can be grounded against
without ever shipping the raw notes back to the model.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import Code, Evidence, EvidenceStore

_ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
    "no.",
    "fig.",
    "jr.",
    "sr.",
    "approx.",
    "p.o.",
    "b.i.d.",
    "t.i.d.",
    "q.d.",
    "q.i.d.",
    "prn.",
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter that respects common medical abbreviations."""
    if not text or not text.strip():
        return []

    candidates = _SENTENCE_BOUNDARY.split(text.strip())
    merged: list[str] = []
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        if merged and merged[-1].lower().endswith(tuple(_ABBREVIATIONS)):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def _normalize_display(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _first_coding(codeable_concept: dict[str, Any] | None) -> Code | None:
    if not codeable_concept:
        return None
    codings = codeable_concept.get("coding") or []
    if codings:
        coding = codings[0]
        return Code(
            system=coding.get("system"),
            code=coding.get("code"),
            display=coding.get("display") or codeable_concept.get("text"),
        )
    text = codeable_concept.get("text")
    if text:
        return Code(display=text)
    return None


def _display_for(resource: dict[str, Any], code_field: str = "code") -> str:
    cc = resource.get(code_field) or {}
    if isinstance(cc, dict):
        if cc.get("text"):
            return cc["text"]
        for coding in cc.get("coding", []) or []:
            if coding.get("display"):
                return coding["display"]
    return resource.get("resourceType", "Unknown")


def _effective_date(resource: dict[str, Any]) -> str | None:
    for key in (
        "effectiveDateTime",
        "onsetDateTime",
        "recordedDate",
        "authoredOn",
        "performedDateTime",
        "issued",
        "period",
    ):
        value = resource.get(key)
        if isinstance(value, str):
            return value[:10]
        if isinstance(value, dict):
            start = value.get("start") or value.get("end")
            if isinstance(start, str):
                return start[:10]
    return None


def _value_and_unit(resource: dict[str, Any]) -> tuple[str | None, str | None]:
    qty = resource.get("valueQuantity")
    if isinstance(qty, dict) and qty.get("value") is not None:
        return str(qty["value"]), qty.get("unit")
    if isinstance(resource.get("valueString"), str):
        return resource["valueString"], None
    if isinstance(resource.get("valueCodeableConcept"), dict):
        cc = resource["valueCodeableConcept"]
        return cc.get("text") or _display_for({"code": cc}), None
    return None, None


def _resource_ref(entry: dict[str, Any], resource: dict[str, Any]) -> str:
    full_url = entry.get("fullUrl")
    if full_url:
        return full_url
    rid = resource.get("id") or "unknown"
    return f"{resource.get('resourceType', 'Resource')}/{rid}"


def evidence_from_bundle(bundle: dict[str, Any]) -> list[Evidence]:
    """Convert a FHIR R4 Bundle into a flat list of Evidence."""
    out: list[Evidence] = []
    counters: dict[str, int] = defaultdict(int)

    for entry in bundle.get("entry", []) or []:
        resource = entry.get("resource") or {}
        rtype = resource.get("resourceType")

        if rtype == "Condition":
            counters["cond"] += 1
            eid = f"E:cond:{counters['cond']}"
            out.append(
                Evidence(
                    id=eid,
                    kind="condition",
                    display=_display_for(resource),
                    code=_first_coding(resource.get("code")),
                    effective=_effective_date(resource),
                    source_ref=_resource_ref(entry, resource),
                )
            )

        elif rtype in ("MedicationRequest", "MedicationStatement"):
            counters["med"] += 1
            eid = f"E:med:{counters['med']}"
            med_cc = resource.get("medicationCodeableConcept") or {}
            display = med_cc.get("text") or _display_for({"code": med_cc}) or "Medication"
            dosage = ""
            for instr in resource.get("dosageInstruction") or []:
                if instr.get("text"):
                    dosage = instr["text"]
                    break
            value = dosage or None
            out.append(
                Evidence(
                    id=eid,
                    kind="medication",
                    display=display,
                    code=_first_coding(med_cc),
                    value=value,
                    effective=_effective_date(resource),
                    source_ref=_resource_ref(entry, resource),
                )
            )

        elif rtype == "Observation":
            counters["obs"] += 1
            eid = f"E:obs:{counters['obs']}"
            value, unit = _value_and_unit(resource)
            out.append(
                Evidence(
                    id=eid,
                    kind="observation",
                    display=_display_for(resource),
                    code=_first_coding(resource.get("code")),
                    value=value,
                    unit=unit,
                    effective=_effective_date(resource),
                    source_ref=_resource_ref(entry, resource),
                )
            )

        elif rtype == "AllergyIntolerance":
            counters["allergy"] += 1
            eid = f"E:allergy:{counters['allergy']}"
            out.append(
                Evidence(
                    id=eid,
                    kind="allergy",
                    display=_display_for(resource),
                    code=_first_coding(resource.get("code")),
                    effective=_effective_date(resource),
                    source_ref=_resource_ref(entry, resource),
                )
            )

        elif rtype == "Procedure":
            counters["proc"] += 1
            eid = f"E:proc:{counters['proc']}"
            out.append(
                Evidence(
                    id=eid,
                    kind="procedure",
                    display=_display_for(resource),
                    code=_first_coding(resource.get("code")),
                    effective=_effective_date(resource),
                    source_ref=_resource_ref(entry, resource),
                )
            )

        elif rtype == "Encounter":
            counters["enc"] += 1
            eid = f"E:enc:{counters['enc']}"
            display = "Encounter"
            ec = resource.get("class") or {}
            if isinstance(ec, dict) and ec.get("display"):
                display = ec["display"]
            elif resource.get("type"):
                display = _display_for(resource, code_field="type")
            out.append(
                Evidence(
                    id=eid,
                    kind="encounter",
                    display=display,
                    effective=_effective_date(resource),
                    source_ref=_resource_ref(entry, resource),
                )
            )

    return out


def evidence_from_notes(notes: Iterable[tuple[str, str]]) -> list[Evidence]:
    """Each note becomes one Evidence per sentence.

    notes is an iterable of (doc_id, text). Sentence ids look like E:note:doc:<n>.
    """
    out: list[Evidence] = []
    for doc_id, text in notes:
        sentences = split_sentences(text)
        for idx, sentence in enumerate(sentences, start=1):
            out.append(
                Evidence(
                    id=f"E:note:{doc_id}:{idx}",
                    kind="note_sentence",
                    display=sentence[:80],
                    text=sentence,
                    source_ref=f"note:{doc_id}#s{idx}",
                )
            )
    return out


def build_indexes(evidence: list[Evidence]) -> tuple[
    dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]
]:
    by_code: dict[str, list[str]] = defaultdict(list)
    by_display: dict[str, list[str]] = defaultdict(list)
    by_date: dict[str, list[str]] = defaultdict(list)

    for ev in evidence:
        if ev.code and ev.code.code:
            key = f"{ev.code.system or ''}|{ev.code.code}"
            by_code[key].append(ev.id)
        norm = _normalize_display(ev.display)
        if norm:
            by_display[norm].append(ev.id)
        if ev.effective:
            by_date[ev.effective].append(ev.id)

    return dict(by_code), dict(by_display), dict(by_date)


def load_notes_from_dir(notes_dir: Path) -> list[tuple[str, str]]:
    """Read every .txt/.md file in notes_dir as a note."""
    if not notes_dir.exists():
        return []
    notes: list[tuple[str, str]] = []
    for path in sorted(notes_dir.iterdir()):
        if path.suffix.lower() in (".txt", ".md") and path.is_file():
            notes.append((path.stem, path.read_text(encoding="utf-8")))
    return notes


def build_evidence_store(
    *,
    case_id: str,
    bundle_path: Path,
    notes_dir: Path | None,
) -> EvidenceStore:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    evidence = evidence_from_bundle(bundle)
    if notes_dir is not None:
        notes = load_notes_from_dir(notes_dir)
        evidence.extend(evidence_from_notes(notes))
    by_code, by_display, by_date = build_indexes(evidence)
    return EvidenceStore(
        case_id=case_id,
        evidence=evidence,
        by_code=by_code,
        by_display=by_display,
        by_date=by_date,
    )


def candidates_for_claim(
    store: EvidenceStore,
    *,
    display_query: str | None,
    code: dict[str, str] | None,
    date_hint: str | None,
    top_k: int,
) -> list[Evidence]:
    """Deterministic candidate retrieval for verification."""
    by_id = {ev.id: ev for ev in store.evidence}
    matched_ids: list[str] = []

    if code and code.get("code"):
        key = f"{code.get('system', '') or ''}|{code['code']}"
        matched_ids.extend(store.by_code.get(key, []))

    norm = _normalize_display(display_query) if display_query else ""
    if norm:
        matched_ids.extend(store.by_display.get(norm, []))
        for key, ids in store.by_display.items():
            if key == norm:
                continue
            if norm in key or key in norm:
                matched_ids.extend(ids)

    if date_hint:
        matched_ids.extend(store.by_date.get(date_hint[:10], []))

    seen: set[str] = set()
    ordered: list[Evidence] = []
    for eid in matched_ids:
        if eid in seen:
            continue
        seen.add(eid)
        ev = by_id.get(eid)
        if ev:
            ordered.append(ev)
        if len(ordered) >= top_k:
            break

    if not ordered and norm:
        token_set = set(re.findall(r"[a-z0-9]+", norm))
        scored: list[tuple[int, Evidence]] = []
        for ev in store.evidence:
            if ev.kind == "note_sentence":
                continue
            display_tokens = set(re.findall(r"[a-z0-9]+", _normalize_display(ev.display)))
            overlap = len(token_set & display_tokens)
            if overlap:
                scored.append((overlap, ev))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        ordered = [ev for _, ev in scored[:top_k]]

    return ordered
