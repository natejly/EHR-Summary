"""MIMIC adapters for the EHR pipeline"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ICD9_SYSTEM = "http://hl7.org/fhir/sid/icd-9-cm"


def load_admissions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_id_for(admission: dict[str, Any]) -> str:
    return f"mimic-{admission['subject_id']}-{admission['hadm_id']}"


def _admit_date(admission: dict[str, Any]) -> str | None:
    raw = admission.get("admittime")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except ValueError:
        return raw[:10]


def _disch_date(admission: dict[str, Any]) -> str | None:
    raw = admission.get("dischtime")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date().isoformat()
    except ValueError:
        return raw[:10]


def admission_to_fhir_bundle(admission: dict[str, Any]) -> dict[str, Any]:
    subject_id = admission["subject_id"]
    hadm_id = admission["hadm_id"]
    admit_date = _admit_date(admission)
    disch_date = _disch_date(admission)

    entries: list[dict[str, Any]] = [
        {
            "fullUrl": f"urn:uuid:patient-{subject_id}",
            "resource": {
                "resourceType": "Patient",
                "id": f"patient-{subject_id}",
                "gender": (admission.get("gender") or "").lower() or "unknown",
            },
        }
    ]

    if admit_date:
        entries.append(
            {
                "fullUrl": f"urn:uuid:enc-{hadm_id}",
                "resource": {
                    "resourceType": "Encounter",
                    "id": f"enc-{hadm_id}",
                    "status": "finished",
                    "class": {
                        "code": "IMP" if admission.get("admission_type") == "EMERGENCY" else "AMB",
                        "display": admission.get("admission_type") or "Encounter",
                    },
                    "type": [{"text": admission.get("admission_type") or "Encounter"}],
                    "period": {
                        "start": f"{admit_date}T00:00:00",
                        "end": f"{disch_date or admit_date}T00:00:00",
                    },
                },
            }
        )

    for diag in admission.get("all_diagnoses", []) or []:
        seq = diag["seq_num"]
        icd = diag.get("icd_code")
        title = diag.get("diagnosis_title") or "Diagnosis"
        entries.append(
            {
                "fullUrl": f"urn:uuid:cond-{hadm_id}-{seq}",
                "resource": {
                    "resourceType": "Condition",
                    "id": f"cond-{hadm_id}-{seq}",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "code": {
                        "coding": [
                            {"system": ICD9_SYSTEM, "code": icd, "display": title}
                        ],
                        "text": title,
                    },
                    "onsetDateTime": admit_date,
                },
            }
        )

    for proc in admission.get("all_procedures", []) or []:
        seq = proc["seq_num"]
        title = proc.get("procedure_title") or "Procedure"
        entries.append(
            {
                "fullUrl": f"urn:uuid:proc-{hadm_id}-{seq}",
                "resource": {
                    "resourceType": "Procedure",
                    "id": f"proc-{hadm_id}-{seq}",
                    "status": "completed",
                    "code": {"text": title},
                    "performedDateTime": admit_date,
                },
            }
        )

    return {
        "resourceType": "Bundle",
        "id": f"mimic-{subject_id}-{hadm_id}",
        "type": "collection",
        "entry": entries,
    }


def admission_to_synthetic_note(admission: dict[str, Any]) -> str:
    """Template a short admission note so claim extraction has prose to chew on.

    The note paraphrases the structured fields rather than introducing new
    information, so any claim the model extracts should be verifiable against
    the FHIR bundle.
    """
    age = admission.get("admission_age", "unknown")
    gender = (admission.get("gender") or "").upper()
    gender_word = {"F": "female", "M": "male"}.get(gender, "patient")
    admit_type = (admission.get("admission_type") or "admission").lower()
    admit_date = _admit_date(admission) or "the admission date"
    disch_date = _disch_date(admission) or "the discharge date"
    icu_units = ", ".join(admission.get("icu_units_visited") or []) or "no ICU"
    icu_los = admission.get("total_icu_los_days", "0")
    expired = admission.get("hospital_expire_flag") == "1"

    lines: list[str] = []
    lines.append(f"Hospital course summary for admission {admission.get('hadm_id')}.")
    lines.append(
        f"{age}-year-old {gender_word} admitted on {admit_date} via {admit_type} pathway "
        f"and discharged on {disch_date} ({'deceased' if expired else 'alive'}). "
        f"Time in ICU: {icu_los} days across units: {icu_units}."
    )

    diags = admission.get("all_diagnoses") or []
    if diags:
        principal = diags[0].get("diagnosis_title", "")
        lines.append(f"Principal diagnosis: {principal}.")
        if len(diags) > 1:
            others = "; ".join(d.get("diagnosis_title", "") for d in diags[1:])
            lines.append(f"Additional diagnoses noted during this admission: {others}.")

    procs = admission.get("all_procedures") or []
    if procs:
        proc_text = "; ".join(p.get("procedure_title", "") for p in procs)
        lines.append(f"Procedures performed: {proc_text}.")

    if expired:
        lines.append("Patient expired during this admission.")
    else:
        loc = admission.get("discharge_location")
        if loc:
            lines.append(f"Discharge disposition: {loc}.")

    return "\n\n".join(lines) + "\n"


def admission_to_reference_summary(admission: dict[str, Any]) -> str:
    """Deterministic gold-standard summary built from the structured fields.

    Used as the reference for ROUGE / BERTScore / entity recall scoring.
    """
    diags = admission.get("all_diagnoses") or []
    procs = admission.get("all_procedures") or []
    age = admission.get("admission_age", "unknown")
    gender_word = {"F": "female", "M": "male"}.get(
        (admission.get("gender") or "").upper(), "patient"
    )
    admit_date = _admit_date(admission) or ""
    disch_date = _disch_date(admission) or ""
    expired = admission.get("hospital_expire_flag") == "1"

    out: list[str] = []
    out.append("## HPI")
    out.append(
        f"{age}-year-old {gender_word} with hospital admission from {admit_date} to {disch_date}."
    )

    out.append("## Active Problems")
    if diags:
        for d in diags:
            out.append(f"- {d.get('diagnosis_title', '')}.")
    else:
        out.append("- None documented.")

    out.append("## Procedures")
    if procs:
        for p in procs:
            out.append(f"- {p.get('procedure_title', '')}.")
    else:
        out.append("- None documented.")

    out.append("## Disposition")
    if expired:
        out.append("Patient expired during this admission.")
    else:
        loc = admission.get("discharge_location") or "discharged"
        out.append(f"Discharge disposition: {loc}.")

    return "\n".join(out) + "\n"


def gold_entities(admission: dict[str, Any]) -> set[str]:
    """The gold entity set for entity precision/recall.

    Each entity is the lower-cased title of a diagnosis or procedure.
    """
    out: set[str] = set()
    for d in admission.get("all_diagnoses") or []:
        title = (d.get("diagnosis_title") or "").strip().lower()
        if title:
            out.add(title)
    for p in admission.get("all_procedures") or []:
        title = (p.get("procedure_title") or "").strip().lower()
        if title:
            out.add(title)
    return out


def materialize_case(
    admission: dict[str, Any],
    work_dir: Path,
) -> tuple[Path, Path]:
    """Write bundle.json and notes/<case_id>.txt under work_dir/<case_id>/.

    Returns (bundle_path, notes_dir). Uses the synthetic note; for real notes
    use :func:`materialize_real_case`.
    """
    cid = case_id_for(admission)
    case_dir = work_dir / cid
    notes_dir = case_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = case_dir / "bundle.json"
    bundle_path.write_text(
        json.dumps(admission_to_fhir_bundle(admission), indent=2), encoding="utf-8"
    )
    note_path = notes_dir / f"{cid}.txt"
    note_path.write_text(admission_to_synthetic_note(admission), encoding="utf-8")

    return bundle_path, notes_dir


# ---------------------------------------------------------------------------
# Real MIMIC-III-Ext-Notes loader
#
# Joins notes.csv + labels.csv with the structured EHR JSON by hadm_id so we
# can benchmark the pipeline on actual clinical text with expert-annotated
# clinical concepts as the gold entity set.
# ---------------------------------------------------------------------------


@dataclass
class GoldConcept:
    """A clinician-annotated concept from labels.csv that we treat as gold.

    Filtered upstream to detection=yes, encounter=yes, negation=no by default
    (overridable in :func:`load_real_cases`).
    """

    concept: str
    trigger_word: str
    semtypes: str
    start: int
    end: int


@dataclass
class RealCase:
    case_id: str
    hadm_id: str
    subject_id: str
    note_row_id: str
    note_text: str
    admission: dict[str, Any]
    gold_concepts: list[GoldConcept]


def _load_notes_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_labels_csv(path: Path) -> dict[str, list[GoldConcept]]:
    """Group all label rows by row_id (no filtering yet)."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[row["row_id"]].append(row)

    out: dict[str, list[GoldConcept]] = {}
    for row_id, rows in grouped.items():
        out[row_id] = [
            GoldConcept(
                concept=(r.get("concept") or "").strip(),
                trigger_word=(r.get("trigger_word") or "").strip(),
                semtypes=(r.get("semtypes") or "").strip(),
                start=int(r.get("start") or 0),
                end=int(r.get("end") or 0),
            )
            for r in rows
        ]
    return out


def _filter_gold(
    rows: list[dict[str, str]],
    *,
    require_detection: bool,
    require_encounter: bool,
    drop_negated: bool,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        if require_detection and (r.get("detection") or "").strip().lower() != "yes":
            continue
        if require_encounter and (r.get("encounter") or "").strip().lower() != "yes":
            continue
        if drop_negated and (r.get("negation") or "").strip().lower() == "yes":
            continue
        out.append(r)
    return out


def load_real_cases(
    *,
    ehr_json_path: Path,
    notes_csv_path: Path,
    labels_csv_path: Path,
    require_detection: bool = True,
    require_encounter: bool = True,
    drop_negated: bool = True,
) -> list[RealCase]:
    """Load real MIMIC-III-Ext-Notes cases joined with the structured EHR.

    Returns one :class:`RealCase` per note row (so multiple cases per
    admission are possible if a hadm_id has more than one note).
    """
    admissions = load_admissions(ehr_json_path)
    by_hadm = {str(a["hadm_id"]): a for a in admissions}

    note_rows = _load_notes_csv(notes_csv_path)

    grouped_label_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    with labels_csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped_label_rows[row["row_id"]].append(row)

    cases: list[RealCase] = []
    for n in note_rows:
        hadm_id = str(n.get("hadm_id"))
        adm = by_hadm.get(hadm_id)
        if adm is None:
            continue
        row_id = str(n["row_id"])
        label_rows = grouped_label_rows.get(row_id, [])
        kept = _filter_gold(
            label_rows,
            require_detection=require_detection,
            require_encounter=require_encounter,
            drop_negated=drop_negated,
        )
        gold = [
            GoldConcept(
                concept=(r.get("concept") or "").strip(),
                trigger_word=(r.get("trigger_word") or "").strip(),
                semtypes=(r.get("semtypes") or "").strip(),
                start=int(r.get("start") or 0),
                end=int(r.get("end") or 0),
            )
            for r in kept
        ]
        case_id = f"mimic-{n['subject_id']}-{hadm_id}-{row_id}"
        cases.append(
            RealCase(
                case_id=case_id,
                hadm_id=hadm_id,
                subject_id=str(n["subject_id"]),
                note_row_id=row_id,
                note_text=n["text"],
                admission=adm,
                gold_concepts=gold,
            )
        )
    return cases


def materialize_real_case(case: RealCase, work_dir: Path) -> tuple[Path, Path]:
    """Write the FHIR bundle (from the structured EHR) and the actual note text."""
    case_dir = work_dir / case.case_id
    notes_dir = case_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = case_dir / "bundle.json"
    bundle_path.write_text(
        json.dumps(admission_to_fhir_bundle(case.admission), indent=2),
        encoding="utf-8",
    )
    note_path = notes_dir / f"row{case.note_row_id}.txt"
    note_path.write_text(case.note_text, encoding="utf-8")

    return bundle_path, notes_dir


def real_case_reference_summary(case: RealCase) -> str:
    """Reference summary for a real case.

    Combines the gold clinical concepts (from labels.csv) with the structured
    EHR diagnoses + procedures into a short Markdown reference for ROUGE /
    BERTScore comparison.
    """
    out: list[str] = []
    age = case.admission.get("admission_age", "unknown")
    gender_word = {"F": "female", "M": "male"}.get(
        (case.admission.get("gender") or "").upper(), "patient"
    )
    out.append("## HPI")
    out.append(f"{age}-year-old {gender_word}.")

    out.append("## Active Problems")
    if case.gold_concepts:
        seen: set[str] = set()
        for g in case.gold_concepts:
            key = g.concept.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(f"- {g.concept}.")
    else:
        out.append("- None documented in the source note.")

    diags = case.admission.get("all_diagnoses") or []
    if diags:
        out.append("## EHR Diagnoses")
        for d in diags:
            out.append(f"- {d.get('diagnosis_title', '')}.")

    procs = case.admission.get("all_procedures") or []
    if procs:
        out.append("## Procedures")
        for p in procs:
            out.append(f"- {p.get('procedure_title', '')}.")

    expired = case.admission.get("hospital_expire_flag") == "1"
    out.append("## Disposition")
    if expired:
        out.append("Patient expired during this admission.")
    else:
        loc = case.admission.get("discharge_location") or "discharged"
        out.append(f"Discharge disposition: {loc}.")
    return "\n".join(out) + "\n"


def real_case_gold_entities(case: RealCase) -> set[str]:
    """Gold entity set used for entity precision/recall on a real case.

    Combines the labeled clinical concepts (the most important signal for
    entity-level reasoning over the *note*) with the structured diagnosis and
    procedure titles from the matching EHR.
    """
    out: set[str] = set()
    for g in case.gold_concepts:
        if g.concept:
            out.add(g.concept.strip().lower())
    for d in case.admission.get("all_diagnoses") or []:
        title = (d.get("diagnosis_title") or "").strip().lower()
        if title:
            out.add(title)
    for p in case.admission.get("all_procedures") or []:
        title = (p.get("procedure_title") or "").strip().lower()
        if title:
            out.add(title)
    return out
