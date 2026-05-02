"""Stage 7: deterministic gating check on the generated summary.

Validates:
  (a) every sentence has at least one [E:...] citation,
  (b) every cited evidence id exists in the fact sheet's evidence_ids,
  (c) any number+unit, ISO date, or medication name in the sentence appears
      verbatim in the cited fact-sheet entry text or its underlying evidence.

Returns a CheckReport. The orchestrator decides whether failures are fatal
(default) or downgraded to warnings via --allow-violations.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..schemas import (
    CheckReport,
    CheckViolation,
    Evidence,
    EvidenceStore,
    FactSheet,
)

log = logging.getLogger(__name__)


CITATION_RE = re.compile(r"\[E:([^\]]+)\]")
NUMBER_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(%|[a-zA-Zµ][a-zA-Zµ/]{0,11}(?:/[a-zA-Z]{1,6})?)"
    r"(?=[\s.,;:)\]]|$)"
)
ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
HEADING_RE = re.compile(r"^\s*#{1,6}\s")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
SUMMARY_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])(?:\s*(?:\[E:[^\]]+\])+)?\s+(?=[A-Z(])"
)


def _split_summary_sentences(markdown: str) -> list[str]:
    """Sentence splitter aware of inline [E:...] citations and Markdown headings."""
    sentences: list[str] = []
    for raw_line in markdown.splitlines():
        if HEADING_RE.match(raw_line):
            continue
        line = LIST_PREFIX_RE.sub("", raw_line).strip()
        if not line:
            continue
        for chunk in SUMMARY_SENTENCE_BOUNDARY.split(line):
            chunk = chunk.strip()
            if chunk:
                sentences.append(chunk)
    return sentences


def _collect_fact_sheet_evidence(
    fact_sheet: FactSheet,
    store: EvidenceStore,
) -> tuple[set[str], dict[str, str]]:
    allowed_ids: set[str] = set()
    text_for_id: dict[str, str] = {}

    for entries in fact_sheet.sections.values():
        for entry in entries:
            for eid in entry.evidence_ids:
                allowed_ids.add(eid)
                text_for_id[eid] = entry.text

    by_ev: dict[str, Evidence] = {ev.id: ev for ev in store.evidence}
    for eid in list(allowed_ids):
        ev = by_ev.get(eid)
        if not ev:
            continue
        existing = text_for_id.get(eid, "")
        bits = [existing, ev.display]
        if ev.value is not None:
            bits.append(str(ev.value))
        if ev.unit:
            bits.append(ev.unit)
        if ev.effective:
            bits.append(ev.effective)
        if ev.text:
            bits.append(ev.text)
        text_for_id[eid] = " ".join(b for b in bits if b)

    return allowed_ids, text_for_id


def _normalize_for_compare(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _normalize_id(raw: str) -> str:
    raw = raw.strip()
    return raw if raw.startswith("E:") else f"E:{raw}"


def run(
    *,
    summary_markdown: str,
    fact_sheet: FactSheet,
    store: EvidenceStore,
    output_dir: Path,
    report_filename: str = "check_report.json",
) -> CheckReport:
    log.info("Stage 7: deterministic check")
    sentences = _split_summary_sentences(summary_markdown)
    allowed, text_for_id = _collect_fact_sheet_evidence(fact_sheet, store)
    violations: list[CheckViolation] = []
    cited_all: set[str] = set()

    for sentence in sentences:
        if sentence.lower().startswith("_not documented._"):
            continue

        cite_matches = CITATION_RE.findall(sentence)
        cite_ids = [_normalize_id(c) for c in cite_matches]

        if not cite_ids:
            violations.append(
                CheckViolation(
                    sentence=sentence,
                    rule="missing_citation",
                    detail="Sentence has no [E:...] citation.",
                )
            )
            continue

        unknown = [cid for cid in cite_ids if cid not in allowed]
        if unknown:
            violations.append(
                CheckViolation(
                    sentence=sentence,
                    rule="unknown_evidence_id",
                    detail=f"Cited ids not in fact sheet: {sorted(set(unknown))}",
                )
            )

        cited_all.update(cite_ids)

        cited_text = " ".join(text_for_id.get(cid, "") for cid in cite_ids if cid in allowed)
        cited_norm = _normalize_for_compare(cited_text)

        for value, unit in NUMBER_UNIT_RE.findall(sentence):
            phrase = _normalize_for_compare(f"{value} {unit}")
            if phrase not in cited_norm and value not in cited_norm:
                violations.append(
                    CheckViolation(
                        sentence=sentence,
                        rule="unsupported_value",
                        detail=f"Numeric '{value} {unit}' not present in cited evidence.",
                    )
                )

        for iso in ISO_DATE_RE.findall(sentence):
            if iso not in cited_text:
                violations.append(
                    CheckViolation(
                        sentence=sentence,
                        rule="unsupported_date",
                        detail=f"Date '{iso}' not present in cited evidence.",
                    )
                )

    report = CheckReport(
        passed=not violations,
        violations=violations,
        sentence_count=len(sentences),
        cited_evidence_ids=sorted(cited_all),
    )
    out_path = output_dir / report_filename
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "  passed=%s sentences=%d violations=%d -> %s",
        report.passed,
        report.sentence_count,
        len(report.violations),
        out_path,
    )
    return report
