"""Shared pydantic schemas for the pipeline's intermediate JSON artifacts.

These types are also used to derive JSON Schemas for Ollama's
structured-output mode so that LLM stages return predictable shapes.
"""

from __future__ import annotations

from typing import Any, Literal

import logging as _logging

from pydantic import BaseModel, Field, field_validator

_schema_log = _logging.getLogger(__name__)

EvidenceKind = Literal[
    "condition",
    "medication",
    "observation",
    "allergy",
    "procedure",
    "encounter",
    "note_sentence",
]

ClaimType = Literal[
    "diagnosis",
    "medication",
    "lab",
    "vital",
    "procedure",
    "allergy",
    "plan",
]

VerificationStatus = Literal["verified", "contradicted", "unsupported"]

SectionKey = Literal[
    "hpi",
    "active_problems",
    "medications",
    "labs",
    "vitals",
    "plan",
]


class Code(BaseModel):
    system: str | None = None
    code: str | None = None
    display: str | None = None


class Evidence(BaseModel):
    """A single atomic, citable piece of evidence."""

    id: str
    kind: EvidenceKind
    display: str
    code: Code | None = None
    value: str | None = None
    unit: str | None = None
    effective: str | None = None
    source_ref: str
    text: str | None = None


class EvidenceStore(BaseModel):
    case_id: str
    evidence: list[Evidence]
    by_code: dict[str, list[str]] = Field(default_factory=dict)
    by_display: dict[str, list[str]] = Field(default_factory=dict)
    by_date: dict[str, list[str]] = Field(default_factory=dict)


# Aliases for LLM-hallucinated type values → nearest valid ClaimType
_CLAIM_TYPE_ALIASES: dict[str, str] = {
    "action": "plan",
    "finding": "diagnosis",
    "symptom": "diagnosis",
    "condition": "diagnosis",
    "problem": "diagnosis",
    "treatment": "procedure",
    "intervention": "procedure",
    "drug": "medication",
    "med": "medication",
    "test": "lab",
    "laboratory": "lab",
    "measurement": "vital",
    "sign": "vital",
    "sign_symptom": "vital",
}

_VALID_CLAIM_TYPES: frozenset[str] = frozenset(
    {"diagnosis", "medication", "lab", "vital", "procedure", "allergy", "plan"}
)


class Claim(BaseModel):
    """A structured clinical assertion extracted from notes."""

    claim_id: str
    type: ClaimType
    subject: str = "patient"
    predicate: str
    value: str | None = None
    time_ref: str | None = None
    source_span: str

    @field_validator("type", mode="before")
    @classmethod
    def coerce_claim_type(cls, v: object) -> str:
        """Accept the 7 defined ClaimType values; map common LLM aliases to the
        nearest valid type and fall back to 'plan' for anything else."""
        if not isinstance(v, str):
            return "plan"
        lower = v.lower().strip()
        if lower in _VALID_CLAIM_TYPES:
            return lower
        mapped = _CLAIM_TYPE_ALIASES.get(lower, "plan")
        _schema_log.warning(
            "Unknown claim type %r — coerced to %r", v, mapped
        )
        return mapped


class ClaimList(BaseModel):
    claims: list[Claim]


class Verification(BaseModel):
    claim_id: str
    status: VerificationStatus
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


class VerificationResult(BaseModel):
    verifications: list[Verification]


class ContextSuggestion(BaseModel):
    description: str
    suggested_claim: Claim | None = None


class ContextReport(BaseModel):
    missing_context: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    suggested_supporting_facts: list[ContextSuggestion] = Field(default_factory=list)


class FactSheetEntry(BaseModel):
    text: str
    evidence_ids: list[str]


class FactSheet(BaseModel):
    case_id: str
    sections: dict[str, list[FactSheetEntry]]


class CheckViolation(BaseModel):
    sentence: str
    rule: str
    detail: str


class CheckReport(BaseModel):
    passed: bool
    violations: list[CheckViolation] = Field(default_factory=list)
    sentence_count: int = 0
    cited_evidence_ids: list[str] = Field(default_factory=list)


class ReviewConcern(BaseModel):
    sentence: str
    severity: Literal["low", "medium", "high"]
    reason: str


class ReviewRevision(BaseModel):
    original: str
    suggested: str
    reason: str


class ReviewReport(BaseModel):
    concerns: list[ReviewConcern] = Field(default_factory=list)
    recommended_revisions: list[ReviewRevision] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when there are no high-severity concerns.

        Severity thresholds:
          high   → fails review (clinically significant misstatement)
          medium → noted but does not block (imprecision, abbreviation)
          low    → advisory only
        """
        return not any(c.severity == "high" for c in self.concerns)

    @property
    def concern_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for c in self.concerns:
            counts[c.severity] = counts.get(c.severity, 0) + 1
        return counts


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON Schema usable as Ollama's `format` argument."""
    return model.model_json_schema()
