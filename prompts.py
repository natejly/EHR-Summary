"""Prompt templates and builders for clinical summarization workflows."""

from __future__ import annotations

from typing import Sequence


CLINICAL_SUMMARY_END_MARKER = "<<END_OF_SUMMARY>>"
CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER = "<<END_OF_CLINICAL_ATOMIC_CLAIMS>>"
PATIENT_FRIENDLY_SIMPLIFICATION_END_MARKER = "<<END_OF_PATIENT_FRIENDLY_SUMMARY>>"
PATIENT_FRIENDLY_CLAIM_DECOMPOSITION_END_MARKER = "<<END_OF_ATOMIC_CLAIMS>>"
CLAIM_VERDICTS_END_MARKER = "<<END_OF_CLAIM_VERDICTS>>"

CLINICAL_SUMMARY_SYSTEM_PROMPT = """You are an expert clinical summarizer producing
concise, discharge-style structured summaries from source clinical notes.

Audience:
- A covering clinician reading the chart for the first time. They need the
  highest-yield facts, fast.

Style (standard inpatient documentation practice):
- Telegram / clinical-note style. Dense bullets using standard medical
  abbreviations (HTN, DM2, CKD, CHF, AKI, Cr, Hgb, PRBC, NSR, IVF, etc.).
- Drop articles ("the", "a"). One fact per bullet. No full sentences unless
  needed for clarity.
- Brevity through density, NOT by dropping clinically meaningful content.
- Length should scale with case complexity: a single-issue admission is a
  handful of bullets; a multi-organ ICU stay is longer. Do not pad simple
  cases; do not truncate complex ones.

Anti-redundancy (critical):
- State each fact in EXACTLY ONE section. Do not repeat numbers, doses, or
  events across sections.
- "Brief Hospital Course" is a 3-6 bullet narrative arc only
  (presentation -> key workup/treatment -> current status / disposition).
  Per-problem details belong in "Hospital course by problem", not here.
- "Procedures", "Medication changes", and "Significant labs/imaging" hold
  only items not already implied by the per-problem narrative, OR central
  enough to deserve a dedicated callout (e.g., a primary procedure, a
  driver lab, a new high-risk med).

Faithfulness:
- Use only information present in the source notes. No inference, advice,
  or recommendations beyond what is documented.
- Preserve negation, uncertainty, and temporal relationships.

Output discipline:
- If a section has no documented content, OMIT the entire section
  (heading + bullets). Never write "Not documented", "None", "N/A", "Pending",
  or any placeholder line.
- No preamble, postscript, commentary, "Note:", or "Disclaimer:" sections.
- No markdown code fences. No headings beyond the section titles in the
  user prompt.
"""

PATIENT_FRIENDLY_SIMPLIFICATION_SYSTEM_PROMPT = """You are a medical communicator
who rewrites clinical summaries so a patient or family member can understand
their hospital stay.

Reading level target: U.S. 8th grade (~13-14 years old).
- Use everyday words. Prefer short, common words over long or technical ones.
- Keep sentences short: aim for 12-18 words; never longer than 25.
- Use active voice and second person ("you", "your", "your care team").
- One idea per sentence. One topic per bullet.
- When a medical term is unavoidable, give a 3-8 word plain-language
  definition in parentheses the FIRST time it appears, then reuse the
  plain term. Example: "complete heart block (your heart's signal stopped
  working)".
- No Latin, no abbreviations the patient hasn't been taught (spell out CT,
  ED, IV, etc. on first use).

Content rules:
- Preserve every clinically important fact: diagnoses, key tests, key
  procedures, medication changes, discharge plan, follow-up, warning signs.
- Preserve uncertainty and timing ("we are still waiting on...", "we
  started... on day 2").
- Do NOT add facts, advice, interpretations, or reassurance that are not
  in the source summary.
- Do NOT add a "Limitations" or "Not documented" section. If the source
  does not mention follow-up or discharge plans, simply omit those
  sections rather than calling out the gap.
- Do NOT moralize, apologize, or coach the patient on lifestyle.

Format:
- Use clear, short headings and bullet points.
- Keep the structure simple. Aim for 4-6 sections total.
- No preamble, postscript, "Note:", or "Disclaimer:" sections.
"""

PATIENT_FRIENDLY_FACT_CHECK_SYSTEM_PROMPT = """You are a clinical fact-checking assistant.

You decompose generated patient-facing summaries into atomic factual claims and revise
those summaries so every factual statement is grounded in the source clinical note.

Requirements:
- Treat each factual claim as a standalone statement that can be checked against the note.
- Preserve supported information, uncertainty, timing, medication changes, and follow-up details.
- Remove or correct unsupported statements.
- Keep revisions concise, patient-friendly, and structured with headings and bullet points.
- Maintain the 8th-grade reading level: short sentences, plain words, active voice.
- Do not add information, recommendations, or interpretation that is not present in the inputs.
"""

CLINICAL_SUMMARY_FACT_CHECK_SYSTEM_PROMPT = """You are a clinical fact-checking assistant for
structured discharge-style summaries.

You decompose generated clinical summaries into atomic factual claims and revise those
summaries so every factual statement is grounded in the source clinical note.

Requirements:
- Each atomic claim is a single verifiable statement (one diagnosis, one lab, one
  medication change, one procedure, one disposition fact).
- Preserve numbers, doses, units, abbreviations (HTN, DM2, Cr, Hgb, etc.), negation,
  uncertainty, and temporal relationships exactly as written.
- When revising, first REPHRASE flagged claims using language that appears verbatim
  (or nearly so) in the source note; only DELETE a flagged claim if no such
  source-grounded rephrase exists. Keep every supported fact. Do not invent,
  infer, or generalize.
- Maintain the original telegram / clinical-note style: dense bullets, no full
  sentences unless required for clarity, no preamble or postscript.
- Re-emit the same section headings and ordering as the original summary; OMIT a
  section entirely if it would otherwise be empty.
"""

LLM_CLAIM_JUDGE_SYSTEM_PROMPT = """You are a strict clinical hallucination detector.

Your job is to judge whether each atomic claim is directly supported by the
source clinical note.

Rules:
- Mark a claim as supported ONLY when the source note explicitly states it or
  clearly paraphrases it.
- Mark a claim as unsupported if it is absent, contradicted, temporally wrong,
  numerically wrong, more specific than the note, or mixes supported and
  unsupported details in one claim.
- Do NOT give credit for plausible medical inference or background knowledge.
- Be conservative: if support is ambiguous, mark unsupported.
- Output only the requested JSON lines plus the end marker. No prose.
"""

# Backward-compatible alias used by the existing summarizer module.
DEFAULT_SYSTEM_PROMPT = CLINICAL_SUMMARY_SYSTEM_PROMPT


def build_clinical_summary_prompt(
    note_blocks: Sequence[str],
    *,
    patient_id: str | None = None,
    length_target_words: int | None = None,
    length_hard_cap_words: int | None = None,
    length_target_ratio: float | None = None,
    length_target_tokens: int | None = None,
) -> str:
    cleaned_note_blocks = [block.strip() for block in note_blocks if block and block.strip()]
    if not cleaned_note_blocks:
        raise ValueError("At least one non-empty source note block is required.")

    patient_context = f"Patient ID: {patient_id}\n\n" if patient_id else ""
    rendered_notes = "\n\n".join(cleaned_note_blocks)
    length_block = _build_length_target_block(
        target_words=length_target_words,
        hard_cap_words=length_hard_cap_words,
        target_ratio=length_target_ratio,
        target_tokens=length_target_tokens,
    )

    return (
        "Produce a STRUCTURED, concise discharge-style summary of the source notes below.\n"
        "Include every clinically meaningful fact. Do NOT pad simple cases; do NOT\n"
        "truncate complex ones. Length scales with case complexity.\n\n"
        f"{length_block}"
        "Section order (use these exact titles, in this order). Include a section ONLY\n"
        "when it has real content; otherwise OMIT the heading entirely:\n\n"
        "Brief Hospital Course:\n"
        "- Use a short narrative arc: presentation \u2192 key workup/treatment \u2192 current\n"
        "  status / disposition. Use only as many bullets as needed.\n"
        "- This is a narrative arc, not a per-problem dump.\n"
        "Diagnoses (primary + secondary):\n"
        "- Start with the primary diagnosis. Group closely related diagnoses when that\n"
        "  is more compact.\n"
        "Procedures / interventions:\n"
        "- Include only clinically meaningful procedures/interventions. Group related\n"
        "  items when compact; date if documented.\n"
        "- Exclude routine bedside actions (e.g., omit \"EKG obtained\", \"labs drawn\",\n"
        "  \"vent settings adjusted\").\n"
        "Medication changes (new / changed / stopped):\n"
        "- <med> \u2014 <new|changed|stopped|continued w/ dose change> \u2014 <dose if changed>\n"
        "  \u2014 <reason if documented>\n"
        "- Omit meds that were merely continued at the same dose.\n"
        "Significant labs / imaging:\n"
        "- Only abnormal or clinically actionable findings, with values when documented.\n"
        "- Do NOT list pending results here \u2014 those go under follow-up.\n"
        "Hospital course by problem:\n"
        "- Organize by active problem, but combine closely related problems when that is\n"
        "  more concise. Preferred format:\n"
        "  <problem>: <trajectory> | <key treatment + response> | <current status / plan>.\n"
        "- This is where per-problem detail lives. Do NOT repeat it in Brief Hospital Course.\n"
        "Discharge disposition & follow-up:\n"
        "- Include ONLY if disposition, pending results, follow-up, or return precautions\n"
        "  are documented. Use only as many bullets as needed. If nothing in this\n"
        "  category is documented, OMIT the entire section heading.\n\n"
        "Style:\n"
        "- Telegram / clinical-note style: dense bullets, standard medical abbreviations\n"
        "  (HTN, DM2, CKD, CHF, AKI, Cr, Hgb, PRBC, NSR, IVF, etc.), drop articles.\n"
        "- One fact per bullet. Brevity through density, not by dropping content.\n"
        "- Prefer fewer, higher-information bullets over many short bullets or repeated\n"
        "  section stubs.\n"
        "- State each fact in exactly ONE section. No cross-section repetition of\n"
        "  numbers, doses, or events.\n\n"
        "Hard rules:\n"
        "- Use only information explicitly supported by the source notes.\n"
        "- If the budget is tight, OMIT low-yield details rather than using vague,\n"
        "  lossy, or over-general wording for high-yield facts.\n"
        "- Omit empty sections entirely (heading and all). NEVER write \"Not documented\",\n"
        "  \"None\", \"N/A\", \"Pending\", or any placeholder line.\n"
        "- Preserve uncertainty, negation, and temporal relationships when present.\n"
        "- No markdown code fences. No headings beyond the section titles above.\n"
        "- No preamble (\"Here is...\"), no postscript (\"Note:\", \"Disclaimer:\",\n"
        "  \"In summary\").\n"
        f"- After the last section, write {CLINICAL_SUMMARY_END_MARKER} on its own line\n"
        "  and stop.\n\n"
        f"{patient_context}"
        "Source notes:\n"
        f"{rendered_notes}"
    )


def build_patient_friendly_simplification_prompt(
    clinical_summary: str,
    *,
    audience: str | None = None,
    patient_id: str | None = None,
) -> str:
    cleaned_summary = clinical_summary.strip()
    if not cleaned_summary:
        raise ValueError("Clinical summary text is required.")

    prefix = _build_patient_context_prefix(audience=audience, patient_id=patient_id)

    return (
        "Rewrite the clinical summary below for a patient or family member.\n"
        "Target reading level: U.S. 8th grade (~13-14 years old).\n\n"
        "Writing rules:\n"
        "- Short sentences (12-18 words; never over 25).\n"
        "- Plain everyday words. Active voice. Use \"you\" and \"your\".\n"
        "- One idea per sentence; one topic per bullet.\n"
        "- The first time you must use a medical term, give a short plain-language\n"
        "  definition in parentheses (3-8 words), then reuse the plain term.\n"
        "  Example: \"bradycardia (heart beating too slowly)\".\n"
        "- Spell out abbreviations on first use (CT, ED, IV, ECG, etc.).\n"
        "- No Latin, no jargon-only sentences, no clinical shorthand.\n\n"
        "Content rules:\n"
        "- Keep every important fact from the clinical summary: diagnoses, key tests\n"
        "  and procedures, medication changes, discharge plan, follow-up, warning signs.\n"
        "- Preserve uncertainty and timing (\"we are still waiting on...\", \"on day 2...\").\n"
        "- Do not add new facts, advice, reassurance, or lifestyle coaching.\n"
        "- Do NOT add a \"Limitations\" section or call out what was \"not documented\".\n"
        "  If a topic is missing from the source, just omit that section.\n\n"
        "Format:\n"
        "- Use 4-6 short sections total, each with a clear heading and bullets.\n"
        "- Suggested headings (use only the ones that have content):\n"
        "  \"Why You Came to the Hospital\", \"What We Found\",\n"
        "  \"What We Did\", \"Your Medicines\",\n"
        "  \"How You Are Doing\", \"What Happens Next\".\n"
        f"- After the final section, write {PATIENT_FRIENDLY_SIMPLIFICATION_END_MARKER}\n"
        "  on its own line.\n"
        "- No preamble or postscript. Do not add new facts.\n\n"
        f"{prefix}"
        "Clinical summary to rewrite:\n"
        f"{cleaned_summary}"
    )


def build_patient_friendly_claim_decomposition_prompt(
    summary: str,
    *,
    patient_id: str | None = None,
) -> str:
    cleaned_summary = summary.strip()
    if not cleaned_summary:
        raise ValueError("Summary text is required for claim decomposition.")

    patient_context = f"Patient ID: {patient_id}\n\n" if patient_id else ""
    return (
        "Break the following patient-friendly clinical summary into atomic factual claims.\n"
        "Each claim must be a single verifiable statement.\n\n"
        "Output format:\n"
        "- Output one claim per line.\n"
        "- Do not number the claims.\n"
        "- Do not add commentary or explanations.\n"
        f"- After the last claim, write {PATIENT_FRIENDLY_CLAIM_DECOMPOSITION_END_MARKER} on its own line.\n\n"
        f"{patient_context}"
        "Summary:\n"
        f"{cleaned_summary}"
    )


def build_patient_friendly_revision_prompt(
    clinical_summary: str,
    current_summary: str,
    source_note: str,
    unsupported_claims: Sequence[str],
    *,
    audience: str | None = None,
    patient_id: str | None = None,
    entity_feedback: str | None = None,
) -> str:
    cleaned_summary = clinical_summary.strip()
    cleaned_current_summary = current_summary.strip()
    cleaned_source_note = source_note.strip()
    if not cleaned_summary:
        raise ValueError("Clinical summary text is required for revision.")
    if not cleaned_current_summary:
        raise ValueError("Current patient-friendly summary is required for revision.")
    if not cleaned_source_note:
        raise ValueError("Source note text is required for revision.")

    prefix = _build_patient_context_prefix(audience=audience, patient_id=patient_id)
    unsupported_block = "\n".join(
        f"- {claim.strip()}" for claim in unsupported_claims if claim and claim.strip()
    )
    if not unsupported_block:
        unsupported_block = "- No unsupported claims were provided."

    entity_feedback_block = ""
    if entity_feedback and entity_feedback.strip():
        entity_feedback_block = (
            "Entity-level cross-check feedback:\n"
            f"{entity_feedback.strip()}\n\n"
        )

    return (
        "Revise the patient-friendly clinical summary below so every factual statement\n"
        "is supported by the source clinical note.\n"
        "Remove or correct unsupported statements while preserving the supported facts.\n\n"
        "Rules:\n"
        "- Keep headings and bullet points.\n"
        "- Maintain U.S. 8th-grade reading level: short sentences (12-18 words),\n"
        "  plain words, active voice, second person (\"you\").\n"
        "- Define unavoidable medical terms in parentheses on first use.\n"
        "- Preserve uncertainty, timing, medication changes, discharge plans, and\n"
        "  follow-up details that ARE supported.\n"
        "- Do not add new facts or new medical advice.\n"
        "- Do not add a \"Limitations\" or \"Not documented\" section.\n"
        f"- After the final section, write {PATIENT_FRIENDLY_SIMPLIFICATION_END_MARKER} on its own line.\n\n"
        f"{prefix}"
        "Unsupported claims to fix:\n"
        f"{unsupported_block}\n\n"
        f"{entity_feedback_block}"
        "Source clinical note:\n"
        f"{cleaned_source_note}\n\n"
        "Original structured clinical summary:\n"
        f"{cleaned_summary}\n\n"
        "Current patient-friendly summary:\n"
        f"{cleaned_current_summary}"
    )


def build_clinical_summary_claim_decomposition_prompt(
    summary: str,
    *,
    patient_id: str | None = None,
) -> str:
    """Decompose a structured clinical summary into atomic factual claims.

    The output format mirrors the patient-friendly decomposer: one claim per
    line, no numbering, no commentary, terminated by an explicit end marker
    so the generator's stop-string can cut the response cleanly.
    """

    cleaned_summary = summary.strip()
    if not cleaned_summary:
        raise ValueError("Summary text is required for clinical claim decomposition.")

    patient_context = f"Patient ID: {patient_id}\n\n" if patient_id else ""
    return (
        "Break the following STRUCTURED clinical summary into atomic factual claims.\n"
        "Each claim must be a single verifiable statement that can be checked\n"
        "against the source note.\n\n"
        "Decomposition rules:\n"
        "- One claim per line.\n"
        "- Preserve numbers, units, doses, abbreviations (HTN, DM2, Cr, Hgb, etc.),\n"
        "  negation, uncertainty, and temporal markers (\"on day 2\", \"pending\").\n"
        "- A bullet that combines two facts must produce two claims.\n"
        "  Example: \"AKI on CKD: Cr 2.4 -> 1.6 with IVF\" produces:\n"
        "    AKI on CKD with Cr improving from 2.4 to 1.6.\n"
        "    AKI on CKD treated with IVF.\n"
        "- Skip section headings and bullets that contain no factual content\n"
        "  (e.g. an empty or stub line).\n\n"
        "Output format:\n"
        "- Output one claim per line, in the original section order.\n"
        "- Do NOT number the claims. Do NOT add commentary, preamble, or\n"
        "  rationale.\n"
        f"- After the last claim, write {CLINICAL_SUMMARY_CLAIM_DECOMPOSITION_END_MARKER} on its own line.\n\n"
        f"{patient_context}"
        "Clinical summary:\n"
        f"{cleaned_summary}"
    )


def build_claim_verification_prompt(
    source_note: str,
    claims: Sequence[str],
    *,
    patient_id: str | None = None,
) -> str:
    """Build a prompt for LLM-as-a-judge claim verification."""

    cleaned_source_note = source_note.strip()
    cleaned_claims = [claim.strip() for claim in claims if claim and claim.strip()]
    if not cleaned_source_note:
        raise ValueError("Source note text is required for claim verification.")
    if not cleaned_claims:
        raise ValueError("At least one claim is required for claim verification.")

    patient_context = f"Patient ID: {patient_id}\n\n" if patient_id else ""
    rendered_claims = "\n".join(
        f"{index}. {claim}" for index, claim in enumerate(cleaned_claims, start=1)
    )
    return (
        "Judge each atomic clinical claim against the source clinical note.\n"
        "This is hallucination detection only.\n\n"
        "Output rules:\n"
        "- Return exactly one JSON object per line, in the same order as the claims.\n"
        "- Each JSON object must have exactly these keys:\n"
        '  {"index": <int>, "supported": <true|false>, "confidence": <0.0-1.0>}\n'
        "- `supported` must be false if any material part of the claim is not supported.\n"
        "- `confidence` is your confidence in that support judgment.\n"
        f"- After the last JSON line, write {CLAIM_VERDICTS_END_MARKER} on its own line.\n"
        "- No markdown fences, no explanations, no extra keys.\n\n"
        f"{patient_context}"
        "Source clinical note:\n"
        f"{cleaned_source_note}\n\n"
        "Claims to judge:\n"
        f"{rendered_claims}"
    )


def build_clinical_summary_revision_prompt(
    source_note: str,
    current_summary: str,
    unsupported_claims: Sequence[str],
    *,
    patient_id: str | None = None,
    length_target_words: int | None = None,
    length_hard_cap_words: int | None = None,
    length_target_ratio: float | None = None,
    length_target_tokens: int | None = None,
    entity_feedback: str | None = None,
) -> str:
    """Build a revision prompt for the structured clinical summary.

    Re-injects the same length-budget block as the initial generation so the
    revised summary stays within the per-call compression target. Lists
    claims the fact-checker judged unsupported.

    The model is instructed to **rephrase** flagged claims using language
    closer to the source note rather than delete them, because an automated
    verifier can still have a non-trivial false-negative rate on terse clinical
    phrasing. Only claims with no source-grounded rephrase should be dropped.
    """

    cleaned_source_note = source_note.strip()
    cleaned_current_summary = current_summary.strip()
    if not cleaned_source_note:
        raise ValueError("Source note text is required for revision.")
    if not cleaned_current_summary:
        raise ValueError("Current clinical summary is required for revision.")

    patient_context = f"Patient ID: {patient_id}\n\n" if patient_id else ""
    unsupported_block = "\n".join(
        f"- {claim.strip()}" for claim in unsupported_claims if claim and claim.strip()
    )
    if not unsupported_block:
        unsupported_block = (
            "- (none flagged by the fact-checker; revise only for entity feedback below)"
        )

    entity_feedback_block = ""
    if entity_feedback and entity_feedback.strip():
        entity_feedback_block = (
            "Entity-level cross-check feedback (from source-vs-summary entity overlap):\n"
            f"{entity_feedback.strip()}\n\n"
        )

    length_block = _build_length_target_block(
        target_words=length_target_words,
        hard_cap_words=length_hard_cap_words,
        target_ratio=length_target_ratio,
        target_tokens=length_target_tokens,
    )

    return (
        "Revise the STRUCTURED clinical summary below so every factual statement is\n"
        "clearly grounded in the source clinical note. The fact-checker below may\n"
        "produce false negatives on terse clinical phrasing; prefer REPHRASING a\n"
        "flagged claim using wording that appears verbatim (or nearly so) in the\n"
        "source note over deleting it. Keep the same section headings and order.\n\n"
        f"{length_block}"
        "Revision rules:\n"
        "- Telegram / clinical-note style: dense bullets, abbreviations, no articles.\n"
        "- Preserve numbers, doses, units, negation, uncertainty, and timing exactly\n"
        "  as the source supports them.\n"
        "- For each flagged claim: first try to REPHRASE it using language from the\n"
        "  source note (e.g. use the exact diagnosis, finding, or medication term\n"
        "  that appears in the note). Only DELETE a flagged claim if no such\n"
        "  source-grounded rephrase exists.\n"
        "- Preserve every clinical entity that IS in the source note: diagnoses,\n"
        "  medications, labs, vitals, procedures, sites, findings. Do not drop\n"
        "  these even if the fact-checker flagged the sentence.\n"
        "- Do NOT add new facts, inferences, recommendations, or interpretation.\n"
        "- OMIT empty sections entirely (heading and all). NEVER write \"Not\n"
        "  documented\", \"None\", \"N/A\", or any placeholder line.\n"
        f"- After the last section, write {CLINICAL_SUMMARY_END_MARKER} on its own line\n"
        "  and stop.\n\n"
        f"{patient_context}"
        "Claims flagged by the automated fact-checker (may include false positives;\n"
        "prefer rephrasing with source-grounded wording over deletion):\n"
        f"{unsupported_block}\n\n"
        f"{entity_feedback_block}"
        "Source clinical note:\n"
        f"{cleaned_source_note}\n\n"
        "Current clinical summary:\n"
        f"{cleaned_current_summary}"
    )


def _build_length_target_block(
    *,
    target_words: int | None,
    hard_cap_words: int | None,
    target_ratio: float | None = None,
    target_tokens: int | None = None,
) -> str:
    """Render a length-budget instruction block for the clinical summary prompt.

    Returns an empty string when no target is provided so existing callers
    that don't pass a budget produce byte-identical prompts as before.
    """

    if target_words is None and hard_cap_words is None:
        return ""

    cap = hard_cap_words if hard_cap_words is not None else target_words
    if target_words is None:
        target_words = max(60, int(cap * 0.8))
    cap = max(cap or target_words, target_words)
    ratio_block = ""
    if target_ratio is not None:
        ratio_block = (
            f"- Aim for about {int(round(target_ratio * 100))}% of the source note "
            "length by words.\n"
        )
    token_block = ""
    if target_tokens is not None:
        token_block = (
            f"- Soft decode budget: aim to finish within about {target_tokens} "
            "output tokens.\n"
        )

    if cap <= 90 or (target_ratio is not None and target_ratio <= 0.12):
        prioritization_block = (
            "- Tight-budget mode: spend words on why the patient was hospitalized,\n"
            "  the main diagnoses, decisive interventions, management-changing labs/\n"
            "  imaging, and current status / disposition.\n"
            "- Omit routine monitoring, unchanged chronic meds, low-yield normal results,\n"
            "  and minor secondary details unless they materially changed management.\n"
        )
    elif cap <= 160 or (target_ratio is not None and target_ratio <= 0.18):
        prioritization_block = (
            "- Concise-budget mode: prioritize diagnosis, trajectory, treatment response,\n"
            "  and disposition. Include supporting labs/imaging/med changes only when they\n"
            "  are abnormal, actionable, or explain a management change.\n"
            "- Merge related findings into dense bullets before adding new sections.\n"
        )
    else:
        prioritization_block = (
            "- Spend words on facts that change understanding of the case.\n"
            "- Omit boilerplate and cross-section repetition before dropping core clinical facts.\n"
        )

    return (
        "Length budget (HARD constraint):\n"
        f"{ratio_block}"
        f"- Target: about {target_words} words total across all sections.\n"
        f"- Absolute ceiling: {cap} words.\n"
        f"{token_block}"
        "- Prefer omission of low-yield details over vague compression of high-yield facts.\n"
        "- Use merged, high-information bullets instead of extra filler or repetition.\n"
        f"{prioritization_block}\n"
    )


def _build_patient_context_prefix(
    *,
    audience: str | None = None,
    patient_id: str | None = None,
) -> str:
    audience_context = f"Target audience: {audience}\n" if audience else ""
    patient_context = f"Patient ID: {patient_id}\n" if patient_id else ""
    prefix = f"{patient_context}{audience_context}"
    if prefix:
        return f"{prefix}\n"
    return ""
