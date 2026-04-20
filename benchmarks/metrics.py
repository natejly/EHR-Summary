"""Metric helpers: ROUGE, BERTScore (PubMedBERT), and gold entity recall."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd


# --- ROUGE --------------------------------------------------------------------


@dataclass(slots=True)
class RougeRow:
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float


def rouge_scores(
    candidates: Sequence[str],
    references: Sequence[str],
) -> list[RougeRow]:
    """Per-pair ROUGE-1/2/L F1 against the source notes.

    The reference here is the source clinical note, so recall is biased high
    (the summary is a strict subset of the source); F1 is the more meaningful
    aggregate.
    """

    if len(candidates) != len(references):
        raise ValueError(
            "candidates and references must be the same length "
            f"(got {len(candidates)} vs {len(references)})."
        )

    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    rows: list[RougeRow] = []
    for cand, ref in zip(candidates, references):
        result = scorer.score(target=ref, prediction=cand)
        rows.append(
            RougeRow(
                rouge1_f=result["rouge1"].fmeasure,
                rouge2_f=result["rouge2"].fmeasure,
                rougeL_f=result["rougeL"].fmeasure,
            )
        )
    return rows


# --- BERTScore (PubMedBERT) ---------------------------------------------------


DEFAULT_PUBMEDBERT_MODEL = (
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
# PubMedBERT-base is a 12-layer BERT-base encoder. `bert-score` ships a
# `model2layers` table for popular encoders but not for biomedical ones, so we
# must pass `num_layers` ourselves. The bert-score authors' recommendation for
# unseen BERT-base models is to use the second-to-last layer; that matches the
# choice they make for `bert-base-uncased` (layer 9 of 12) and is what most
# clinical-NLP papers report when using PubMedBERT with BERTScore.
DEFAULT_PUBMEDBERT_NUM_LAYERS = 9


@dataclass(slots=True)
class BertScoreRow:
    precision: float
    recall: float
    f1: float


def bertscore_pubmedbert(
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str = DEFAULT_PUBMEDBERT_MODEL,
    num_layers: int | None = None,
    batch_size: int = 8,
    device: str | None = None,
    verbose: bool = False,
) -> list[BertScoreRow]:
    """Batched BERTScore using a clinical / biomedical encoder.

    `rescale_with_baseline` is left off because the bert-score package only
    ships baselines for a small set of generic encoders (PubMedBERT is not
    among them). Raw F1 is still useful for relative comparison across runs.

    `num_layers` defaults to `DEFAULT_PUBMEDBERT_NUM_LAYERS` for the default
    PubMedBERT model; for any other encoder the caller should pass an explicit
    value or rely on `bert-score`'s built-in `model2layers` lookup.
    """

    if len(candidates) != len(references):
        raise ValueError(
            "candidates and references must be the same length "
            f"(got {len(candidates)} vs {len(references)})."
        )

    from bert_score import BERTScorer
    from bert_score.utils import model2layers

    resolved_device = device or _autoselect_device()
    resolved_num_layers = num_layers
    if resolved_num_layers is None and model_type not in model2layers:
        if model_type == DEFAULT_PUBMEDBERT_MODEL:
            resolved_num_layers = DEFAULT_PUBMEDBERT_NUM_LAYERS
        else:
            raise ValueError(
                f"bert-score has no default num_layers for model '{model_type}'. "
                "Pass num_layers=<int> explicitly."
            )

    scorer = BERTScorer(
        model_type=model_type,
        num_layers=resolved_num_layers,
        lang="en",
        rescale_with_baseline=False,
        device=resolved_device,
        batch_size=batch_size,
    )

    # PubMedBERT (and several other biomedical encoders) ship a tokenizer
    # with `model_max_length` set to int(1e30) as a "no limit" sentinel.
    # When `bert-score` enables truncation, the underlying Rust tokenizer
    # tries to serialize that value into a 32-bit int and crashes with
    # `OverflowError: int too big to convert`. Cap it to a safe BERT max.
    _tokenizer = getattr(scorer, "_tokenizer", None)
    if _tokenizer is not None:
        current = getattr(_tokenizer, "model_max_length", 512)
        _tokenizer.model_max_length = min(int(current), 512)

    precision, recall, f1 = scorer.score(
        cands=list(candidates),
        refs=list(references),
        verbose=verbose,
        batch_size=batch_size,
    )

    return [
        BertScoreRow(precision=float(p), recall=float(r), f1=float(f))
        for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist())
    ]


def _autoselect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --- Gold entity recall -------------------------------------------------------


@dataclass(slots=True)
class EntityRecallRow:
    row_id: int
    gold_n: int
    matched_n: int
    recall: float
    per_semtype: dict[str, tuple[int, int]]  # semtype -> (matched_n, gold_n)
    missed_concepts: list[str]


# Sentinel value for gold concepts whose `semtypes` cell is empty, NaN, or
# not in the configured bucket list. MIMIC-III-Ext-Notes leaves ~27% of gold
# rows untyped, so dropping them would make per-semtype recall stop adding
# up to the overall recall. The "other" bucket catches everything else.
UNTYPED_SEMTYPE: str = "other"


def _normalize_semtype(raw: object) -> str:
    """Normalize a `semtypes` cell to a lowercase non-empty string, or '' if
    the cell is missing / NaN / blank.

    pandas stores NaN for empty CSV cells which turns into the literal string
    "nan" under `str()`, so check for that explicitly.
    """

    if raw is None:
        return ""
    text = str(raw).strip().lower()
    if not text or text == "nan":
        return ""
    return text


def entity_recall_per_note(
    summaries: Mapping[int, str],
    gold_labels: pd.DataFrame,
    *,
    semtype_buckets: Sequence[str] = ("dsyn", "sosy", "mobd"),
    relaxed: bool = False,
    include_other_bucket: bool = True,
) -> list[EntityRecallRow]:
    """Compute per-note recall of gold positive concepts in the generated summary.

    A concept is considered "recalled" if either its `trigger_word` (literal
    span from the source note) or its normalized `concept` name appears in the
    summary as a case-insensitive whole-word/phrase match.

    When `relaxed=True`, the matcher additionally accepts:
    - Porter-stemmed equivalents (so "hypertension" matches "hypertensive").
    - The phrase with generic clinical modifiers stripped ("acute on
      chronic kidney disease" matches "kidney disease").
    - A small set of standard clinical abbreviations and their expansions
      (HTN <-> hypertension, etc.).
    Strict mode (the default) preserves byte-identical behavior with prior
    benchmark runs so reported numbers stay comparable.

    When `include_other_bucket=True` (default), gold rows whose `semtypes`
    is NaN/empty or outside `semtype_buckets` are pooled under an "other"
    bucket so that per-semtype totals reconcile with `gold_n` / `matched_n`.
    Set this to False to preserve the historical behavior of silently
    dropping such rows from the per-semtype breakdown.
    """

    effective_buckets: tuple[str, ...] = tuple(semtype_buckets)
    if include_other_bucket and UNTYPED_SEMTYPE not in effective_buckets:
        effective_buckets = effective_buckets + (UNTYPED_SEMTYPE,)

    rows: list[EntityRecallRow] = []
    grouped = gold_labels.groupby("row_id", sort=True)

    for row_id, group in grouped:
        summary = summaries.get(int(row_id), "")
        gold_n = len(group)
        matched_n = 0
        missed: list[str] = []
        per_semtype: dict[str, list[int]] = {b: [0, 0] for b in effective_buckets}

        for _, label_row in group.iterrows():
            trigger = str(label_row.get("trigger_word", "")).strip()
            concept = str(label_row.get("concept", "")).strip()
            semtype = _normalize_semtype(label_row.get("semtypes"))

            hit = _phrase_in_text(trigger, summary, relaxed=relaxed) or _phrase_in_text(
                concept, summary, relaxed=relaxed
            )
            if hit:
                matched_n += 1
            else:
                missed.append(concept or trigger)

            bucket_key = semtype if semtype in per_semtype else (
                UNTYPED_SEMTYPE if include_other_bucket else None
            )
            if bucket_key is not None and bucket_key in per_semtype:
                per_semtype[bucket_key][1] += 1
                if hit:
                    per_semtype[bucket_key][0] += 1

        recall = matched_n / gold_n if gold_n else 0.0
        rows.append(
            EntityRecallRow(
                row_id=int(row_id),
                gold_n=gold_n,
                matched_n=matched_n,
                recall=recall,
                per_semtype={k: (v[0], v[1]) for k, v in per_semtype.items()},
                missed_concepts=missed,
            )
        )

    return rows


_NEGATION_CUES: tuple[str, ...] = (
    "no",
    "not",
    "denies",
    "denied",
    "denying",
    "negative",
    "without",
    "w/o",
    "r/o",
    "ruled out",
    "absent",
    "absence of",
    "free of",
)


@dataclass(slots=True)
class NegationPreservationRow:
    row_id: int
    negated_n: int
    preserved_n: int  # concept appears AND a negation cue precedes it
    rate: float


@dataclass(slots=True)
class HallucinatedPositiveRow:
    """Gold-negated concepts that the summary asserts positively.

    A concept counts as "hallucinated positive" when it appears in the
    summary and the preceding `window_chars` do *not* contain a negation
    cue. This is a safety-critical signal: the model is saying "yes" to
    something the gold says was explicitly negated in the source note.
    """

    row_id: int
    negated_n: int  # total gold-negated concepts for this note
    hallucinated_n: int  # of those, how many appear affirmatively in summary
    rate: float  # hallucinated_n / negated_n (0.0 if no gold-negated)


def hallucinated_positive_per_note(
    summaries: Mapping[int, str],
    gold_negated: pd.DataFrame,
    *,
    window_chars: int = 40,
) -> list[HallucinatedPositiveRow]:
    """For each note, count gold-negated concepts that leak into the summary
    without a negation cue in front of them.

    Complements `negation_preservation_per_note`: that metric rewards the
    model for carrying negation over; this one penalizes it for flipping a
    negated concept into an affirmative claim. Both can be > 0 on the same
    note (the model mentions a concept twice, once negated, once not).
    """

    rows: list[HallucinatedPositiveRow] = []
    grouped = gold_negated.groupby("row_id", sort=True)

    for row_id, group in grouped:
        summary = summaries.get(int(row_id), "")
        negated_n = len(group)
        hallucinated = 0

        for _, label_row in group.iterrows():
            trigger = str(label_row.get("trigger_word", "")).strip()
            concept = str(label_row.get("concept", "")).strip()
            flagged = False
            for phrase in (trigger, concept):
                if not phrase:
                    continue
                if _phrase_appears_affirmatively(phrase, summary, window_chars):
                    flagged = True
                    break
            if flagged:
                hallucinated += 1

        rate = hallucinated / negated_n if negated_n else 0.0
        rows.append(
            HallucinatedPositiveRow(
                row_id=int(row_id),
                negated_n=negated_n,
                hallucinated_n=hallucinated,
                rate=rate,
            )
        )
    return rows


def entity_rows_to_dataframe(
    rows: Sequence[EntityRecallRow],
    *,
    semtype_buckets: Sequence[str] = ("dsyn", "sosy", "mobd"),
    include_missed_concepts: bool = True,
) -> pd.DataFrame:
    """Flatten a list of `EntityRecallRow` into a per-note DataFrame.

    Columns:
        row_id, gold_n, matched_n, entity_recall,
        gold_{semtype}_n, matched_{semtype}_n, recall_{semtype}
            for each semtype in `semtype_buckets`,
        missed_concepts  (semicolon-separated, if `include_missed_concepts`)
    """

    records: list[dict[str, object]] = []
    for row in rows:
        record: dict[str, object] = {
            "row_id": row.row_id,
            "gold_n": row.gold_n,
            "matched_n": row.matched_n,
            "entity_recall": row.recall,
        }
        for semtype in semtype_buckets:
            matched_n, gold_n = row.per_semtype.get(semtype, (0, 0))
            record[f"gold_{semtype}_n"] = gold_n
            record[f"matched_{semtype}_n"] = matched_n
            record[f"recall_{semtype}"] = (
                matched_n / gold_n if gold_n else float("nan")
            )
        if include_missed_concepts:
            record["missed_concepts"] = "; ".join(row.missed_concepts)
        records.append(record)
    return pd.DataFrame.from_records(records)


def entity_columns_for_semtypes(
    semtype_buckets: Sequence[str] = ("dsyn", "sosy", "mobd"),
) -> list[str]:
    """Column names produced by `entity_rows_to_dataframe` (excluding row_id).

    Useful when a caller needs to know which columns to carry over from a
    baseline DataFrame (e.g. when only recomputing metrics on a subset of
    rows during an A/B comparison).
    """

    cols: list[str] = ["gold_n", "matched_n", "entity_recall"]
    for semtype in semtype_buckets:
        cols.extend([f"gold_{semtype}_n", f"matched_{semtype}_n", f"recall_{semtype}"])
    return cols


def negation_preservation_per_note(
    summaries: Mapping[int, str],
    gold_negated: pd.DataFrame,
    *,
    window_chars: int = 40,
) -> list[NegationPreservationRow]:
    """For each note, the fraction of gold-negated concepts that the summary
    represents WITH a negation cue within `window_chars` before the trigger.

    This is a coarse heuristic; a False positive happens when the model writes
    the concept twice (once positive, once negated) and we only see the first.
    """

    rows: list[NegationPreservationRow] = []
    grouped = gold_negated.groupby("row_id", sort=True)

    for row_id, group in grouped:
        summary = summaries.get(int(row_id), "")
        negated_n = len(group)
        preserved = 0

        for _, label_row in group.iterrows():
            trigger = str(label_row.get("trigger_word", "")).strip()
            concept = str(label_row.get("concept", "")).strip()
            for phrase in (trigger, concept):
                if not phrase:
                    continue
                if _phrase_preceded_by_negation_cue(phrase, summary, window_chars):
                    preserved += 1
                    break

        rate = preserved / negated_n if negated_n else 0.0
        rows.append(
            NegationPreservationRow(
                row_id=int(row_id),
                negated_n=negated_n,
                preserved_n=preserved,
                rate=rate,
            )
        )
    return rows


# --- Aggregation helpers ------------------------------------------------------


def aggregate_per_semtype(rows: Iterable[EntityRecallRow]) -> dict[str, float]:
    """Micro-averaged recall per semantic type across all notes."""

    totals: dict[str, list[int]] = {}
    for row in rows:
        for semtype, (matched, gold) in row.per_semtype.items():
            bucket = totals.setdefault(semtype, [0, 0])
            bucket[0] += matched
            bucket[1] += gold

    return {
        semtype: (matched / gold if gold else 0.0)
        for semtype, (matched, gold) in totals.items()
    }


def micro_recall(rows: Iterable[EntityRecallRow]) -> float:
    matched = 0
    gold = 0
    for row in rows:
        matched += row.matched_n
        gold += row.gold_n
    return matched / gold if gold else 0.0


# --- Internals ----------------------------------------------------------------


# Generic clinical modifiers that don't change the underlying entity. Stripping
# these lets "acute on chronic kidney disease" match a summary that just says
# "kidney disease". Ordered loosely from "context" to "lateralization".
_CLINICAL_MODIFIERS: frozenset[str] = frozenset(
    {
        "acute",
        "chronic",
        "subacute",
        "recurrent",
        "new",
        "newonset",
        "old",
        "history",
        "h/o",
        "hx",
        "mild",
        "moderate",
        "severe",
        "worsening",
        "improving",
        "stable",
        "unstable",
        "left",
        "right",
        "bilateral",
        "lower",
        "upper",
        "primary",
        "secondary",
        "possible",
        "probable",
        "suspected",
        "presumed",
        "early",
        "late",
        "mild-moderate",
        # Connector / function words that show up inside multi-word UMLS
        # concept names ("acute on chronic", "history of") but never carry
        # entity meaning on their own.
        "on",
        "of",
        "the",
        "a",
        "an",
        "and",
        "with",
        "without",
        "to",
        "in",
        "due",
    }
)


# Bidirectional clinical abbreviation map. Keys are normalized lowercase.
# Each value is a tuple of accepted expansions (also normalized). The matcher
# checks both directions so "HTN" can match "hypertension" and vice versa.
_CLINICAL_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "htn": ("hypertension",),
    "dm": ("diabetes mellitus", "diabetes"),
    "dm2": ("type 2 diabetes mellitus", "type 2 diabetes", "t2dm"),
    "t2dm": ("type 2 diabetes mellitus", "type 2 diabetes", "dm2"),
    "dm1": ("type 1 diabetes mellitus", "type 1 diabetes", "t1dm"),
    "t1dm": ("type 1 diabetes mellitus", "type 1 diabetes", "dm1"),
    "ckd": ("chronic kidney disease",),
    "esrd": ("end stage renal disease", "end-stage renal disease"),
    "aki": ("acute kidney injury", "acute renal failure"),
    "chf": ("congestive heart failure", "heart failure"),
    "hfref": ("heart failure with reduced ejection fraction",),
    "hfpef": ("heart failure with preserved ejection fraction",),
    "cad": ("coronary artery disease",),
    "afib": ("atrial fibrillation",),
    "a-fib": ("atrial fibrillation",),
    "af": ("atrial fibrillation",),
    "copd": ("chronic obstructive pulmonary disease",),
    "uti": ("urinary tract infection",),
    "dvt": ("deep vein thrombosis", "deep venous thrombosis"),
    "pe": ("pulmonary embolism",),
    "mi": ("myocardial infarction",),
    "stemi": ("st elevation myocardial infarction", "st-elevation myocardial infarction"),
    "nstemi": (
        "non st elevation myocardial infarction",
        "non-st-elevation myocardial infarction",
    ),
    "cva": ("cerebrovascular accident", "stroke"),
    "tia": ("transient ischemic attack",),
    "gerd": ("gastroesophageal reflux disease",),
    "gib": ("gastrointestinal bleed", "gi bleed"),
    "uti's": ("urinary tract infections",),
    "sob": ("shortness of breath", "dyspnea"),
    "loc": ("loss of consciousness",),
    "n/v": ("nausea and vomiting", "nausea/vomiting"),
    "cp": ("chest pain",),
    "ams": ("altered mental status",),
}
# Build the reverse index once: expansion -> canonical abbrev set.
_CLINICAL_EXPANSIONS_TO_ABBREV: dict[str, tuple[str, ...]] = {}
for _abbr, _expansions in _CLINICAL_ABBREVIATIONS.items():
    for _exp in _expansions:
        _CLINICAL_EXPANSIONS_TO_ABBREV.setdefault(_exp, ())
        _CLINICAL_EXPANSIONS_TO_ABBREV[_exp] = (
            *_CLINICAL_EXPANSIONS_TO_ABBREV[_exp],
            _abbr,
        )


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9/+\-']*")


def _phrase_in_text(phrase: str, text: str, *, relaxed: bool = False) -> bool:
    if not phrase or not text:
        return False
    pattern = _whole_phrase_pattern(phrase)
    if pattern is not None and pattern.search(text) is not None:
        return True
    if not relaxed:
        return False
    return _phrase_in_text_relaxed(phrase, text)


def _whole_phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    cleaned = phrase.strip()
    if not cleaned:
        return None
    # Use \b only when the phrase boundaries are word characters; for
    # something like "w/o" or "r/o" the \b on a slash side still works
    # but doesn't hurt to keep the simple form because we always wrap.
    escaped = re.escape(cleaned)
    return re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)


def _get_stemmer():
    """Lazy-import nltk's PorterStemmer; cache on the function attribute."""

    cached = getattr(_get_stemmer, "_cached", None)
    if cached is not None:
        return cached
    try:
        from nltk.stem.porter import PorterStemmer
    except Exception:  # pragma: no cover - nltk unavailable
        _get_stemmer._cached = False  # type: ignore[attr-defined]
        return False
    stemmer = PorterStemmer()
    _get_stemmer._cached = stemmer  # type: ignore[attr-defined]
    return stemmer


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def _stem_tokens(tokens: Sequence[str]) -> list[str]:
    stemmer = _get_stemmer()
    if not stemmer:
        return list(tokens)
    return [stemmer.stem(tok) for tok in tokens]


def _stemmed_subsequence_match(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    if not needle:
        return False
    n = len(needle)
    if n > len(haystack):
        return False
    for start in range(0, len(haystack) - n + 1):
        if list(haystack[start : start + n]) == list(needle):
            return True
    return False


def _strip_modifiers(tokens: Sequence[str]) -> list[str]:
    return [tok for tok in tokens if tok not in _CLINICAL_MODIFIERS]


def _phrase_in_text_relaxed(phrase: str, text: str) -> bool:
    """Multi-stage fuzzy match used after a strict match has already failed.

    Stages, cheapest first:
      1. Stemmed contiguous-token subsequence.
      2. Same as (1) but with generic clinical modifiers stripped from the
         phrase (e.g. "acute" / "chronic" / "left").
      3. Bidirectional abbreviation expansion (HTN <-> hypertension), with
         each expansion run through stage (1).
    """

    phrase_tokens = _tokenize(phrase)
    if not phrase_tokens:
        return False
    text_tokens = _tokenize(text)
    if not text_tokens:
        return False
    stemmed_text = _stem_tokens(text_tokens)

    # Stage 1: stemmed phrase as-is.
    stemmed_phrase = _stem_tokens(phrase_tokens)
    if _stemmed_subsequence_match(stemmed_phrase, stemmed_text):
        return True

    # Stage 2: drop generic clinical modifiers and retry. Only run when this
    # actually changes the phrase, otherwise it's the same check as stage 1.
    stripped = _strip_modifiers(phrase_tokens)
    if stripped and stripped != phrase_tokens:
        stemmed_stripped = _stem_tokens(stripped)
        if _stemmed_subsequence_match(stemmed_stripped, stemmed_text):
            return True

    # Stage 3: abbreviation <-> expansion. Try both the original normalized
    # phrase and the modifier-stripped form ("history of myocardial
    # infarction" -> "myocardial infarction" -> "MI").
    candidate_keys = {" ".join(phrase_tokens)}
    if stripped:
        candidate_keys.add(" ".join(stripped))
    candidate_strings: list[str] = []
    for key in candidate_keys:
        if key in _CLINICAL_ABBREVIATIONS:
            candidate_strings.extend(_CLINICAL_ABBREVIATIONS[key])
        if key in _CLINICAL_EXPANSIONS_TO_ABBREV:
            candidate_strings.extend(_CLINICAL_EXPANSIONS_TO_ABBREV[key])

    for candidate in candidate_strings:
        cand_tokens = _tokenize(candidate)
        if not cand_tokens:
            continue
        # Cheap path: exact whole-phrase match against the original text.
        cand_pattern = _whole_phrase_pattern(candidate)
        if cand_pattern is not None and cand_pattern.search(text) is not None:
            return True
        # Stemmed contiguous fallback for completeness.
        if _stemmed_subsequence_match(_stem_tokens(cand_tokens), stemmed_text):
            return True

    return False


def _phrase_preceded_by_negation_cue(
    phrase: str,
    text: str,
    window_chars: int,
) -> bool:
    pattern = _whole_phrase_pattern(phrase)
    if pattern is None:
        return False
    for match in pattern.finditer(text):
        start = match.start()
        window_start = max(0, start - window_chars)
        window = text[window_start:start].lower()
        if any(cue in window for cue in _NEGATION_CUES):
            return True
    return False


def _phrase_appears_affirmatively(
    phrase: str,
    text: str,
    window_chars: int,
) -> bool:
    """True if `phrase` occurs in `text` at least once without a negation cue
    in the preceding `window_chars` characters."""

    pattern = _whole_phrase_pattern(phrase)
    if pattern is None:
        return False
    for match in pattern.finditer(text):
        start = match.start()
        window_start = max(0, start - window_chars)
        window = text[window_start:start].lower()
        if not any(cue in window for cue in _NEGATION_CUES):
            return True
    return False
