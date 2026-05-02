"""Reference-based metrics for the EHR pipeline benchmark.

Implements:
    - Rouge 1/2/L (F-measure)
    - BERTScore precision/recall/F1
    - Entity precision/recall/F1 against a gold entity set

All metric functions return plain dicts so they're easy to flatten into a pandas DataFrame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _content_tokens(text: str, min_len: int = 4) -> set[str]:
    return {tok for tok in _tokens(text) if len(tok) >= min_len}


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def bertscore_scores(
    prediction: str,
    reference: str,
    *,
    model_type: str = "roberta-large",
    lang: str = "en",
) -> dict[str, float]:
    """Single-pair BERTScore. Loads the model on first call (slow)."""
    from bert_score import score as bs_score

    P, R, F1 = bs_score(
        cands=[prediction],
        refs=[reference],
        lang=lang,
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=False,
    )
    return {
        "bertscore_p": float(P[0]),
        "bertscore_r": float(R[0]),
        "bertscore_f1": float(F1[0]),
    }


def bertscore_batch(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str = "roberta-large",
    lang: str = "en",
) -> list[dict[str, float]]:
    """Batched BERTScore. Use this when scoring many cases at once."""
    from bert_score import score as bs_score

    if not predictions:
        return []
    P, R, F1 = bs_score(
        cands=predictions,
        refs=references,
        lang=lang,
        model_type=model_type,
        verbose=False,
        rescale_with_baseline=False,
    )
    return [
        {
            "bertscore_p": float(P[i]),
            "bertscore_r": float(R[i]),
            "bertscore_f1": float(F1[i]),
        }
        for i in range(len(predictions))
    ]


@dataclass(frozen=True)
class EntityMatchConfig:
    jaccard_threshold: float = 0.5
    min_overlap_tokens: int = 2


def _entity_matches(entity: str, summary_text: str, cfg: EntityMatchConfig) -> bool:
    if not entity:
        return False
    if entity in summary_text.lower():
        return True

    entity_tokens = _content_tokens(entity)
    if not entity_tokens:
        return False

    summary_tokens = _content_tokens(summary_text)
    if not summary_tokens:
        return False

    overlap = entity_tokens & summary_tokens
    if not overlap:
        return False

    if len(overlap) < cfg.min_overlap_tokens and len(entity_tokens) > 1:
        return False

    union = entity_tokens | summary_tokens
    jaccard_local = len(overlap) / len(entity_tokens)
    return jaccard_local >= cfg.jaccard_threshold


def entity_recall_precision(
    *,
    prediction: str,
    reference: str,
    gold_entities: Iterable[str],
    config: EntityMatchConfig | None = None,
) -> dict[str, float]:
    """Entity precision/recall/F1 against a gold entity set.

    TP: gold entities that appear in BOTH the system summary
    FN: gold entities in the reference but NOT in the system.
    FP: gold entities in the system but NOT in the reference.

    Uses a fuzzy token-overlap match (substring or Jaccard).
    """
    cfg = config or EntityMatchConfig()
    gold = [e for e in {g.lower().strip() for g in gold_entities} if e]
    if not gold:
        return {
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
            "entity_tp": 0,
            "entity_fp": 0,
            "entity_fn": 0,
            "entity_gold": 0,
        }

    in_pred = {e for e in gold if _entity_matches(e, prediction, cfg)}
    in_ref = {e for e in gold if _entity_matches(e, reference, cfg)}

    tp = in_pred & in_ref
    fp = in_pred - in_ref
    fn = in_ref - in_pred

    precision = len(tp) / len(in_pred) if in_pred else 0.0
    recall = len(tp) / len(in_ref) if in_ref else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1,
        "entity_tp": len(tp),
        "entity_fp": len(fp),
        "entity_fn": len(fn),
        "entity_gold": len(gold),
    }


def all_textual_metrics(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE only (BERTScore is loaded separately to avoid model warm-up)."""
    return rouge_scores(prediction, reference)


# ---------------------------------------------------------------------------
# Flesch-Kincaid grade level (pure Python, no extra deps)
# ---------------------------------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"[.!?]+(?:\s+|$)")
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CITATION_RE = re.compile(r"\[E:[^\]]+\]")
_MARKDOWN_HEADER = re.compile(r"^\s*#{1,6}\s+", flags=re.MULTILINE)
_MARKDOWN_LIST = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", flags=re.MULTILINE)


def _strip_markdown_for_readability(text: str) -> str:
    """Remove headers, list markers, and inline citations so they don't skew
    the FK calculation (citations are not real words)."""
    text = _CITATION_RE.sub("", text)
    text = _MARKDOWN_HEADER.sub("", text)
    text = _MARKDOWN_LIST.sub("", text)
    return text


def _count_syllables(word: str) -> int:
    """Heuristic syllable counter — based on vowel-group counting with the
    standard silent-e adjustment. Good enough for FK grade computation."""
    word = word.lower()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1  # silent e
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1  # "table", "bottle"
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> dict[str, float | int]:
    """Flesch-Kincaid grade level for a Markdown-aware text.

    FKGL = 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

    Returns a dict with the grade plus the constituent counts so we can
    diagnose why a given summary scored where it did.
    """
    cleaned = _strip_markdown_for_readability(text)
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(cleaned) if s.strip()]
    words = _WORD_RE.findall(cleaned)
    n_sent = max(len(sentences), 1)
    n_words = len(words)

    if n_words == 0:
        return {
            "fk_grade": 0.0,
            "fk_words": 0,
            "fk_sentences": 0,
            "fk_syllables": 0,
            "fk_in_target_band": False,
        }

    n_syll = sum(_count_syllables(w) for w in words)
    grade = 0.39 * (n_words / n_sent) + 11.8 * (n_syll / n_words) - 15.59

    return {
        "fk_grade": round(grade, 2),
        "fk_words": n_words,
        "fk_sentences": n_sent,
        "fk_syllables": n_syll,
        "fk_in_target_band": 7.0 <= grade <= 8.99,
    }
