"""End-to-end plumbing check for the MIMIC benchmark.

Runs at SAMPLE_SIZE=3 with a stub summarizer (no MLX call). Validates:
  * data loading + gold filtering
  * disk-cached runner (atomic write + cache hit on second pass)
  * ROUGE-1/2/L scoring
  * BERTScore (PubMedBERT) batched scoring
  * gold entity recall + per-semtype breakdown + negation preservation
  * results-table merge + CSV write

Run directly:  python -m benchmarks._smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import CACHE_DIR, RESULTS_DIR  # noqa: E402
from benchmarks.data import (  # noqa: E402
    gold_negated_concepts,
    gold_positive_concepts,
    load_notes_and_labels,
)
from benchmarks.metrics import (  # noqa: E402
    aggregate_per_semtype,
    bertscore_pubmedbert,
    entity_recall_per_note,
    micro_recall,
    negation_preservation_per_note,
    rouge_scores,
)
from benchmarks.runner import clear_cache, run as run_summarizer  # noqa: E402


SMOKE_TAG = "_smoke_stub"


class _StubSummarizer:
    """Returns deterministic text that mentions some gold concepts so the
    entity-recall metric exercises a non-empty intersection.
    """

    def __init__(self, gold_pos: pd.DataFrame, gold_neg: pd.DataFrame) -> None:
        self._gold_pos = gold_pos
        self._gold_neg = gold_neg
        self.calls = 0

    def summarize_text(self, note_text: str, *, patient_id: str | None = None) -> str:
        self.calls += 1
        rid = int(patient_id) if patient_id else -1
        pos_for_note = self._gold_pos[self._gold_pos["row_id"] == rid]
        neg_for_note = self._gold_neg[self._gold_neg["row_id"] == rid]

        # Mention the first ~half of the positive concepts (so recall != 0
        # and != 1) using the literal trigger word from the source note.
        n_to_mention = max(1, len(pos_for_note) // 2)
        pos_lines = [
            f"- {row['trigger_word']}"
            for _, row in pos_for_note.head(n_to_mention).iterrows()
        ]

        neg_lines = [
            f"- denies {row['trigger_word']}"
            for _, row in neg_for_note.head(2).iterrows()
        ]

        body_lines = [
            "Brief Hospital Course:",
            "- Stub summary for plumbing test only.",
        ]
        if pos_lines:
            body_lines.append("Diagnoses (primary + secondary):")
            body_lines.extend(pos_lines)
        if neg_lines:
            body_lines.append("Pertinent negatives:")
            body_lines.extend(neg_lines)

        return "\n".join(body_lines) + "\n"


def main() -> int:
    print("[smoke] loading data ...")
    bundle = load_notes_and_labels(sample_size=3, seed=0)
    notes_df = bundle.notes
    labels_df = bundle.labels
    gold_pos_df = gold_positive_concepts(labels_df)
    gold_neg_df = gold_negated_concepts(labels_df)
    print(
        f"[smoke] notes={len(notes_df)}  labels={len(labels_df)}  "
        f"gold_pos={len(gold_pos_df)}  gold_neg={len(gold_neg_df)}"
    )

    # Force a clean cache for the stub tag so we exercise the generate path.
    print("[smoke] clearing prior smoke-stub cache entries ...")
    for stale in CACHE_DIR.glob(f"{SMOKE_TAG}__*.txt"):
        stale.unlink()

    summarizer = _StubSummarizer(gold_pos_df, gold_neg_df)

    print("[smoke] first pass (should generate) ...")
    summaries, stats1 = run_summarizer(
        notes_df,
        summarizer,
        cache_dir=CACHE_DIR,
        model_tag=SMOKE_TAG,
        progress=False,
    )
    assert stats1.n_generated == len(notes_df), (
        f"first pass should generate all {len(notes_df)} notes, got {stats1.n_generated}"
    )
    assert stats1.n_cache_hits == 0
    print(
        f"[smoke] first pass: generated={stats1.n_generated}  "
        f"hits={stats1.n_cache_hits}  elapsed={stats1.elapsed_seconds:.2f}s"
    )

    print("[smoke] second pass (should hit cache) ...")
    summaries2, stats2 = run_summarizer(
        notes_df,
        summarizer,
        cache_dir=CACHE_DIR,
        model_tag=SMOKE_TAG,
        progress=False,
    )
    assert stats2.n_generated == 0, "second pass should not regenerate"
    assert stats2.n_cache_hits == len(notes_df)
    assert summaries == summaries2, "cache round-trip mismatch"
    print(
        f"[smoke] second pass: generated={stats2.n_generated}  "
        f"hits={stats2.n_cache_hits}  elapsed={stats2.elapsed_seconds:.2f}s"
    )

    ordered = notes_df["row_id"].tolist()
    cands = [summaries[r] for r in ordered]
    refs = notes_df["text"].tolist()

    print("[smoke] ROUGE ...")
    rouge_rows = rouge_scores(cands, refs)
    assert len(rouge_rows) == len(ordered)
    print(
        f"[smoke] ROUGE means: r1={sum(r.rouge1_f for r in rouge_rows)/len(rouge_rows):.3f} "
        f"r2={sum(r.rouge2_f for r in rouge_rows)/len(rouge_rows):.3f} "
        f"rL={sum(r.rougeL_f for r in rouge_rows)/len(rouge_rows):.3f}"
    )

    print("[smoke] BERTScore (PubMedBERT, first call may download ~440MB) ...")
    bert_rows = bertscore_pubmedbert(cands, refs, batch_size=4, verbose=False)
    assert len(bert_rows) == len(ordered)
    print(
        f"[smoke] BERTScore means: P={sum(b.precision for b in bert_rows)/len(bert_rows):.3f} "
        f"R={sum(b.recall for b in bert_rows)/len(bert_rows):.3f} "
        f"F1={sum(b.f1 for b in bert_rows)/len(bert_rows):.3f}"
    )

    print("[smoke] entity recall ...")
    entity_rows = entity_recall_per_note(summaries, gold_pos_df)
    micro = micro_recall(entity_rows)
    per_semtype = aggregate_per_semtype(entity_rows)
    assert len(entity_rows) == gold_pos_df["row_id"].nunique()
    assert 0.0 <= micro <= 1.0
    assert micro > 0.0, "stub mentions ~half the gold concepts; recall must be > 0"
    print(f"[smoke] micro recall: {micro:.3f}  per-semtype: {per_semtype}")

    if len(gold_neg_df):
        print("[smoke] negation preservation ...")
        neg_rows = negation_preservation_per_note(summaries, gold_neg_df)
        total = sum(r.negated_n for r in neg_rows)
        preserved = sum(r.preserved_n for r in neg_rows)
        rate = preserved / total if total else 0.0
        print(f"[smoke] negation preservation: {preserved}/{total} = {rate:.3f}")
    else:
        print("[smoke] no gold negated concepts in sample, skipping neg metric")

    print("[smoke] building results_df ...")
    rouge_df = pd.DataFrame(
        {
            "row_id": ordered,
            "rouge1_f": [r.rouge1_f for r in rouge_rows],
            "rouge2_f": [r.rouge2_f for r in rouge_rows],
            "rougeL_f": [r.rougeL_f for r in rouge_rows],
        }
    )
    bert_df = pd.DataFrame(
        {
            "row_id": ordered,
            "bert_p": [b.precision for b in bert_rows],
            "bert_r": [b.recall for b in bert_rows],
            "bert_f1": [b.f1 for b in bert_rows],
        }
    )
    entity_df = pd.DataFrame(
        {
            "row_id": [r.row_id for r in entity_rows],
            "gold_n": [r.gold_n for r in entity_rows],
            "matched_n": [r.matched_n for r in entity_rows],
            "entity_recall": [r.recall for r in entity_rows],
        }
    )
    results_df = (
        notes_df[["row_id", "hadm_id", "subject_id"]]
        .merge(rouge_df, on="row_id")
        .merge(bert_df, on="row_id")
        .merge(entity_df, on="row_id")
    )
    assert len(results_df) == len(notes_df)
    print(results_df.round(3).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "_smoke_results.csv"
    results_df.to_csv(out, index=False)
    print(f"[smoke] wrote {out}")

    # Clean up smoke artifacts so a real run isn't polluted.
    for stale in CACHE_DIR.glob(f"{SMOKE_TAG}__*.txt"):
        stale.unlink()
    out.unlink(missing_ok=True)
    print("[smoke] cleaned up smoke artifacts")
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
