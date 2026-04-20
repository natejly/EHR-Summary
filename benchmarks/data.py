"""Loaders for the MIMIC-III-Ext-Notes dataset used by the benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "mimic-iii-ext-notes-1.0.0"
)


@dataclass(slots=True)
class MimicNotesBundle:
    """Sampled notes plus the labels that belong to those notes."""

    notes: pd.DataFrame  # columns: row_id, hadm_id, subject_id, text
    labels: pd.DataFrame  # columns: row_id, trigger_word, concept, semtypes, ...
    dataset_dir: Path


def load_notes_and_labels(
    sample_size: int | None = 25,
    seed: int = 0,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
) -> MimicNotesBundle:
    """Load notes.csv + labels.csv, optionally sampling a subset of notes.

    The sample is deterministic for a given (sample_size, seed). Labels are
    filtered to the row_ids present in the sampled notes so downstream metric
    code never has to think about it.
    """

    dataset_dir = Path(dataset_dir)
    notes_path = dataset_dir / "notes.csv"
    labels_path = dataset_dir / "labels.csv"

    if not notes_path.exists():
        raise FileNotFoundError(
            f"notes.csv not found at {notes_path}. Place the MIMIC-III-Ext-Notes "
            "release under datasets/mimic-iii-ext-notes-1.0.0/."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv not found at {labels_path}.")

    notes = pd.read_csv(notes_path)
    labels = pd.read_csv(labels_path)

    notes = notes.dropna(subset=["text"]).copy()
    notes["text"] = notes["text"].astype(str)

    if sample_size is not None and sample_size < len(notes):
        notes = (
            notes.sample(n=sample_size, random_state=seed)
            .sort_values("row_id")
            .reset_index(drop=True)
        )
    else:
        notes = notes.sort_values("row_id").reset_index(drop=True)

    labels = labels[labels["row_id"].isin(notes["row_id"])].reset_index(drop=True)

    # Normalize the categorical-ish columns to plain lower-case strings so
    # downstream filters are simple string equality.
    for column in ("detection", "encounter", "negation"):
        if column in labels.columns:
            labels[column] = labels[column].astype(str).str.strip().str.lower()

    if "semtypes" in labels.columns:
        labels["semtypes"] = labels["semtypes"].fillna("unknown").astype(str)

    return MimicNotesBundle(notes=notes, labels=labels, dataset_dir=dataset_dir)


def gold_positive_concepts(labels: pd.DataFrame) -> pd.DataFrame:
    """Return the gold "this concept actually happened to this patient" rows.

    A concept is considered a non-negated positive when:
      detection == "yes"  (real concept, not a spurious string match)
      encounter == "yes"  (relevant to this encounter, not just PMH mention)
      negation  == "no"   (not negated)
    """

    return labels[
        (labels["detection"] == "yes")
        & (labels["encounter"] == "yes")
        & (labels["negation"] == "no")
    ].reset_index(drop=True)


def gold_negated_concepts(labels: pd.DataFrame) -> pd.DataFrame:
    """Return gold concepts that are explicitly negated in the note.

    Used by the optional negation-preservation metric.
    """

    return labels[
        (labels["detection"] == "yes")
        & (labels["encounter"] == "yes")
        & (labels["negation"] == "yes")
    ].reset_index(drop=True)
