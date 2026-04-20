# **MIMIC-III-Ext-Notes**

## Overview

**MIMIC-III-Ext-Notes** is a derived dataset created from the MIMIC-III Clinical Database (v1.4) that provides expert-annotated unstructured clinical notes for research in clinical natural language processing (NLP) and large language model (LLM) evaluation.

The dataset focuses on extracting and contextualizing clinically relevant concepts—such as symptoms and disease mentions—from real-world clinical documentation. In addition to concept detection, annotations capture encounter relevance and negation status, enabling evaluation of models’ ability to reason about clinical context rather than surface-level term matching.

## File Description

The dataset is distributed as comma-separated value (CSV) files and is centered around the unique identifier `row_id`, which links annotations to their source clinical notes.

### Included Files

* **`notes.csv`**
  Contains 150 deidentified clinical notes sampled from MIMIC-III.

  * `row_id`: Unique identifier for each note
  * `subject_id`: Patient identifier
  * `hadm_id`: Hospital admission identifier
  * `text`: Full clinical note text

* **`labels.csv`**
  Contains 2,288 clinician-annotated clinical concepts extracted from the notes.

  * `row_id`: Identifier linking the concept to its source note
  * `trigger_word`: Text span that triggered concept identification
  * `concept`: Normalized clinical concept name
  * `semtypes`: Semantic type(s) of the concept
  * `num_sents`: Sentence index of the trigger word (sentences separated by `\n`)
  * `start`, `end`: Character offsets of the trigger word in the note text
  * `detection`: Correct detection label
  * `encounter`: Encounter relevance label
  * `negation`: Negation status label


## Basic Usage Guidance

This dataset is intended for **research and benchmarking**, particularly in:

* Clinical NLP and information extraction
* Evaluation of LLMs for contextual understanding
* Negation detection and temporal reasoning
* Development and testing of annotation or adjudication workflows


## Limitations

* Annotations are limited to three MetaMap semantic groups and do not include procedures or medications.
* Context is restricted to individual notes rather than longitudinal patient histories.
* The dataset is designed for evaluation and benchmarking rather than population-level inference.


## Ethics and Data Use

This dataset is derived entirely from the **deidentified MIMIC-III Clinical Database (v1.4)** and contains no new patient data. All data handling complied with the MIMIC-III Data Use Agreement and HIPAA Safe Harbor provisions.

Users must have credentialed access to MIMIC-III and comply with all PhysioNet and institutional data use requirements. No attempts should be made to reidentify patients or clinicians.

