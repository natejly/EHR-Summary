# MLX Qwen3.5 Inference

Standalone Python wrapper for `mlx-lm` using `mlx-community/Qwen3.5-9B-4bit`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from mlx_qwen_inference import Qwen35MlxInference

llm = Qwen35MlxInference()
reply = llm.complete("Explain MLX-LM in one paragraph.")
print(reply)
```

## Terminal Chat

```bash
python3 mlx_qwen_inference.py
```

Render markdown responses in the terminal:

```bash
python3 mlx_qwen_inference.py --render-markdown
```

With a system prompt:

```bash
python3 mlx_qwen_inference.py --system-prompt "You are a concise coding assistant."
```

If you explicitly want Qwen's reasoning stream:

```bash
python3 mlx_qwen_inference.py --thinking
```

Useful commands inside the chat:

- `/help`
- `/clear`
- `/system <prompt>`
- `/exit`

## Streaming

```python
from mlx_qwen_inference import Qwen35MlxInference

llm = Qwen35MlxInference()
for chunk in llm.stream("List three uses for local LLM inference."):
    print(chunk, end="", flush=True)
print()
```

## Single Prompt CLI

```bash
python3 mlx_qwen_inference.py --prompt "Explain MLX-LM in one paragraph."
```

The summarization CLIs also accept `--render-markdown` to format headings, lists, and code blocks in terminal output.

## Clinical Summarization

`clinical_summarization.py` layers a source-grounded clinical summarizer on top of the
MLX Qwen wrapper. It asks the model for:

- a concise `Brief Hospital Course`
- a structured clinical summary covering diagnoses, procedures, medication changes,
  important labs and imaging, hospital course by problem, discharge disposition,
  and follow-up

The prompt explicitly tells the model to include only facts present in the source notes
and to write `Not documented.` when information is missing.

### Python API

```python
from clinical_summarization import ClinicalSummarizer

summarizer = ClinicalSummarizer()
summary = summarizer.summarize_text(
    """
    Discharge summary: 74-year-old admitted with acute decompensated heart failure.
    Treated with IV furosemide with improved oxygenation and edema.
    TTE showed EF 35%. Discharged home with cardiology follow-up.
    """,
    patient_id="enc-001",
    note_type="discharge summary",
)
print(summary)
```

You can also summarize a file or a directory of notes:

```python
from clinical_summarization import ClinicalSummarizer

summarizer = ClinicalSummarizer()
summary = summarizer.summarize_path("notes/", recursive=True)
print(summary)
```

### CLI

Summarize a single note file:

```bash
python3 clinical_summarization.py --input notes/discharge_summary.txt
```

Summarize a directory of notes:

```bash
python3 clinical_summarization.py --input notes/ --recursive --suffix txt --suffix md
```

Summarize raw text directly:

```bash
python3 clinical_summarization.py \
  --text "Discharge summary: admitted for pneumonia, treated with ceftriaxone and azithromycin, discharged home on room air." \
  --note-type "discharge summary"
```

Stream the summary as it generates:

```bash
python3 clinical_summarization.py --input notes/discharge_summary.txt --stream
```

For summarization, prefer a generous `--max-tokens` ceiling rather than a tight cap.
The prompt now tells the model to end with a sentinel marker, and the wrapper stops
generation when that marker appears, so the model can finish the structured summary
without continuing indefinitely.

## Patient-Friendly Simplification

`patient_friendly_simplification.py` is the second stage. It takes the structured
clinical summary and rewrites it into plain language for a patient or caregiver,
while keeping the original facts and structure.

It also now supports an optional iterative fact-checking loop:

- decompose the generated rewrite into atomic factual claims
- judge each claim against the original source note with the local LLM
- revise unsupported claims and repeat for up to 3 passes

### Python API

```python
from patient_friendly_simplification import PatientFriendlySimplifier

simplifier = PatientFriendlySimplifier()
rewrite = simplifier.simplify_text(
    """
    Brief Hospital Course:
    Admitted with heart failure exacerbation and improved with IV diuretics.

    Structured Clinical Summary:
    Primary and secondary diagnoses:
    - Acute on chronic systolic heart failure exacerbation.
    """,
    audience="patient",
)
print(rewrite)
```

### CLI

Rewrite a summary from a file:

```bash
python3 patient_friendly_simplification.py --input summaries/clinical_summary.txt
```

Rewrite raw summary text directly:

```bash
python3 patient_friendly_simplification.py \
  --text "Brief Hospital Course: Admitted for pneumonia. Structured Clinical Summary: Primary diagnosis: pneumonia." \
  --audience caregiver
```

Stream the rewrite as it generates:

```bash
python3 patient_friendly_simplification.py --input summaries/clinical_summary.txt --stream
```

Enable iterative fact-checking by providing the original source note:

```bash
python3 patient_friendly_simplification.py \
  --input summaries/clinical_summary.txt \
  --source-note notes/discharge_summary.txt \
  --max-verification-passes 3 \
  --claim-support-threshold 0.5
```

Programmatic verification returns both the final summary and pass-by-pass metadata:

```python
result = simplifier.simplify_text_with_verification(
    clinical_summary,
    source_note_text=source_note,
    audience="patient",
)
print(result.summary)
print(result.verified)
```

Claim verification now uses the local LLM as the judge for hallucination detection.
# EHR-Summary
