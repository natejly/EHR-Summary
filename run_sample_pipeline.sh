#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT_PATH="${1:-$ROOT_DIR/sample_notes/synthetic_discharge_summary.txt}"
STEP1_OUTPUT="${STEP1_OUTPUT:-$ROOT_DIR/sample_notes/step1_clinical_summary.txt}"
STEP2_OUTPUT="${STEP2_OUTPUT:-$ROOT_DIR/sample_notes/step2_patient_friendly.txt}"

MODEL_PATH="${MODEL_PATH:-mlx-community/Qwen3.5-9B-4bit}"
STEP1_MAX_TOKENS="${STEP1_MAX_TOKENS:-1200}"
STEP2_MAX_TOKENS="${STEP2_MAX_TOKENS:-1200}"
AUDIENCE="${AUDIENCE:-patient}"
PATIENT_ID="${PATIENT_ID:-}"
ENABLE_FACT_CHECKING="${ENABLE_FACT_CHECKING:-0}"
MAX_VERIFICATION_PASSES="${MAX_VERIFICATION_PASSES:-3}"
CLAIM_SUPPORT_THRESHOLD="${CLAIM_SUPPORT_THRESHOLD:-0.5}"
ENTITY_MIN_PRECISION="${ENTITY_MIN_PRECISION:-}"
ENTITY_MIN_RECALL="${ENTITY_MIN_RECALL:-}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input note not found: $INPUT_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$STEP1_OUTPUT")" "$(dirname "$STEP2_OUTPUT")"

step1_cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/clinical_summarization.py"
  --input "$INPUT_PATH"
  --model "$MODEL_PATH"
  --max-tokens "$STEP1_MAX_TOKENS"
)

step2_cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/patient_friendly_simplification.py"
  --input "$STEP1_OUTPUT"
  --model "$MODEL_PATH"
  --audience "$AUDIENCE"
  --max-tokens "$STEP2_MAX_TOKENS"
)

if [[ -n "$PATIENT_ID" ]]; then
  step1_cmd+=(--patient-id "$PATIENT_ID")
  step2_cmd+=(--patient-id "$PATIENT_ID")
fi

if [[ "$ENABLE_FACT_CHECKING" == "1" ]]; then
  step2_cmd+=(
    --source-note "$INPUT_PATH"
    --max-verification-passes "$MAX_VERIFICATION_PASSES"
    --claim-support-threshold "$CLAIM_SUPPORT_THRESHOLD"
  )
  if [[ -n "$ENTITY_MIN_PRECISION" ]]; then
    step2_cmd+=(--entity-min-precision "$ENTITY_MIN_PRECISION")
  fi
  if [[ -n "$ENTITY_MIN_RECALL" ]]; then
    step2_cmd+=(--entity-min-recall "$ENTITY_MIN_RECALL")
  fi
fi

echo "Step 1: clinical summarization"
printf 'Command:'
printf ' %q' "${step1_cmd[@]}"
printf '\n'
"${step1_cmd[@]}" > "$STEP1_OUTPUT"

echo "Step 2: patient-friendly simplification"
printf 'Command:'
printf ' %q' "${step2_cmd[@]}"
printf '\n'
"${step2_cmd[@]}" > "$STEP2_OUTPUT"

echo
echo "Pipeline complete."
echo "Input:  $INPUT_PATH"
echo "Step 1: $STEP1_OUTPUT"
echo "Step 2: $STEP2_OUTPUT"
