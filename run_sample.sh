#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

python -m ehr_pipeline.cli \
  --bundle data/sample_bundle.json \
  --notes data/sample_notes \
  --case-id sample \
  "$@"
