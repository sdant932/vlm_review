#!/usr/bin/env bash
#
# One-shot setup. Idempotent: safe to re-run.
#
#   ./setup.sh              use the active python (venv, conda, system)
#   ./setup.sh --conda      create/reuse the conda env named `blindspot` first
#
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--conda" ]]; then
  if ! command -v conda >/dev/null; then echo "conda not found on PATH"; exit 1; fi
  conda env list | grep -qE '^blindspot\s' \
    && echo "==> conda env 'blindspot' exists, reusing" \
    || { echo "==> creating conda env 'blindspot'"; conda env create -f environment.yml; }
  echo
  echo "Now run:  conda activate blindspot && ./setup.sh"
  exit 0
fi

echo "==> python: $(python -V) at $(command -v python)"

echo "==> installing the package (editable) + download extras"
python -m pip install -q -e '.[download,dev]'

echo "==> creating the runtime directories git does not track"
mkdir -p results outputs cache third_party

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> wrote .env  -- put your key in it: ANTHROPIC_API_KEY=sk-ant-..."
else
  echo "==> .env already present, left alone"
fi

echo "==> verifying the install"
python scripts/verify_install.py
python -m pytest

cat <<'MSG'

Setup complete.

Next:
  1. Put your key in .env             ANTHROPIC_API_KEY=sk-ant-...
  2. Read docs/PIPELINE.md            what to run, in what order
  3. Smallest useful thing to try:

       python -m blindspot.core.runner --datasets svg_localization \
              --limit 20 --max-spend 0.10

     That scores 20 localization questions against Haiku 4.5 and appends to
     results/. It is resumable and stops hard at --max-spend.

Note: the scraped benchmarks (CharXiv, InfographicVQA, ScreenSpot-Pro, ...) are
NOT in this repo -- they are third-party data. scripts/download/ fetches them;
see docs/DATASETS.md for what each one is and why it was chosen.
MSG
