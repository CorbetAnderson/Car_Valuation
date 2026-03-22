#!/usr/bin/env bash
set -euo pipefail

# cd to repo root (directory above /scripts)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# If a local venv exists, use it
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

# Run training pipeline (forward any args)
exec "$PY" -m src.car_valuation.pipelines.run_train "$@"
