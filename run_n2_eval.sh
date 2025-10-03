#!/usr/bin/env bash
# Wrapper to run n2 evaluator with the project-local virtualenv
# Usage: ./run_n2_eval.sh [args...]

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PY="$ROOT_DIR/n2/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "Python executable not found: $VENV_PY"
  echo "Make sure the n2 virtualenv exists (python -m venv n2/.venv) or adjust the path."
  exit 1
fi
exec "$VENV_PY" "$ROOT_DIR/n2/evaluate_n2.py" "$@"
