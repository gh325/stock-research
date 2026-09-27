#!/usr/bin/env bash
# run.sh — Activate venv and run main.py, passing all arguments through.
# Usage: bash run.sh run
#        bash run.sh schedule
#        bash run.sh track buy AAPL --price 182
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Virtual environment not found. Run setup first:"
    echo "  bash setup.sh"
    exit 1
fi

source "$SCRIPT_DIR/.venv/bin/activate"
python "$SCRIPT_DIR/main.py" "$@"
