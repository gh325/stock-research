#!/usr/bin/env bash
# setup.sh — Create virtualenv and install all dependencies.
# Run once: bash setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Stock Research Pipeline — Setup ==="
echo ""

# Check Python version (3.10+ required for str | None syntax)
PYTHON=$(command -v python3 || command -v python)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PYTHON ($PY_VER)"

if "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    echo "✓ Python version OK"
else
    echo "✗ Python 3.10+ required. Please upgrade Python."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating .venv …"
"$PYTHON" -m venv .venv
echo "✓ Virtual environment created at .venv/"

# Activate and install
source .venv/bin/activate
echo ""
echo "Installing dependencies …"
pip install --upgrade pip --quiet
pip install -r requirements.txt

# Copy .env template if not present
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "✓ .env created from template (edit it to add optional API keys)"
fi

echo ""
echo "========================================"
echo "✓ Setup complete!"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Run the pipeline now:"
echo "  python main.py run"
echo ""
echo "Start the nightly scheduler (9 PM ET):"
echo "  python main.py schedule"
echo ""
echo "Other commands:"
echo "  python main.py status                     Show last report"
echo "  python main.py track buy  AAPL --price 182"
echo "  python main.py track sell AAPL --price 195"
echo "  python main.py positions                   Open positions"
echo "  python main.py history                     Past recommendations"
echo "  python main.py run --no-cache              Force fresh download"
echo "========================================"
