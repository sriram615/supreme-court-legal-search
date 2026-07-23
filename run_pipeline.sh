#!/bin/bash
# run_pipeline.sh — wrapper that always uses the correct venv binary
VENV_PYTHON="/Users/apple/Desktop/AI-ML/LAWdata/.venv311/bin/python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$VENV_PYTHON" "$SCRIPT_DIR/fetch_and_preprocess.py" "$@"
