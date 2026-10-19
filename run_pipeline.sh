#!/usr/bin/env bash
# ==============================================================================
# Industrial Anomaly Synthesis Benchmark - Bash Runner
# Usage:
#   bash run_pipeline.sh                     (Runs default SKAB benchmark)
#   bash run_pipeline.sh --dataset ai4i      (Runs AI4I benchmark)
#   bash run_pipeline.sh --csv data/raw/my_sensor.csv --target_col defect
# ==============================================================================

if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment: .venv"
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

python scripts/run_pipeline.py "$@"
