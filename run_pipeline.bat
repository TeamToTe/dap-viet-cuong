@echo off
REM ==============================================================================
REM Industrial Anomaly Synthesis Benchmark - Windows Batch Runner
REM Usage:
REM   run_pipeline.bat                     (Runs default SKAB benchmark)
REM   run_pipeline.bat --dataset ai4i      (Runs AI4I benchmark)
REM   run_pipeline.bat --csv data\raw\my_sensor.csv --target_col defect
REM ==============================================================================

if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: .venv
    call .venv\Scripts\activate.bat
)

python scripts\run_pipeline.py %*
