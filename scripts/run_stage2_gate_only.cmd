@echo off
REM Stage2 Gate-Only Report Regeneration
REM Runs only the gate report and finalize steps (no backtests)
REM Useful for re-running gate report with updated criteria
REM
REM Prerequisites:
REM   - outputs/stage2_full_v1/stage2_sensitivity.csv must exist
REM   - outputs/stage2_full_v1/stage2_window_stress.csv must exist
REM   - outputs/stage2_full_v1/stage2_data_quality.csv must exist

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set PYTHON=.\.venv\Scripts\python.exe
set STAGE2_DIR=outputs\stage2_full_v1

echo ========================================
echo Stage2 Gate-Only Report Regeneration
echo ========================================
echo.

REM Verify prerequisites exist
if not exist "%STAGE2_DIR%\stage2_sensitivity.csv" (
    echo [ERROR] %STAGE2_DIR%\stage2_sensitivity.csv not found
    echo         Run scripts\run_stage2_all.cmd first
    exit /b 1
)
if not exist "%STAGE2_DIR%\stage2_window_stress.csv" (
    echo [ERROR] %STAGE2_DIR%\stage2_window_stress.csv not found
    echo         Run scripts\run_stage2_all.cmd first
    exit /b 1
)
if not exist "%STAGE2_DIR%\stage2_data_quality.csv" (
    echo [ERROR] %STAGE2_DIR%\stage2_data_quality.csv not found
    echo         Run scripts\run_stage2_all.cmd first
    exit /b 1
)

echo [INFO] Prerequisites found, regenerating gate report...
echo.

REM 1. Gate Report
echo [1/2] Generating Gate Report (with WARN channel)...
%PYTHON% scripts\stage2_gate_report.py --stage2-dir %STAGE2_DIR% --out %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Gate report generation failed
    exit /b 1
)
echo.

REM 2. Finalize
echo [2/2] Regenerating Metadata and Manifest...
%PYTHON% scripts\stage2_finalize.py --stage2-dir %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Finalization failed
    exit /b 1
)
echo.

echo ========================================
echo Stage2 Gate-Only Complete!
echo ========================================
echo Output directory: %STAGE2_DIR%
echo.

REM List updated files
echo Updated files:
echo   - stage2_gate_report.json
echo   - stage2_gate_report.md
echo   - stage2_final_pass.csv
echo   - stage2_final_pass_with_warn.csv
echo   - run_metadata.json
echo   - sha256_manifest.txt
echo.

exit /b 0
