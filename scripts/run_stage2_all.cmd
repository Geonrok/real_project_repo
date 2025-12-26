@echo off
REM Stage2 Full Run Orchestrator
REM Runs all Stage2 verification scripts in sequence

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set PYTHON=.\.venv\Scripts\python.exe
set STAGE1_DIR=outputs\stage1_full_v2
set STAGE2_DIR=outputs\stage2_full_v1

echo ========================================
echo Stage2 Full Run
echo ========================================
echo.

REM Create output directory
if not exist "%STAGE2_DIR%" mkdir "%STAGE2_DIR%"

REM 1. Cost Sensitivity
echo [1/4] Running Cost Sensitivity Analysis...
%PYTHON% scripts\stage2_sensitivity.py --stage1-dir %STAGE1_DIR% --out %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Sensitivity analysis failed
    exit /b 1
)
echo.

REM 2. Window Stress
echo [2/4] Running Window Stress Testing...
%PYTHON% scripts\stage2_window_stress.py --stage1-dir %STAGE1_DIR% --out %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Window stress testing failed
    exit /b 1
)
echo.

REM 3. Data Quality
echo [3/4] Running Data Quality Diagnostics...
%PYTHON% scripts\stage2_data_quality.py --stage2-dir %STAGE2_DIR% --out %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Data quality diagnostics failed
    exit /b 1
)
echo.

REM 4. Gate Report
echo [4/4] Generating Gate Report...
%PYTHON% scripts\stage2_gate_report.py --stage2-dir %STAGE2_DIR% --out %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Gate report generation failed
    exit /b 1
)
echo.

REM 5. Generate metadata and manifest
echo [5/5] Generating Metadata and Manifest...
%PYTHON% scripts\stage2_finalize.py --stage2-dir %STAGE2_DIR%
if errorlevel 1 (
    echo [ERROR] Finalization failed
    exit /b 1
)
echo.

echo ========================================
echo Stage2 Complete!
echo ========================================
echo Output directory: %STAGE2_DIR%
echo.

REM List output files
dir /b "%STAGE2_DIR%\*.csv" "%STAGE2_DIR%\*.json" "%STAGE2_DIR%\*.md" "%STAGE2_DIR%\*.txt" 2>nul

exit /b 0
