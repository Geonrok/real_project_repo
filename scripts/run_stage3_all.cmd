@echo off
REM Stage3 Root-Cause Analysis Orchestrator
REM Analyzes clamp events from Stage2 failed strategies

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set PYTHON=.\.venv\Scripts\python.exe
set STAGE2_DIR=outputs\stage2_full_v1
set STAGE1_DIR=outputs\stage1_full_v2
set STAGE3_DIR=outputs\stage3_rootcause_v1

echo ========================================
echo Stage3 Root-Cause Analysis
echo ========================================
echo.

REM Create output directory
if not exist "%STAGE3_DIR%" mkdir "%STAGE3_DIR%"

REM 1. Run Clamp Root-Cause Analysis
echo [1/2] Running Clamp Root-Cause Analysis...
%PYTHON% scripts\stage3_clamp_rootcause.py --stage2-dir %STAGE2_DIR% --stage1-dir %STAGE1_DIR% --out %STAGE3_DIR%
if errorlevel 1 (
    echo [ERROR] Root-cause analysis failed
    exit /b 1
)
echo.

REM 2. Finalize (metadata + manifest + snapshots)
echo [2/2] Generating Metadata and Manifest...
%PYTHON% scripts\stage3_finalize.py --stage3-dir %STAGE3_DIR%
if errorlevel 1 (
    echo [ERROR] Finalization failed
    exit /b 1
)
echo.

REM Sanity checks
echo [SANITY] Checking output files...

if not exist "%STAGE3_DIR%\stage3_clamp_rootcause.csv" (
    echo [ERROR] Missing stage3_clamp_rootcause.csv
    exit /b 1
)

if not exist "%STAGE3_DIR%\stage3_clamp_rootcause.md" (
    echo [ERROR] Missing stage3_clamp_rootcause.md
    exit /b 1
)

if not exist "%STAGE3_DIR%\run_metadata.json" (
    echo [ERROR] Missing run_metadata.json
    exit /b 1
)

if not exist "%STAGE3_DIR%\sha256_manifest.txt" (
    echo [ERROR] Missing sha256_manifest.txt
    exit /b 1
)

REM Check row count (should be >= 13 for binance_spot failures)
for /f %%a in ('type "%STAGE3_DIR%\stage3_clamp_rootcause.csv" ^| find /c /v ""') do set ROWS=%%a
set /a ROWS=ROWS-1
echo   stage3_clamp_rootcause.csv: %ROWS% rows

if %ROWS% LSS 13 (
    echo [WARN] Expected at least 13 rows, got %ROWS%
)

echo.
echo ========================================
echo Stage3 Complete!
echo ========================================
echo Output directory: %STAGE3_DIR%
echo.

REM List output files
dir /b "%STAGE3_DIR%\*.csv" "%STAGE3_DIR%\*.json" "%STAGE3_DIR%\*.md" "%STAGE3_DIR%\*.txt" 2>nul

echo.
echo Snapshots copied to: docs\snapshots\stage3_rootcause_v1\

exit /b 0
