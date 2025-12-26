@echo off
REM Stage4 Event Trace Orchestrator
REM Traces clamp-only WARN events to concrete symbol/date evidence

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set PYTHON=.\.venv\Scripts\python.exe
set STAGE1_DIR=outputs\stage1_full_v2
set STAGE2_DIR=outputs\stage2_full_v1
set STAGE4_DIR=outputs\stage4_event_trace_v1

echo ========================================
echo Stage4 Event Trace
echo ========================================
echo.

REM Verify prerequisites exist
if not exist "%STAGE2_DIR%\stage2_final_pass_with_warn.csv" (
    echo [ERROR] %STAGE2_DIR%\stage2_final_pass_with_warn.csv not found
    echo         Run Stage2 first
    exit /b 1
)
if not exist "%STAGE1_DIR%\stage1_summary.csv" (
    echo [ERROR] %STAGE1_DIR%\stage1_summary.csv not found
    echo         Run Stage1 first
    exit /b 1
)

echo [INFO] Prerequisites found
echo.

REM Create output directory
if not exist "%STAGE4_DIR%" mkdir "%STAGE4_DIR%"

REM 1. Event Trace
echo [1/2] Running Event Trace...
%PYTHON% scripts\stage4_event_trace.py --stage2-dir %STAGE2_DIR% --stage1-dir %STAGE1_DIR% --out %STAGE4_DIR%
if errorlevel 1 (
    echo [ERROR] Event trace failed
    exit /b 1
)
echo.

REM 2. Finalize
echo [2/2] Generating Metadata and Manifest...
%PYTHON% scripts\stage4_finalize.py --stage4-dir %STAGE4_DIR%
if errorlevel 1 (
    echo [ERROR] Finalization failed
    exit /b 1
)
echo.

REM Sanity checks
echo [SANITY] Verifying outputs...

if not exist "%STAGE4_DIR%\stage4_event_trace.csv" (
    echo [ERROR] stage4_event_trace.csv not found
    exit /b 1
)

if not exist "%STAGE4_DIR%\sha256_manifest.txt" (
    echo [ERROR] sha256_manifest.txt not found
    exit /b 1
)

if not exist "docs\snapshots\stage4_event_trace_v1\sha256_manifest.txt" (
    echo [ERROR] Snapshot manifest not found
    exit /b 1
)

REM Count rows in CSV (excluding header)
for /f %%a in ('find /c /v "" ^< "%STAGE4_DIR%\stage4_event_trace.csv"') do set /a ROW_COUNT=%%a-1
echo [SANITY] Event trace CSV has %ROW_COUNT% rows

echo.
echo ========================================
echo Stage4 Complete!
echo ========================================
echo Output directory: %STAGE4_DIR%
echo.

REM List output files
dir /b "%STAGE4_DIR%\*.csv" "%STAGE4_DIR%\*.json" "%STAGE4_DIR%\*.md" "%STAGE4_DIR%\*.txt" 2>nul

exit /b 0
