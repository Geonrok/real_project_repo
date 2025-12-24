@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  run_ci_chunks.cmd - CI test runner with chunk selection
REM  Usage: run_ci_chunks.cmd [chunk]
REM    chunk: core|p22|p23|p24|p25|p26|all (default: all)
REM
REM  Python selection:
REM    1) %ROOT%\.venv\Scripts\python.exe (if exists)
REM    2) FAIL-FAST (default) or system python (if ALLOW_SYSTEM_PYTHON=1)
REM ============================================================

REM ---- Parse chunk argument (default: all) ----
set "CHUNK=%~1"
if "%CHUNK%"=="" set "CHUNK=all"

REM ---- Validate chunk argument (case-insensitive) ----
set "VALID_CHUNK="
for %%C in (core p22 p23 p24 p25 p26 all) do (
    if /I "%CHUNK%"=="%%C" set "VALID_CHUNK=1"
)
if not defined VALID_CHUNK (
    echo [FATAL] Unknown CHUNK=%CHUNK% ^(use: core^|p22^|p23^|p24^|p25^|p26^|all^)
    exit /b 2
)

REM ---- Resolve project root (script dir's parent) ----
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"

REM ---- Choose python interpreter ----
REM Priority 1: project .venv
set "PY="
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PY=%ROOT%\.venv\Scripts\python.exe"
    goto :PY_FOUND
)

REM No .venv found - check ALLOW_SYSTEM_PYTHON
if /I not "%ALLOW_SYSTEM_PYTHON%"=="1" (
    echo [FATAL] No .venv found at %ROOT%\.venv
    echo [HINT] Create venv: python -m venv .venv
    echo [HINT] Or set ALLOW_SYSTEM_PYTHON=1 to use system python ^(NOT recommended^)
    exit /b 2
)

REM ALLOW_SYSTEM_PYTHON=1: fallback to system python
echo [WARN] ALLOW_SYSTEM_PYTHON=1 enabled; using system python fallback...
set "PY=python"

:PY_FOUND
echo [INFO] ROOT=%ROOT%
echo [INFO] PY=%PY%
echo [INFO] CHUNK=%CHUNK%

REM ---- Ensure pytest is available in the chosen interpreter ----
"%PY%" -c "import pytest" >nul 2>&1
if errorlevel 1 goto :PYTEST_MISSING

REM ---- Print env header for reproducibility ----
:ENV_HEADER
"%PY%" -c "import sys,platform; import numpy,pandas,pytest; print('exe=',sys.executable); print('py=',sys.version); print('os=',platform.platform()); print('numpy=',numpy.__version__); print('pandas=',pandas.__version__); print('pytest=',pytest.__version__)" 2>nul

REM ---- Route to selected chunk(s) ----
if /I "%CHUNK%"=="all" goto :RUN_ALL
if /I "%CHUNK%"=="core" goto :RUN_CORE
if /I "%CHUNK%"=="p22" goto :RUN_P22
if /I "%CHUNK%"=="p23" goto :RUN_P23
if /I "%CHUNK%"=="p24" goto :RUN_P24
if /I "%CHUNK%"=="p25" goto :RUN_P25
if /I "%CHUNK%"=="p26" goto :RUN_P26

REM ---- Should not reach here ----
echo [FATAL] Unknown CHUNK=%CHUNK% ^(use: core^|p22^|p23^|p24^|p25^|p26^|all^)
exit /b 2

REM ============================================================
REM  ALL: Run all chunks sequentially with fail-fast
REM ============================================================
:RUN_ALL
call :CHUNK_CORE
if errorlevel 1 exit /b 1

call :CHUNK_P22
if errorlevel 1 exit /b 1

call :CHUNK_P23
if errorlevel 1 exit /b 1

call :CHUNK_P24
if errorlevel 1 exit /b 1

call :CHUNK_P25
if errorlevel 1 exit /b 1

call :CHUNK_P26
if errorlevel 1 exit /b 1

echo [7/7] Done.
exit /b 0

REM ============================================================
REM  Single chunk runners (for selective execution)
REM ============================================================
:RUN_CORE
call :CHUNK_CORE
exit /b %ERRORLEVEL%

:RUN_P22
call :CHUNK_P22
exit /b %ERRORLEVEL%

:RUN_P23
call :CHUNK_P23
exit /b %ERRORLEVEL%

:RUN_P24
call :CHUNK_P24
exit /b %ERRORLEVEL%

:RUN_P25
call :CHUNK_P25
exit /b %ERRORLEVEL%

:RUN_P26
call :CHUNK_P26
exit /b %ERRORLEVEL%

REM ============================================================
REM  Chunk subroutines (reusable test blocks)
REM  - Explicit file lists only (no glob patterns)
REM  - Always use: "%PY%" -m pytest -q ...
REM ============================================================
:CHUNK_CORE
echo [1/7] Phase 2 core...
"%PY%" -m pytest -q ^
  tests/test_phase2_cost_models_handcalc.py ^
  tests/test_phase2_integration_smoke.py ^
  tests/test_phase2_sensitivity_smoke.py
exit /b %ERRORLEVEL%

:CHUNK_P22
echo [2/7] Phase 2.2 event builder...
"%PY%" -m pytest -q tests/test_phase22_event_builder.py
exit /b %ERRORLEVEL%

:CHUNK_P23
echo [3/7] Phase 2.3 guardrails...
"%PY%" -m pytest -q tests/test_phase23_guardrails.py
exit /b %ERRORLEVEL%

:CHUNK_P24
echo [4/7] Phase 2.4 filter impact...
"%PY%" -m pytest -q tests/test_phase24_filter_impact.py
exit /b %ERRORLEVEL%

:CHUNK_P25
echo [5/7] Phase 2.5 distortion...
"%PY%" -m pytest -q tests/test_phase25_distortion.py
exit /b %ERRORLEVEL%

:CHUNK_P26
echo [6/7] Phase 2.6 recommendation...
"%PY%" -m pytest -q tests/test_phase26_recommendation.py
exit /b %ERRORLEVEL%

REM ============================================================
REM  Error handling: pytest missing
REM ============================================================
:PYTEST_MISSING
echo [FATAL] pytest not installed in selected interpreter: %PY%
echo [HINT] Fix by running:
echo        "%PY%" -m pip install -U pip pytest
echo(

REM If already using system python and still no pytest, abort
if /I "%ALLOW_SYSTEM_PYTHON%"=="1" (
    if "%PY%"=="python" (
        echo [FATAL] System python also lacks pytest. Aborting.
        exit /b 2
    )
    REM Try system python fallback
    echo [WARN] Trying system python fallback...
    set "PY=python"
    "%PY%" -c "import pytest" >nul 2>&1
    if errorlevel 1 (
        echo [FATAL] System python also lacks pytest. Aborting.
        exit /b 2
    )
    echo [INFO] Fallback PY=%PY%
    goto :ENV_HEADER
)

echo [FATAL] Aborting. Set ALLOW_SYSTEM_PYTHON=1 to try system python ^(NOT recommended^).
exit /b 2
