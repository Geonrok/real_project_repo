@echo off
REM Stage1 Full Backtest Runner
REM Purpose: Execute Stage1 grid backtest on all symbols across 4 markets
REM Output: outputs/stage1_full/

setlocal
cd /d "%~dp0.."

echo [PURPOSE] Run Stage1 full backtest (all symbols, all markets, eval-mode both)
echo ======================================================================

if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)

echo Python: %PY%
echo Start time: %date% %time%
echo ----------------------------------------------------------------------

%PY% scripts/backtest_runner.py ^
    --markets configs/markets.yaml ^
    --normalized-dir outputs/normalized_1d ^
    --grid configs/grid_stage1.yaml ^
    --out outputs/stage1_full ^
    --eval-mode both

set EC=%ERRORLEVEL%
echo ----------------------------------------------------------------------
echo End time: %date% %time%
echo Exit code: %EC%

if %EC% EQU 0 (
    echo [STATUS] Stage1 full backtest completed successfully.
) else (
    echo [ERROR] Stage1 full backtest failed with exit code %EC%.
)

endlocal
exit /b %EC%
