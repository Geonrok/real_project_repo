@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "VENV_PY=%SCRIPT_DIR%..\.venv\Scripts\python.exe"

if exist "%VENV_PY%" (
  set "PYTHON_EXE=%VENV_PY%"
) else (
  set "PYTHON_EXE=python"
)

%PYTHON_EXE% -m pytest -q %*
endlocal
