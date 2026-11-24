@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-grounding-local\Scripts\python.exe" (
    echo The prepared Python environment is missing. See docs\passenger-release.md for installation.
    pause
    exit /b 1
)
".venv-grounding-local\Scripts\python.exe" -m tools.presentation_start
if errorlevel 1 (
    pause
    exit /b 1
)
