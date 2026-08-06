@echo off
REM Windows launcher for object-detector
cd /d "%~dp0"

set "PYTHON="
if exist "venv\Scripts\python.exe" set "PYTHON=venv\Scripts\python.exe"
if not defined PYTHON set "PYTHON=python"

if not defined DETECTOR_HOST set "DETECTOR_HOST=0.0.0.0"
if not defined DETECTOR_PORT set "DETECTOR_PORT=7860"
if not defined DETECTOR_DEVICE set "DETECTOR_DEVICE=auto"

echo Starting object-detector with %PYTHON% on %DETECTOR_HOST%:%DETECTOR_PORT% (device=%DETECTOR_DEVICE%)
"%PYTHON%" app.py
