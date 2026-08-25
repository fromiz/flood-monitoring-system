@echo off
setlocal
cd /d "%~dp0"

echo [V8.5.20] Flood Monitor starting from:
echo %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv was not found.
  echo Please complete the first-time setup in RUN_V8.5.20_KO.md.
  echo.
  pause
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

echo.
echo Server stopped.
pause
