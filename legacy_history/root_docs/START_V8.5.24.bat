@echo off
setlocal
cd /d "%~dp0"
echo [V8.5.24] Flood Monitor starting from:
echo %CD%
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [ERROR] .venv was not found.
  echo Please complete the first-time setup in RUN_V8.5.24_KO.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m app.main
pause
