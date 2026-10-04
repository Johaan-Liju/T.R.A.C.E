@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Install the Python environment first. See docs\CRIME_TRAINING.md.
  pause
  exit /b 1
)
if not exist "runs\crime_baseline\best.pt" (
  echo Train in Colab and copy best.pt into runs\crime_baseline first.
  pause
  exit /b 1
)
if not exist "models\r3d_18-b3b3357e.pth" (
  echo Copy the exported encoder into models first.
  pause
  exit /b 1
)
set /p "TRACE_CRIME_CLIP=Enter the path to a recorded video: "
set "TRACE_CRIME_CLIP=%TRACE_CRIME_CLIP:"=%"
".venv\Scripts\python.exe" -m trace.predict_crime --source "%TRACE_CRIME_CLIP%"
pause
