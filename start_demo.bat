@echo off
setlocal

cd /d "%~dp0"
title Music Denoising Demo

set "NO_PROXY=127.0.0.1,localhost"
set "no_proxy=127.0.0.1,localhost"
set "GRADIO_SERVER_NAME=localhost"
set "DEMO_CONDA_CMD=D:\anaconda\Scripts\conda.exe"
set "HF_PYTHON=D:\anaconda\envs\hf\python.exe"

set "CONDA_CMD="
if exist "D:\anaconda\Scripts\conda.exe" set "CONDA_CMD=D:\anaconda\Scripts\conda.exe"
if not defined CONDA_CMD if exist "D:\anaconda\condabin\conda.bat" set "CONDA_CMD=D:\anaconda\condabin\conda.bat"
if not defined CONDA_CMD (
  where conda >nul 2>nul
  if not errorlevel 1 set "CONDA_CMD=conda"
)

set "DEMO_CONDA_CMD=%CONDA_CMD%"

if not exist "%HF_PYTHON%" (
  echo Python for conda environment "hf" was not found:
  echo   %HF_PYTHON%
  pause
  exit /b 1
)

if not defined CONDA_CMD (
  echo Conda was not found.
  echo Expected paths:
  echo   D:\anaconda\Scripts\conda.exe
  echo   D:\anaconda\condabin\conda.bat
  echo Or add conda to PATH, then run this file again.
  pause
  exit /b 1
)

"%CONDA_CMD%" env list | findstr /R /C:"^[ ]*hf[ ]" >nul
if errorlevel 1 (
  echo Conda environment "hf" was not found.
  echo Create it first, then run this file again.
  pause
  exit /b 1
)

"%HF_PYTHON%" -c "import torch, numpy, scipy, soundfile, matplotlib, pandas, yaml" >nul 2>nul
if errorlevel 1 (
  echo Installing project dependencies into conda env hf...
  "%HF_PYTHON%" -m pip install --index-url https://pypi.org/simple -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed.
    pause
    exit /b 1
  )
)

"%HF_PYTHON%" -c "import gradio" >nul 2>nul
if errorlevel 1 (
  echo Installing Gradio into conda env hf...
  "%HF_PYTHON%" -m pip install --index-url https://pypi.org/simple gradio==6.14.0
  if errorlevel 1 (
    echo Gradio installation failed.
    pause
    exit /b 1
  )
)

"%HF_PYTHON%" -c "import numpy; import torch; print(numpy.__version__)" >nul 2>nul
if errorlevel 1 (
  echo The hf environment still has an invalid NumPy or PyTorch setup.
  echo Try rerunning after dependency installation finishes cleanly.
  pause
  exit /b 1
)

"%CONDA_CMD%" env list | findstr /R /C:"^[ ]*music_demucs[ ]" >nul
if errorlevel 1 (
  echo Optional Demucs environment "music_demucs" was not found.
  echo Demucs will fail until that environment is created.
)

echo Cleaning stale demo processes...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'conda.exe') -and ($_.CommandLine -like '*music_denoising_lab*demo_app.py*' -or $_.CommandLine -like '*music_denoising_lab*demucs.separate*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul

echo Launching web demo...
"%HF_PYTHON%" demo_app.py

if errorlevel 1 (
  echo Demo exited with an error.
  pause
)

echo Cleaning demo processes...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'conda.exe') -and ($_.CommandLine -like '*music_denoising_lab*demo_app.py*' -or $_.CommandLine -like '*music_denoising_lab*demucs.separate*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>nul
