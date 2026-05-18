@echo off
setlocal

title Stop Music Denoising Demo

echo Stopping Music Denoising Demo processes...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'python.exe' -or $_.Name -eq 'conda.exe') -and ($_.CommandLine -like '*music_denoising_lab*demo_app.py*' -or $_.CommandLine -like '*music_denoising_lab*demucs.separate*') } | ForEach-Object { Write-Output ('Stopping PID ' + $_.ProcessId + ' ' + $_.Name); Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

echo Done.
pause
