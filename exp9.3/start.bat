@echo off
rem exp9.3: run the remaining (model, task) pairs on the local Ollama, then print the score.
rem Pairs that already have output-<id>.txt and stats-<id>.json are skipped.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_local.ps1 -Ctx 8192
python score.py "%~dp0truth_9.1.json"
pause
