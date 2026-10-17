@echo off
rem exp9.3 on panteon: the larger models, model calls only (indexes and 1bcoder run here).
rem Pairs that already have output-<id>.txt and stats-<id>.json are skipped.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_local.ps1 -Ctx 8192 -OllamaHost http://panteon.local:11434 -Models gemma4:e4b,ornith:9b,gpt-oss:20b,qwen3.6:35b-a3b
python score.py "%~dp0truth_9.1.json"
pause
