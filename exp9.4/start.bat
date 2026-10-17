@echo off
rem exp9.4: qwen3:1.7b here, gemma4:e4b and gpt-oss:20b on panteon, all 10 tasks, in two windows.
rem Pairs already done are skipped, so this also resumes a stopped run.
cd /d "%~dp0"
start "exp9.4 local" powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 -OllamaHost http://localhost:11434 -Models qwen3:1.7b
start "exp9.4 panteon" powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1 -OllamaHost http://panteon.local:11434 -Models gemma4:e4b,gpt-oss:20b
