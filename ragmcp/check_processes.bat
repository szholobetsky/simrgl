@echo off
echo ============================================================
echo CHECKING FOR RUNNING PYTHON PROCESSES
echo ============================================================
echo.

echo [1/4] All Python processes:
tasklist | findstr /I python

echo.
echo [2/4] Checking for MCP server processes:
wmic process where "commandline like '%%mcp_server%%'" get processid,commandline 2>nul
if errorlevel 1 echo No MCP server processes found

echo.
echo [3/4] Checking for migration processes:
wmic process where "commandline like '%%migrate_rawdata%%'" get processid,commandline 2>nul
if errorlevel 1 echo No migration processes found

echo.
echo [4/4] Checking for two-phase agent processes:
wmic process where "commandline like '%%two_phase_agent%%'" get processid,commandline 2>nul
if errorlevel 1 echo No two-phase agent processes found

echo.
echo ============================================================
echo If you see multiple MCP server processes, kill them with:
echo   taskkill /F /PID [process_id]
echo ============================================================
pause
