@echo off
echo ========================================
echo Switching to PRODUCTION MODE (all collections)
echo ========================================
echo.

python switch_to_test_collections.py --mode production

echo.
pause
