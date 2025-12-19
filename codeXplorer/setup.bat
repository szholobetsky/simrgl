@echo off
REM Setup script for CodeXplorer Data Gathering Tool (Windows)

echo Setting up CodeXplorer Data Gathering Tool...

REM Check Python version
python --version

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the tool, edit config.py and then run:
echo   python main.py
echo.
pause
