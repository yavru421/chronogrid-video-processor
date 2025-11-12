@echo off
REM Chronogrid GUI Launcher (Windows)
REM This script activates the venv and launches the PyQt GUI

SET VENV_DIR=.\venv
SET PYTHON_EXE=%VENV_DIR%\Scripts\python.exe

IF NOT EXIST %PYTHON_EXE% (
    echo [ERROR] Python virtual environment not found. Please run:
    echo    python -m venv venv
    echo    pip install -r requirements.txt
    exit /b 1
)

REM Activate venv and run GUI
call %PYTHON_EXE% chronogrid-gui-pyqt.py
