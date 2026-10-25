@echo off
echo Starting Hand Gesture Recognition Dashboard...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run.py
pause
