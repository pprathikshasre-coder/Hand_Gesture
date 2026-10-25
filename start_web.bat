@echo off
echo Starting Hand Gesture Recognition Web Studio...
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start http://localhost:5000
python web_app.py
pause
