@echo off
setlocal
cd /d %~dp0
".venv\Scripts\python.exe" start_desktop.py
endlocal
