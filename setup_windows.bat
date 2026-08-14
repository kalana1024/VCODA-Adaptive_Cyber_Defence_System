@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" -Profile full
if errorlevel 1 pause
