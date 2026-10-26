@echo off
cd /d "%~dp0"
uv run python bridge_tui.py
if errorlevel 1 pause
