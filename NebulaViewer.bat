@echo off
rem NebulaViewer launcher (doble clic = bipolar 128^3). Ejemplos:
rem   NebulaViewer.bat --scenario bubbles
rem   NebulaViewer.bat --res 192 --width 1920 --height 1080
python "%~dp0viewer.py" %*
if errorlevel 1 pause
