@echo off
set NO_PROXY=localhost,127.0.0.1
set HTTP_PROXY=
set HTTPS_PROXY=
echo Starting Fake News Detector on Port 5001...
python app.py
pause
