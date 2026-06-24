#!/bin/sh
set -eu
Xvfb :99 -screen 0 1024x768x24 -ac &
export DISPLAY=:99
exec uvicorn minimal_server:app --host 0.0.0.0 --port 8000
