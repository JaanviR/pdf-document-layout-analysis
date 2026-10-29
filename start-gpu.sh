#!/bin/bash
set -e

WORKERS="${PDF_WORKERS:-2}"

# Start Gunicorn in background
gunicorn \
    -k uvicorn.workers.UvicornWorker \
    -w "$WORKERS" \
    --chdir ./src \
    drivers.rest.app:app \
    --bind 0.0.0.0:5060 \
    --timeout 10000 &
GUNICORN_PID=$!

# Warmup: send one request per worker to force GPU memory allocation
python3 /app/src/warmup_workers.py "$WORKERS"

# Bring Gunicorn back to foreground
wait $GUNICORN_PID
