#!/usr/bin/env bash
# wait-for-it.sh — Block until a TCP host:port becomes available.
# Usage: ./scripts/wait-for-it.sh redis:6379 -- uvicorn app.main:app
# Copied from https://github.com/vishnubob/wait-for-it (MIT)

set -e
TIMEOUT=30
HOST="$1"
shift
PORT="$1"
shift

for i in $(seq $TIMEOUT); do
  if nc -z "$HOST" "$PORT" 2>/dev/null; then
    echo "[wait-for-it] $HOST:$PORT is available."
    exec "$@"
    exit 0
  fi
  echo "[wait-for-it] Waiting for $HOST:$PORT... ($i/$TIMEOUT)"
  sleep 1
done

echo "[wait-for-it] Timeout after ${TIMEOUT}s waiting for $HOST:$PORT"
exit 1
