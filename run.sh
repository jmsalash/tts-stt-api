#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
# Accept the XTTS-v2 (Coqui Public Model License, non-commercial) for the
# automatic model download. Remove if you don't use the local XTTS provider.
export COQUI_TOS_AGREED="${COQUI_TOS_AGREED:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
