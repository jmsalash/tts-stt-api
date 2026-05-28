#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
# Accept the XTTS-v2 (Coqui Public Model License, non-commercial) for the
# automatic model download. Remove if you don't use the local XTTS provider.
export COQUI_TOS_AGREED="${COQUI_TOS_AGREED:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Optional HTTPS/TLS: set SSL_CERTFILE and SSL_KEYFILE to serve over https.
SSL_ARGS=()
SCHEME="http"
if [ -n "${SSL_CERTFILE:-}" ] && [ -n "${SSL_KEYFILE:-}" ]; then
  SSL_ARGS=(--ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE")
  SCHEME="https"
fi

echo "Starting TTS/STT API on ${SCHEME}://${HOST}:${PORT}  (UI at /, docs at /docs)"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" "${SSL_ARGS[@]}" "$@"
