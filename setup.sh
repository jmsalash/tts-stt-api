#!/usr/bin/env bash
# Base setup: virtualenv + API + local STT (faster-whisper). No PyTorch.
set -euo pipefail
cd "$(dirname "$0")"

# Create the venv. If the system lacks ensurepip (python3-venv not installed),
# fall back to a pip-less venv and bootstrap pip via get-pip.py (no sudo needed).
if python3 -m venv .venv 2>/dev/null && [ -x .venv/bin/pip ]; then
  :
else
  rm -rf .venv
  python3 -m venv --without-pip .venv
  .venv/bin/python -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
  .venv/bin/python /tmp/get-pip.py
fi

. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

[ -f .env ] || cp .env.example .env
echo
echo "Base install done. Run ./run.sh to start the API (TTS needs ./setup_tts.sh)."
