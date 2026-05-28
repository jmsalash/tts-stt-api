#!/usr/bin/env bash
# Local TTS setup: PyTorch (CUDA 12.1 wheels) + coqui-tts (XTTS-v2).
# WARNING: downloads ~2.5GB (torch) + ~1.8GB model on first synthesis.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Run ./setup.sh first." >&2
  exit 1
fi
. .venv/bin/activate

# GPU build: CUDA 12.1 wheels work with CUDA 12.x drivers (including older
# Pascal-era cards). For CPU-only, swap the index URL for .../whl/cpu.
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-tts.txt

echo
echo "TTS install done. Set TTS_PROVIDER=xtts in .env and start with ./run.sh"
echo "Note: XTTS-v2 is under the non-commercial Coqui Public Model License;"
echo "run.sh sets COQUI_TOS_AGREED=1 to accept it for the model download."
