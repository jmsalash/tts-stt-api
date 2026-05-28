# TTS / STT API

A self-hosted **text-to-speech** and **speech-to-text** backend built with
FastAPI. It ships with high-quality local models, optional cloud providers, an
emotion/intonation layer, voice cloning, real-time streaming transcription, and
a small built-in web UI for testing.

## Features

- **Text → Speech** with [XTTS-v2](https://huggingface.co/coqui/XTTS-v2)
  (local): 17 languages, 58 built-in speakers, GPU-accelerated.
- **Emotion / intonation** presets (happy, sad, angry, excited, calm, whisper, …)
  plus **voice cloning** from a short reference clip.
- **Speech → Text** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (local): switchable model size (`tiny` → `large-v3`), GPU or CPU.
- **Real-time streaming STT** over WebSocket — partial transcripts while you
  speak, finalized per utterance.
- **Pluggable providers**: drop-in cloud alternatives — **ElevenLabs** (TTS) and
  **OpenAI** (STT) — selectable via config.
- **No system `ffmpeg` required**: audio is handled via `soundfile` + `PyAV`.
  Output formats: WAV, FLAC, OGG, MP3, Opus.
- **Built-in web UI** at `/` to test TTS and STT across languages.
- Optional **bearer-token auth**.

## Architecture

```
app/
  main.py            # FastAPI app: REST routes + WebSocket streaming
  config.py          # settings (env / .env)
  schemas.py         # request/response models
  audio_utils.py     # decode/encode audio (soundfile + PyAV)
  emotions.py        # emotion → prosody / provider-setting presets
  static/index.html  # web UI
  providers/
    base.py          # TTSProvider / STTProvider interfaces
    registry.py      # provider selection + lifecycle
    tts_xtts.py      # local TTS (XTTS-v2)
    tts_elevenlabs.py# cloud TTS (ElevenLabs)
    stt_whisper.py   # local STT (faster-whisper)
    stt_openai.py    # cloud STT (OpenAI)
```

Providers are loaded lazily and can be swapped at runtime via `POST /v1/config`,
so models are only downloaded when first used.

## Requirements

- Python 3.10+ (developed on 3.12)
- For local TTS: PyTorch. An **NVIDIA GPU with CUDA 12.x** is recommended (CPU
  works but synthesis is much slower).
- Disk: ~5–6 GB for the local TTS stack (PyTorch ~2.5 GB + XTTS model ~1.8 GB).
  STT models range from ~75 MB (`tiny`) to ~3 GB (`large-v3`).

## Installation

The setup scripts create a virtual environment. If your system lacks
`python3-venv`/`ensurepip`, `setup.sh` bootstraps `pip` automatically (no `sudo`).

```bash
# 1) Base API + local speech-to-text (no PyTorch)
./setup.sh

# 2) Local text-to-speech (PyTorch + XTTS-v2). Optional but recommended.
./setup_tts.sh
```

Or manually:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# local TTS:
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-tts.txt
```

## Configuration

Copy `.env.example` to `.env` and edit as needed (all values have sensible
defaults). Key options:

| Variable | Default | Description |
|---|---|---|
| `TTS_PROVIDER` | `xtts` | `xtts` (local) or `elevenlabs` (cloud) |
| `STT_PROVIDER` | `faster_whisper` | `faster_whisper` (local) or `openai` (cloud) |
| `WHISPER_MODEL` | `small` | `tiny`/`base`/`small`/`medium`/`large-v3` |
| `TTS_SANITIZE_TEXT` | `true` | Strip markdown/symbols/emoji so they aren't spoken |
| `WHISPER_DEVICE` | `auto` | `auto`/`cuda`/`cpu` |
| `XTTS_DEVICE` | `auto` | `auto`/`cuda`/`cpu` |
| `API_KEY` | _(empty)_ | If set, require `Authorization: Bearer <key>` |
| `API_KEYS` | _(empty)_ | Comma-separated bearer tokens (alternative to `API_KEY`) |
| `SSL_CERTFILE` / `SSL_KEYFILE` | _(empty)_ | Set both to serve over HTTPS |
| `ELEVENLABS_API_KEY` | _(empty)_ | Enables the ElevenLabs TTS provider |
| `OPENAI_API_KEY` | _(empty)_ | Enables the OpenAI STT provider |

## Running

```bash
./run.sh                 # http://localhost:8000  (UI at /)
# or:
. .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000> for the web UI.

> Note: the browser microphone (record / live streaming) requires a *secure
> context* — `http://localhost` works; remote access needs HTTPS.

## Authentication (optional)

Auth is off by default. To require a key, set `API_KEY=your-secret` (env or
`.env`) and restart. Every request must then include:

```
Authorization: Bearer your-secret
```

For the WebSocket streaming endpoint, pass the key as a query parameter instead
(`?token=your-secret`), since browsers can't set custom headers on a WebSocket.
In the web UI, enter the key under **Settings** (it's stored in your browser).
To turn auth off, clear `API_KEY`/`API_KEYS` and restart.

```bash
curl -H "Authorization: Bearer your-secret" \
  -X POST http://localhost:8000/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Authenticated request","format":"mp3"}' --output speech.mp3
```

## HTTPS / TLS (optional)

Two common ways to serve over `https`:

**1. Directly via uvicorn** (simplest for LAN/dev). Generate a self-signed cert,
then set the two env vars and `run.sh` serves over TLS:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \
  -days 365 -subj "/CN=localhost"
SSL_CERTFILE=./cert.pem SSL_KEYFILE=./key.pem ./run.sh
```

Browsers warn on self-signed certs (accept once). For a *trusted* LAN cert with
no warning, use [`mkcert`](https://github.com/FiloSottile/mkcert).

**2. Behind a reverse proxy** (recommended for anything real). Run the app on
plain HTTP bound to `127.0.0.1`, and put [Caddy](https://caddyserver.com) or
nginx in front to terminate TLS (Caddy auto-provisions Let's Encrypt certs if
you have a domain). Example `Caddyfile`:

```
your.domain { reverse_proxy 127.0.0.1:8000 }
```

When served over HTTPS, the web UI's same-origin calls (including the WebSocket,
which switches to `wss://`) use HTTPS automatically.

## API reference

All examples assume no auth. If `API_KEY`/`API_KEYS` is set, add
`-H "Authorization: Bearer <token>"` (and `?token=<token>` for the WebSocket).
See [Authentication](#authentication-optional) for details.

### `GET /health`
Service status, current providers, supported formats and emotions.

### `GET /v1/config` · `POST /v1/config`
Read or change providers and the Whisper model at runtime.

```bash
curl -X POST localhost:8000/v1/config \
  -H 'Content-Type: application/json' \
  -d '{"whisper_model":"large-v3"}'   # downloads on next transcription
```

### `GET /v1/voices`
List available voices for the active TTS provider.

### `POST /v1/tts`
Synthesize speech. Returns audio bytes (`Content-Type` per `format`).

```bash
curl -X POST localhost:8000/v1/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there!","language":"en","emotion":"cheerful","format":"mp3"}' \
  --output speech.mp3
```

Body fields: `text` (required), `voice`, `language`, `emotion`, `speed`
(0.5–2.0), `format` (`wav`/`flac`/`ogg`/`mp3`/`opus`), `sanitize` (override text
cleaning, see below), `params` (advanced, provider-specific, e.g. `temperature`).

**Text handling:** by default the server strips markdown and symbols that would
otherwise be spoken aloud (e.g. `**bold**`, `#`, `@`, emoji) while keeping normal
sentence punctuation (`. , ! ? ; :`) for natural prosody. Disable globally with
`TTS_SANITIZE_TEXT=false` or per request with `"sanitize": false`.

### `POST /v1/voices` — voice cloning
Upload a short, clean reference clip; reuse its `id` as the `voice` in `/v1/tts`.

```bash
curl -X POST localhost:8000/v1/voices -F 'name=my_voice' -F 'file=@sample.wav'
```

### `DELETE /v1/voices/{id}`
Remove a cloned voice.

### `POST /v1/stt`
Transcribe an uploaded audio file.

```bash
curl -X POST localhost:8000/v1/stt -F 'file=@audio.wav' -F 'language=en'
```

Returns `{ text, language, duration, segments[] }`.

### `WS /v1/stt/stream` — real-time transcription
Connect, then send **16-bit PCM, mono, 16 kHz** binary frames. Send
`{"type":"stop"}` (or disconnect) to finish. Query params: `language`, `token`.

The server replies with JSON messages:

- `{"type":"partial","text":...}` — running hypothesis for the current utterance
- `{"type":"final","text":...}` — an utterance closed by a pause
- `{"type":"done","text":...}` — full transcript when the stream ends

## Emotion & voice quality notes

Local XTTS-v2 has no explicit emotion input, so the emotion presets approximate
it by adjusting prosody (speed/temperature). For a specific emotional delivery,
the most reliable approach is **cloning from a reference clip that already
carries that emotion**. The ElevenLabs provider exposes richer native emotion
controls if you prefer a cloud option.

## Supported TTS languages

`en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi`

## License

The XTTS-v2 model is distributed under the **Coqui Public Model License (CPML)**,
which is **non-commercial**. The setup accepts this license for the automatic
model download (`COQUI_TOS_AGREED=1` in `run.sh`); review the CPML before any
commercial use, or switch `TTS_PROVIDER` to a provider whose terms suit your use
case.
