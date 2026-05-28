import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.audio_utils import SUPPORTED_FORMATS, encode_audio, load_audio_file
from app.config import get_settings
from app.emotions import SUPPORTED_EMOTIONS
from app.providers.registry import get_stt_provider, get_tts_provider, reset_stt, reset_tts
from app.providers.tts_xtts import XTTS_LANGUAGES
from app.schemas import (
    ConfigUpdate,
    HealthResponse,
    TranscriptionResponse,
    TTSRequest,
    VoiceInfo,
    VoicesResponse,
    VoiceUploadResponse,
)

ALLOWED_WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
STATIC_DIR = Path(__file__).resolve().parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tts_stt_api")

settings = get_settings()
app = FastAPI(title="TTS/STT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


def require_auth(authorization: str | None = Header(None)):
    keys = settings.api_key_set
    if not keys:
        return
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/")
def ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"name": "TTS/STT API", "ui": "not found", "docs": "/docs"}


def _available_tts_providers() -> list[str]:
    providers = ["xtts"]
    if settings.elevenlabs_api_key:
        providers.append("elevenlabs")
    return providers


def _available_stt_providers() -> list[str]:
    providers = ["faster_whisper"]
    if settings.openai_api_key:
        providers.append("openai")
    return providers


def _config_payload() -> dict:
    return {
        "tts_provider": settings.tts_provider,
        "stt_provider": settings.stt_provider,
        "whisper_model": settings.whisper_model,
        "tts_providers": _available_tts_providers(),
        "stt_providers": _available_stt_providers(),
        "whisper_models": ALLOWED_WHISPER_MODELS,
        "tts_languages": XTTS_LANGUAGES,
        "emotions": SUPPORTED_EMOTIONS,
        "formats": SUPPORTED_FORMATS,
        "auth_required": bool(settings.api_key_set),
    }


@app.get("/v1/config", dependencies=[Depends(require_auth)])
def get_config():
    return _config_payload()


@app.post("/v1/config", dependencies=[Depends(require_auth)])
def set_config(body: ConfigUpdate):
    if body.whisper_model is not None:
        if body.whisper_model not in ALLOWED_WHISPER_MODELS:
            raise HTTPException(status_code=400, detail=f"whisper_model must be one of {ALLOWED_WHISPER_MODELS}")
        if body.whisper_model != settings.whisper_model:
            settings.whisper_model = body.whisper_model
            reset_stt()
    if body.stt_provider is not None:
        if body.stt_provider not in _available_stt_providers():
            raise HTTPException(status_code=400, detail="stt_provider unavailable (missing API key?)")
        if body.stt_provider != settings.stt_provider:
            settings.stt_provider = body.stt_provider
            reset_stt()
    if body.tts_provider is not None:
        if body.tts_provider not in _available_tts_providers():
            raise HTTPException(status_code=400, detail="tts_provider unavailable (missing API key?)")
        if body.tts_provider != settings.tts_provider:
            settings.tts_provider = body.tts_provider
            reset_tts()
    return _config_payload()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        tts_provider=settings.tts_provider,
        stt_provider=settings.stt_provider,
        supported_formats=SUPPORTED_FORMATS,
        supported_emotions=SUPPORTED_EMOTIONS,
    )


@app.get("/v1/voices", response_model=VoicesResponse, dependencies=[Depends(require_auth)])
def list_voices():
    provider = get_tts_provider()
    try:
        voices = provider.list_voices()
    except Exception as e:  # noqa: BLE001
        log.exception("list_voices failed")
        raise HTTPException(status_code=500, detail=str(e))
    return VoicesResponse(
        provider=provider.name,
        voices=[VoiceInfo(**v.__dict__) for v in voices],
    )


@app.post("/v1/tts", dependencies=[Depends(require_auth)])
def synthesize(req: TTSRequest):
    fmt = req.format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"format must be one of {SUPPORTED_FORMATS}")
    provider = get_tts_provider()
    try:
        result = provider.synthesize(
            text=req.text,
            voice=req.voice,
            language=req.language,
            emotion=req.emotion,
            speed=req.speed,
            **req.params,
        )
        audio_bytes, content_type = encode_audio(result.samples, result.sample_rate, fmt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        log.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="speech.{fmt}"',
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


@app.post("/v1/voices", response_model=VoiceUploadResponse, dependencies=[Depends(require_auth)])
async def upload_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    voice_id = _SAFE.sub("-", name).strip("-").lower()
    if not voice_id:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")

    suffix = Path(file.filename or "ref").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            samples, sr = load_audio_file(tmp.name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Unsupported audio: {e}")

    out_path = settings.voices_dir / f"{voice_id}.wav"
    sf.write(str(out_path), samples, sr, subtype="PCM_16")
    return VoiceUploadResponse(id=voice_id, message="Voice reference stored. Use it as 'voice' in /v1/tts.")


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(require_auth)])
def delete_voice(voice_id: str):
    safe = _SAFE.sub("-", voice_id).strip("-").lower()
    path = settings.voices_dir / f"{safe}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Voice not found")
    path.unlink()
    return {"deleted": safe}


@app.post("/v1/stt", response_model=TranscriptionResponse, dependencies=[Depends(require_auth)])
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(None),
):
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")

    suffix = Path(file.filename or "audio").suffix or ".wav"
    provider = get_stt_provider()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        try:
            result = provider.transcribe(tmp.name, language=language)
        except Exception as e:  # noqa: BLE001
            log.exception("transcription failed")
            raise HTTPException(status_code=500, detail=str(e))

    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        duration=result.duration,
        segments=[s.__dict__ for s in result.segments],
    )


# --- streaming STT over WebSocket ---
# Client sends 16-bit PCM mono @ 16 kHz binary frames, plus optional JSON control
# messages ({"type":"stop"}). Server replies with JSON:
#   {"type":"partial","text":...}  – running hypothesis for the current utterance
#   {"type":"final","text":...}    – an utterance closed by a pause
#   {"type":"done","text":...}     – full transcript when the stream ends
STREAM_SR = 16000


@app.websocket("/v1/stt/stream")
async def stt_stream(ws: WebSocket):
    await ws.accept()

    keys = settings.api_key_set
    if keys and ws.query_params.get("token", "") not in keys:
        await ws.close(code=4401)
        return

    provider = get_stt_provider()
    if provider.name != "faster_whisper":
        await ws.send_json({"type": "error", "detail": "Streaming requires the faster_whisper STT provider."})
        await ws.close()
        return

    language = ws.query_params.get("language") or None

    silence_rms = 0.012
    speech_rms = 0.02
    min_silence = int(0.7 * STREAM_SR)
    partial_step = int(1.0 * STREAM_SR)
    max_utterance = int(24 * STREAM_SR)
    min_final = int(0.25 * STREAM_SR)

    buf = np.zeros(0, dtype=np.float32)
    spoke = False
    last_partial = 0
    full: list[str] = []

    async def run(samples):
        return await asyncio.to_thread(provider.transcribe_samples, samples, language)

    async def finalize(samples):
        if len(samples) < min_final:
            return
        res = await run(samples)
        text = res.text.strip()
        if text:
            full.append(text)
            await ws.send_json({"type": "final", "text": text})

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data:
                chunk = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                buf = np.concatenate([buf, chunk])
                if not spoke and len(chunk) and float(np.sqrt(np.mean(chunk ** 2))) > speech_rms:
                    spoke = True

                if spoke and len(buf) - last_partial >= partial_step:
                    last_partial = len(buf)
                    res = await run(buf)
                    await ws.send_json({"type": "partial", "text": res.text.strip()})

                tail_silent = (
                    len(buf) >= min_silence
                    and float(np.sqrt(np.mean(buf[-min_silence:] ** 2))) < silence_rms
                )
                if spoke and (tail_silent or len(buf) >= max_utterance):
                    await finalize(buf)
                    buf = np.zeros(0, dtype=np.float32)
                    spoke = False
                    last_partial = 0
                continue

            text = msg.get("text")
            if text:
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    ctrl = {"type": text}
                if ctrl.get("type") == "stop":
                    break

        if spoke:
            await finalize(buf)
        await ws.send_json({"type": "done", "text": " ".join(full)})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("stream failed")
        try:
            await ws.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass
