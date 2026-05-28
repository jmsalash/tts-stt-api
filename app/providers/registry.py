from app.config import Settings, get_settings
from app.providers.base import STTProvider, TTSProvider

_tts: TTSProvider | None = None
_stt: STTProvider | None = None


def _build_tts(settings: Settings) -> TTSProvider:
    p = settings.tts_provider.lower()
    if p == "xtts":
        from app.providers.tts_xtts import XTTSProvider

        return XTTSProvider(settings)
    if p == "elevenlabs":
        from app.providers.tts_elevenlabs import ElevenLabsProvider

        return ElevenLabsProvider(settings)
    raise ValueError(f"Unknown tts_provider: {settings.tts_provider}")


def _build_stt(settings: Settings) -> STTProvider:
    p = settings.stt_provider.lower()
    if p == "faster_whisper":
        from app.providers.stt_whisper import FasterWhisperProvider

        return FasterWhisperProvider(settings)
    if p == "openai":
        from app.providers.stt_openai import OpenAITranscriptionProvider

        return OpenAITranscriptionProvider(settings)
    raise ValueError(f"Unknown stt_provider: {settings.stt_provider}")


def get_tts_provider() -> TTSProvider:
    global _tts
    if _tts is None:
        _tts = _build_tts(get_settings())
    return _tts


def get_stt_provider() -> STTProvider:
    global _stt
    if _stt is None:
        _stt = _build_stt(get_settings())
    return _stt


def reset_tts() -> None:
    global _tts
    _tts = None


def reset_stt() -> None:
    global _stt
    _stt = None
