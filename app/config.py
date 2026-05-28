from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    # Comma-separated bearer tokens. Empty => auth disabled.
    api_keys: str = ""
    cors_origins: str = "*"
    max_upload_mb: int = 50

    # --- provider selection ---
    tts_provider: str = "xtts"            # xtts | elevenlabs
    stt_provider: str = "faster_whisper"  # faster_whisper | openai

    default_language: str = "en"

    # --- storage ---
    voices_dir: Path = BASE_DIR / "data" / "voices"
    cache_dir: Path = BASE_DIR / "data" / "cache"

    # --- XTTS (local TTS) ---
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_device: str = "auto"   # auto | cuda | cpu

    # --- faster-whisper (local STT) ---
    whisper_model: str = "small"        # tiny|base|small|medium|large-v3 ...
    whisper_device: str = "auto"        # auto | cuda | cpu
    whisper_compute_type: str = "auto"  # auto | int8 | int8_float16 | float16 | float32
    whisper_beam_size: int = 5
    whisper_vad: bool = True

    # --- ElevenLabs (cloud TTS) ---
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"

    # --- OpenAI (cloud STT) ---
    openai_api_key: str = ""
    openai_stt_model: str = "whisper-1"
    openai_base_url: str = "https://api.openai.com/v1"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()] or ["*"]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.voices_dir.mkdir(parents=True, exist_ok=True)
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    return s
