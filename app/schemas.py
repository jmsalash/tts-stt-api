from pydantic import BaseModel, Field

from app.audio_utils import SUPPORTED_FORMATS
from app.emotions import SUPPORTED_EMOTIONS


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str | None = Field(None, description="Voice/speaker id from GET /v1/voices")
    language: str | None = Field(None, description="ISO code, e.g. 'en', 'it'")
    emotion: str | None = Field(
        None, description=f"One of: {', '.join(SUPPORTED_EMOTIONS)}"
    )
    speed: float = Field(1.0, ge=0.5, le=2.0)
    format: str = Field("wav", description=f"One of: {', '.join(SUPPORTED_FORMATS)}")
    sanitize: bool | None = Field(
        None, description="Strip markdown/symbols so they aren't spoken. Defaults to server setting."
    )
    # advanced, provider-specific (temperature, top_k, model_id, ...)
    params: dict = Field(default_factory=dict)


class VoiceInfo(BaseModel):
    id: str
    name: str
    provider: str
    languages: list[str]
    description: str = ""
    builtin: bool = True


class VoicesResponse(BaseModel):
    provider: str
    voices: list[VoiceInfo]


class SegmentInfo(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[SegmentInfo] = []


class VoiceUploadResponse(BaseModel):
    id: str
    message: str


class ConfigUpdate(BaseModel):
    tts_provider: str | None = None
    stt_provider: str | None = None
    whisper_model: str | None = None


class HealthResponse(BaseModel):
    status: str
    tts_provider: str
    stt_provider: str
    supported_formats: list[str]
    supported_emotions: list[str]
