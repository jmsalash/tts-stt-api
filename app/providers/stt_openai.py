import httpx

from app.config import Settings
from app.providers.base import STTProvider, Segment, TranscriptionResult


class OpenAITranscriptionProvider(STTProvider):
    name = "openai"

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.openai_api_key:
            raise RuntimeError("openai_api_key is not set (OPENAI_API_KEY env var).")

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        **params,
    ) -> TranscriptionResult:
        cfg = self.settings
        data = {"model": cfg.openai_stt_model, "response_format": "verbose_json"}
        if language:
            data["language"] = language

        with open(audio_path, "rb") as fh:
            files = {"file": (audio_path.rsplit("/", 1)[-1], fh, "application/octet-stream")}
            r = httpx.post(
                f"{cfg.openai_base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {cfg.openai_api_key}"},
                data=data,
                files=files,
                timeout=300,
            )
        r.raise_for_status()
        body = r.json()
        segments = [
            Segment(
                start=round(s.get("start", 0.0), 3),
                end=round(s.get("end", 0.0), 3),
                text=s.get("text", "").strip(),
            )
            for s in body.get("segments", [])
        ]
        return TranscriptionResult(
            text=body.get("text", "").strip(),
            language=body.get("language", language or "unknown"),
            duration=round(body.get("duration", 0.0), 3),
            segments=segments,
        )
