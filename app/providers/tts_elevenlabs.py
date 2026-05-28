import httpx

from app.audio_utils import pcm16_to_float32
from app.config import Settings
from app.emotions import eleven_settings
from app.providers.base import AudioResult, TTSProvider, Voice

API_ROOT = "https://api.elevenlabs.io/v1"
SAMPLE_RATE = 24000


class ElevenLabsProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.elevenlabs_api_key:
            raise RuntimeError(
                "elevenlabs_api_key is not set (ELEVENLABS_API_KEY env var)."
            )
        self._headers = {"xi-api-key": settings.elevenlabs_api_key}

    def list_voices(self) -> list[Voice]:
        r = httpx.get(f"{API_ROOT}/voices", headers=self._headers, timeout=30)
        r.raise_for_status()
        out = []
        for v in r.json().get("voices", []):
            labels = v.get("labels", {}) or {}
            desc = ", ".join(f"{k}={val}" for k, val in labels.items())
            out.append(
                Voice(
                    id=v["voice_id"],
                    name=v.get("name", v["voice_id"]),
                    provider=self.name,
                    languages=["multi"],
                    description=desc,
                    builtin=True,
                )
            )
        return out

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
        speed: float = 1.0,
        **params,
    ) -> AudioResult:
        if not voice:
            raise ValueError("ElevenLabs requires a 'voice' id (see GET /v1/voices).")

        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        }
        voice_settings.update(eleven_settings(emotion))
        if speed and speed != 1.0:
            voice_settings["speed"] = float(speed)

        payload = {
            "text": text,
            "model_id": params.get("model_id", self.settings.elevenlabs_model),
            "voice_settings": voice_settings,
        }
        if language:
            payload["language_code"] = language

        r = httpx.post(
            f"{API_ROOT}/text-to-speech/{voice}",
            params={"output_format": "pcm_24000"},
            headers={**self._headers, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        samples = pcm16_to_float32(r.content)
        return AudioResult(samples=samples, sample_rate=SAMPLE_RATE)
