import logging
import threading

import numpy as np

from app.config import Settings
from app.emotions import xtts_prosody
from app.providers.base import AudioResult, TTSProvider, Voice

log = logging.getLogger(__name__)

XTTS_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
]


class XTTSProvider(TTSProvider):
    name = "xtts"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._sr = 24000
        self._lock = threading.Lock()

    def _resolve_device(self) -> str:
        dev = self.settings.xtts_device
        if dev != "auto":
            return dev
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from TTS.api import TTS

            device = self._resolve_device()
            log.info("Loading XTTS model %s on %s", self.settings.xtts_model, device)
            model = TTS(self.settings.xtts_model).to(device)
            self._sr = getattr(model.synthesizer, "output_sample_rate", 24000)
            self._model = model
            return model

    def _builtin_speakers(self) -> list[str]:
        model = self._ensure_model()
        speakers = getattr(model, "speakers", None) or []
        return list(speakers)

    def _custom_voices(self) -> dict[str, str]:
        out = {}
        for p in self.settings.voices_dir.glob("*"):
            if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                out[p.stem] = str(p)
        return out

    def list_voices(self) -> list[Voice]:
        voices: list[Voice] = []
        for spk in self._builtin_speakers():
            voices.append(
                Voice(
                    id=spk,
                    name=spk,
                    provider=self.name,
                    languages=XTTS_LANGUAGES,
                    description="Built-in XTTS studio speaker",
                    builtin=True,
                )
            )
        for vid in self._custom_voices():
            voices.append(
                Voice(
                    id=vid,
                    name=vid,
                    provider=self.name,
                    languages=XTTS_LANGUAGES,
                    description="Cloned voice from uploaded reference sample",
                    builtin=False,
                )
            )
        return voices

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
        speed: float = 1.0,
        **params,
    ) -> AudioResult:
        model = self._ensure_model()
        language = (language or self.settings.default_language).lower()
        if language not in XTTS_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported by XTTS. "
                f"Supported: {', '.join(XTTS_LANGUAGES)}"
            )

        emo_speed, emo_temp = xtts_prosody(emotion)
        final_speed = speed * (emo_speed or 1.0)
        temperature = params.get("temperature", emo_temp if emo_temp is not None else 0.70)

        kwargs = dict(
            text=text,
            language=language,
            speed=float(final_speed),
            temperature=float(temperature),
            split_sentences=True,
        )

        custom = self._custom_voices()
        if voice and voice in custom:
            kwargs["speaker_wav"] = custom[voice]
        elif voice:
            kwargs["speaker"] = voice
        else:
            builtin = self._builtin_speakers()
            if builtin:
                kwargs["speaker"] = builtin[0]

        for opt in ("length_penalty", "repetition_penalty", "top_k", "top_p"):
            if opt in params:
                kwargs[opt] = params[opt]

        wav = model.tts(**kwargs)
        samples = np.asarray(wav, dtype=np.float32)
        return AudioResult(samples=samples, sample_rate=self._sr)
