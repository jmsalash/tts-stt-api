import logging
import re
import threading

import numpy as np

from app.config import Settings
from app.emotions import xtts_prosody
from app.providers.base import AudioResult, TTSProvider, Voice

log = logging.getLogger(__name__)

# XTTS truncates text past a per-language character limit per generation, so we
# chunk under it. Values mirror coqui's tokenizer limits, with a small margin.
_CHAR_LIMITS = {
    "en": 230, "de": 230, "fr": 250, "es": 220, "it": 200, "pt": 190,
    "pl": 210, "zh-cn": 78, "ar": 160, "cs": 180, "ru": 175, "nl": 230,
    "tr": 210, "hu": 210, "ko": 90, "ja": 66, "hi": 140,
}
_DEFAULT_LIMIT = 200

# sentence terminators: ASCII (need trailing space) + CJK (no spaces in JA/ZH)
_PRIMARY_SPLIT = re.compile(r"(?<=[。！？])\s*|(?<=[.!?…])\s+")
# secondary breakpoints for over-long sentences
_SECONDARY_SPLIT = re.compile(r"(?<=[、，,;:])\s*")


def char_limit(language: str) -> int:
    return _CHAR_LIMITS.get((language or "").lower(), _DEFAULT_LIMIT)


def _pack(units: list[str], max_len: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for u in units:
        if not u:
            continue
        if not current:
            current = u
        elif len(current) + 1 + len(u) <= max_len:
            current += " " + u
        else:
            chunks.append(current)
            current = u
    if current:
        chunks.append(current)
    return chunks


def _split_long(sentence: str, max_len: int) -> list[str]:
    if len(sentence) <= max_len:
        return [sentence]
    parts = _pack([p.strip() for p in _SECONDARY_SPLIT.split(sentence)], max_len)
    out: list[str] = []
    for p in parts:  # hard-split anything still too long (no usable breakpoints)
        while len(p) > max_len:
            out.append(p[:max_len])
            p = p[max_len:]
        if p:
            out.append(p)
    return out


def _chunk_text(text: str, max_len: int = _DEFAULT_LIMIT) -> list[str]:
    """Split into sentences, then merge adjacent ones into chunks up to max_len.

    XTTS becomes unstable on tiny fragments (it may read the period aloud or
    hallucinate), so we avoid 1-word utterances by packing short sentences
    together, while still breaking up text past the per-language length limit.
    """
    text = text.strip()
    if not text:
        return []
    units: list[str] = []
    for sentence in _PRIMARY_SPLIT.split(text):
        sentence = sentence.strip()
        if sentence:
            units.extend(_split_long(sentence, max_len))
    return _pack(units, max_len) or [text]

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

        # We do our own sentence-aware chunking and disable XTTS's internal
        # splitter, which over-fragments short text and triggers artefacts.
        kwargs = dict(
            language=language,
            speed=float(final_speed),
            temperature=float(temperature),
            split_sentences=False,
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

        chunks = _chunk_text(text, max_len=char_limit(language))
        gap = np.zeros(int(0.15 * self._sr), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for chunk in chunks:
            wav = np.asarray(model.tts(text=chunk, **kwargs), dtype=np.float32)
            pieces.append(wav)
            pieces.append(gap)
        samples = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
        return AudioResult(samples=samples, sample_rate=self._sr)
