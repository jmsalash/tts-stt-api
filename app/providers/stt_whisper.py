import logging
import threading

from app.config import Settings
from app.providers.base import STTProvider, Segment, TranscriptionResult

log = logging.getLogger(__name__)


class FasterWhisperProvider(STTProvider):
    name = "faster_whisper"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _load(self):
        from faster_whisper import WhisperModel

        cfg = self.settings
        compute = cfg.whisper_compute_type
        download_root = str(cfg.cache_dir)

        def make(device: str, compute_type: str):
            log.info(
                "Loading faster-whisper '%s' on %s (%s)",
                cfg.whisper_model, device, compute_type,
            )
            return WhisperModel(
                cfg.whisper_model,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )

        order = []
        if cfg.whisper_device in ("auto", "cuda"):
            order.append(("cuda", compute if compute != "auto" else "int8"))
        order.append(("cpu", compute if compute != "auto" else "int8"))

        last_err = None
        for device, compute_type in order:
            try:
                return make(device, compute_type)
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("faster-whisper load failed on %s: %s", device, e)
        raise RuntimeError(f"Could not load faster-whisper model: {last_err}")

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._model = self._load()
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        **params,
    ) -> TranscriptionResult:
        model = self._ensure_model()
        with self._infer_lock:
            segments_iter, info = model.transcribe(
                audio_path,
                language=language,
                beam_size=params.get("beam_size", self.settings.whisper_beam_size),
                vad_filter=params.get("vad_filter", self.settings.whisper_vad),
            )
            segments = [
                Segment(start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip())
                for s in segments_iter
            ]
        text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            duration=round(info.duration, 3),
            segments=segments,
        )

    def transcribe_samples(
        self,
        samples,
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = False,
    ) -> TranscriptionResult:
        """Transcribe an in-memory float32 mono array at 16 kHz (for streaming)."""
        model = self._ensure_model()
        with self._infer_lock:
            segments_iter, info = model.transcribe(
                samples,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                condition_on_previous_text=False,
            )
            segments = [
                Segment(start=round(s.start, 3), end=round(s.end, 3), text=s.text.strip())
                for s in segments_iter
            ]
        text = " ".join(s.text for s in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            duration=round(info.duration, 3),
            segments=segments,
        )
