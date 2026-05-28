from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Voice:
    id: str
    name: str
    provider: str
    languages: list[str] = field(default_factory=list)
    description: str = ""
    builtin: bool = True


@dataclass
class AudioResult:
    """Always mono float32 samples in [-1, 1] plus the sample rate."""

    samples: np.ndarray
    sample_rate: int


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)


class TTSProvider(ABC):
    name: str = "base"

    @abstractmethod
    def list_voices(self) -> list[Voice]:
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
        speed: float = 1.0,
        **params,
    ) -> AudioResult:
        ...


class STTProvider(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        **params,
    ) -> TranscriptionResult:
        ...
