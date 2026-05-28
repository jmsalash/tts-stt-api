"""Emotion / intonation presets shared across TTS providers.

Local XTTS has no explicit emotion input, so we approximate emotion by
nudging prosody parameters (speed, sampling temperature). The most reliable
way to get a specific emotional delivery from XTTS is still to clone from a
reference clip that already carries that emotion. Cloud providers (ElevenLabs)
expose richer, native emotion controls and are mapped separately.
"""

# name -> (speed_multiplier, temperature)
XTTS_PRESETS: dict[str, tuple[float, float]] = {
    "neutral": (1.00, 0.70),
    "happy": (1.08, 0.80),
    "cheerful": (1.10, 0.85),
    "excited": (1.15, 0.90),
    "sad": (0.90, 0.55),
    "calm": (0.92, 0.50),
    "serious": (0.96, 0.55),
    "angry": (1.06, 0.88),
    "fearful": (1.05, 0.80),
    "whisper": (0.95, 0.45),
}

# name -> ElevenLabs voice_settings overrides (style in [0,1], stability in [0,1])
ELEVEN_PRESETS: dict[str, dict[str, float]] = {
    "neutral": {"stability": 0.5, "style": 0.0},
    "happy": {"stability": 0.4, "style": 0.55},
    "cheerful": {"stability": 0.35, "style": 0.6},
    "excited": {"stability": 0.3, "style": 0.75},
    "sad": {"stability": 0.7, "style": 0.3},
    "calm": {"stability": 0.8, "style": 0.1},
    "serious": {"stability": 0.75, "style": 0.15},
    "angry": {"stability": 0.35, "style": 0.7},
    "fearful": {"stability": 0.45, "style": 0.6},
    "whisper": {"stability": 0.85, "style": 0.2},
}

SUPPORTED_EMOTIONS = sorted(XTTS_PRESETS.keys())


def xtts_prosody(emotion: str | None) -> tuple[float | None, float | None]:
    if not emotion:
        return None, None
    return XTTS_PRESETS.get(emotion.lower(), (None, None))


def eleven_settings(emotion: str | None) -> dict[str, float]:
    if not emotion:
        return {}
    return dict(ELEVEN_PRESETS.get(emotion.lower(), {}))
