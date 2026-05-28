import io

import numpy as np

# format -> (content_type, soundfile_subtype or None, av_codec or None)
WAV = "wav"
FLAC = "flac"
OGG = "ogg"
MP3 = "mp3"
OPUS = "opus"

CONTENT_TYPES = {
    WAV: "audio/wav",
    FLAC: "audio/flac",
    OGG: "audio/ogg",
    MP3: "audio/mpeg",
    OPUS: "audio/ogg",
}

SUPPORTED_FORMATS = list(CONTENT_TYPES.keys())


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """Decode any common audio file to mono float32 + sample rate.

    Tries libsndfile (wav/flac/ogg) first, then PyAV (mp3/m4a/...), so no
    system ffmpeg binary is required.
    """
    try:
        import soundfile as sf

        samples, sr = sf.read(path, dtype="float32", always_2d=False)
        return _to_mono_float32(samples), int(sr)
    except Exception:
        pass

    import av

    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono")
    chunks = []
    sr = stream.rate or 24000
    for frame in container.decode(stream):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        raise ValueError("Could not decode audio file")
    pcm = np.concatenate(chunks).astype("<i2").tobytes()
    return pcm16_to_float32(pcm), int(sr)


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1 if samples.shape[1] <= samples.shape[0] else 0)
    return np.ascontiguousarray(samples)


def encode_audio(samples: np.ndarray, sample_rate: int, fmt: str) -> tuple[bytes, str]:
    """Encode mono float32 samples to the requested container/codec.

    WAV/FLAC/OGG go through libsndfile; MP3/OPUS through PyAV so no system
    ffmpeg binary is required.
    """
    fmt = fmt.lower()
    if fmt not in CONTENT_TYPES:
        raise ValueError(f"Unsupported format: {fmt}")
    samples = _to_mono_float32(samples)

    if fmt in (WAV, FLAC, OGG):
        import soundfile as sf

        subtype = {"wav": "PCM_16", "flac": "PCM_16", "ogg": "VORBIS"}[fmt]
        fmt_name = {"wav": "WAV", "flac": "FLAC", "ogg": "OGG"}[fmt]
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format=fmt_name, subtype=subtype)
        return buf.getvalue(), CONTENT_TYPES[fmt]

    # mp3 / opus via PyAV (bundled with faster-whisper's `av` dependency)
    return _encode_av(samples, sample_rate, fmt), CONTENT_TYPES[fmt]


def _encode_av(samples: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    import av

    codec = {"mp3": "mp3", "opus": "libopus"}[fmt]
    container_fmt = {"mp3": "mp3", "opus": "ogg"}[fmt]

    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_fmt)
    stream = container.add_stream(codec, rate=sample_rate)
    stream.layout = "mono"

    frame = av.AudioFrame.from_ndarray(pcm16.reshape(1, -1), format="s16", layout="mono")
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):  # flush
        container.mux(packet)
    container.close()
    return buf.getvalue()
