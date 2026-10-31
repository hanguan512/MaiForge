"""Audio decoding and timing analysis."""

from .io import AudioData, find_audio_file, find_maidata_file, load_audio, write_wav
from .timing import TimingInfo

__all__ = [
    "AudioData",
    "TimingInfo",
    "find_audio_file",
    "find_maidata_file",
    "load_audio",
    "write_wav",
]


def analyze_audio(
    path: str,
    sr: int = 22050,
    *,
    bpm: float | None = None,
    first: float = 0.0,
) -> TimingInfo:
    from .features import analyze_audio as _analyze

    return _analyze(path, sr=sr, bpm=bpm, first=first)
