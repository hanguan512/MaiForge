"""Audio I/O with ffmpeg fallback.

Only numpy is required for the audio phase.  ffmpeg is used to decode
mp3/flac/m4a and to resample; plain WAV works without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aiff", ".aif")


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "numpy is required for audio analysis. "
            "Install it with: python3 -m pip install numpy"
        ) from exc
    return np


@dataclass
class AudioData:
    samples: Any
    sr: int
    path: str
    tags: dict[str, str]

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sr": self.sr,
            "duration": self.duration,
            "tags": self.tags,
        }


def find_audio_file(directory: str | Path) -> Path | None:
    directory = Path(directory)
    for ext in AUDIO_EXTENSIONS:
        candidates = sorted(directory.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def find_maidata_file(directory: str | Path) -> Path | None:
    directory = Path(directory)
    for name in ("maidata.txt", "maidata"):
        path = directory / name
        if path.exists():
            return path
    candidates = sorted(directory.glob("maidata*"))
    return candidates[0] if candidates else None


def load_audio(path: str | Path, sr: int = 22050) -> AudioData:
    np = _numpy()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    samples: Any
    if shutil.which("ffmpeg"):
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {path}: {proc.stderr.decode('utf-8', 'ignore')[:500]}"
            )
        samples = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    elif path.suffix.lower() == ".wav":
        samples = _load_wav(path, sr)
    else:
        raise RuntimeError(
            f"ffmpeg is required to decode {path.suffix}; install ffmpeg or provide a WAV"
        )

    if samples.size == 0:
        raise RuntimeError(f"decoded audio is empty: {path}")
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak * 0.98
    return AudioData(samples=samples, sr=sr, path=str(path), tags=_read_tags(path))


def _load_wav(path: Path, target_sr: int):
    np = _numpy()
    with wave.open(str(path), "rb") as wav:
        n_channels = wav.getnchannels()
        sampwidth = wav.getsampwidth()
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        raw = wav.readframes(n_frames)
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw_bytes = np.frombuffer(raw, dtype=np.uint8)
        triples = raw_bytes.reshape(-1, 3)
        values = (
            triples[:, 0].astype(np.int32)
            | (triples[:, 1].astype(np.int32) << 8)
            | (triples[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        data = values.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"unsupported WAV sample width: {sampwidth}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    if sr != target_sr and data.size:
        duration = data.size / sr
        target_n = int(round(duration * target_sr))
        x_old = np.linspace(0.0, duration, num=data.size, endpoint=False)
        x_new = np.linspace(0.0, duration, num=target_n, endpoint=False)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = target_sr
    return data


def write_wav(path: str | Path, samples, sr: int) -> None:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


def _read_tags(path: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    try:
        from mutagen import File as MutagenFile  # type: ignore

        audio = MutagenFile(str(path), easy=True)
        if audio and audio.tags:
            for key in ("title", "artist", "album"):
                value = audio.tags.get(key)
                if value:
                    tags[key] = str(value[0])
    except Exception:
        pass
    if "title" not in tags:
        tags["title"] = path.stem
    return tags


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
