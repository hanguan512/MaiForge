"""Audio helpers for the embedded MajdataView WebGL preview.

MajdataView's WebGL build always decodes the uploaded track with
``AudioType.MPEG``, so every source has to reach the browser as mp3.  Files
that are already mp3 are served untouched; everything else is transcoded once
and cached under ``out/webui/cache/majdataview``.  Charts without audio get a
silent track so the visual preview still works.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from pathlib import Path

_LOCK = threading.Lock()
MAX_DURATION = 1800.0


def _cache_path(cache_dir: Path, source_key: str, suffix: str) -> Path:
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}{suffix}"


def _run_ffmpeg(arguments: list[str], target: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise ValueError("预览需要 ffmpeg 来生成 mp3 音频")
    cache_dir = target.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp{target.suffix}"
    )
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    command.extend(arguments)
    command.append(str(temporary))
    try:
        process = subprocess.run(command, capture_output=True, timeout=300)
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", "ignore").strip()[:300]
            raise ValueError(f"ffmpeg 转换音频失败: {detail or '未知错误'}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_mp3(
    source: Path, cache_dir: Path, *, start: float = 0.0
) -> Path:
    """Return an mp3 for ``source``, optionally trimmed to start at ``start``.

    Seeking in the WebGL player is done by reloading it with a trimmed track
    and a chart whose ``&first`` was shifted by the same amount, so the
    trimmed variant gets its own cache entry.
    """
    start = max(0.0, float(start))
    if not start and source.suffix.lower() == ".mp3":
        return source
    stat = source.stat()
    key = (
        f"{source.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
        f"|mp3-v2|{start:.3f}"
    )
    target = _cache_path(cache_dir, key, ".mp3")
    with _LOCK:
        if target.is_file() and target.stat().st_size > 0:
            return target
        arguments = []
        if start > 0:
            arguments.extend(["-ss", f"{start:.3f}"])
        arguments.extend(
            [
                "-i",
                str(source),
                "-vn",
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "2",
            ]
        )
        _run_ffmpeg(arguments, target)
    return target


def ensure_silence(duration: float, cache_dir: Path) -> Path:
    """Return a silent mp3 covering ``duration`` seconds (for audio-less charts)."""
    seconds = max(1.0, min(float(duration or 60.0), MAX_DURATION))
    key = f"silence-v1|{seconds:.2f}"
    target = _cache_path(cache_dir, key, ".mp3")
    with _LOCK:
        if target.is_file() and target.stat().st_size > 0:
            return target
        _run_ffmpeg(
            [
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{seconds:.3f}",
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "9",
            ],
            target,
        )
    return target
