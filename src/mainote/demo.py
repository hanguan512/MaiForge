"""Synthetic demo: click track -> analysis -> rule chart -> maidata.

Useful for smoke-testing the whole P1/P2 pipeline without copyrighted audio.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .audio.io import write_wav
from .audio.timing import TimingInfo
from .generate import generate_from_audio, maidata_for_chart
from .maidata.compiler import compile_maidata
from .maidata.stats import chart_stats


def make_click_track(
    path: str | Path,
    *,
    bpm: float = 143.0,
    bars: int = 32,
    sr: int = 22050,
) -> TimingInfo:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for the demo") from exc

    seconds_per_beat = 60.0 / bpm
    duration = bars * 4 * seconds_per_beat
    n = int(round(duration * sr)) + sr
    buf = np.zeros(n, dtype=np.float32)

    def add_click(time_seconds: float, freq: float, amp: float, decay: float = 70.0) -> None:
        start = int(round(time_seconds * sr))
        length = int(0.08 * sr)
        if start >= n:
            return
        end = min(n, start + length)
        t = np.arange(end - start, dtype=np.float32) / sr
        env = np.exp(-t * decay).astype(np.float32)
        buf[start:end] += amp * np.sin(2 * np.pi * freq * t).astype(np.float32) * env

    for bar in range(bars):
        bar_start = bar * 4 * seconds_per_beat
        for beat in range(4):
            t = bar_start + beat * seconds_per_beat
            if beat == 0:
                add_click(t, 90.0, 0.65, 45.0)
                add_click(t, 1600.0, 0.25, 120.0)
            else:
                add_click(t, 1200.0, 0.42, 90.0)
            # Weak offbeat hi-hat.
            add_click(t + seconds_per_beat / 2, 3200.0, 0.12, 180.0)
        # A simple bass note at the bar start to make the low band less empty.
        t = bar_start
        length = int(0.35 * sr)
        start = int(t * sr)
        end = min(n, start + length)
        if start < n:
            tt = np.arange(end - start, dtype=np.float32) / sr
            env = np.exp(-tt * 5.0).astype(np.float32)
            buf[start:end] += 0.12 * np.sin(
                2 * np.pi * (55.0 * (1 + (bar % 4) * 0.25)) * tt
            ).astype(np.float32) * env

    peak = float(np.max(np.abs(buf)))
    if peak > 0:
        buf /= peak * 1.05
    write_wav(path, buf, sr)
    return TimingInfo(
        path=str(path),
        sr=sr,
        duration=n / sr,
        bpm=bpm,
        bpm_confidence=1.0,
        beat_times=[],
        downbeat_times=[],
        onset_times=[],
        onset_strengths=[],
        tags={"title": "synthetic click demo", "artist": "mainote"},
    )


def run_demo(out_dir: str | Path = "out/demo") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "demo_143.wav"
    make_click_track(audio_path, bpm=143.0, bars=32)

    timing, result = generate_from_audio(
        str(audio_path),
        difficulty="master",
        level=12.0,
        seed=7,
        title="synthetic click demo",
        artist="mainote",
    )
    chart = result.chart
    maidata = maidata_for_chart(
        chart,
        title="synthetic click demo",
        artist="mainote",
        wholebpm=timing.bpm,
    )
    text = compile_maidata(maidata)
    maidata_path = out_dir / "maidata.txt"
    maidata_path.write_text(text, encoding="utf-8")
    timing.save(out_dir / "timing.json")
    (out_dir / "ir.json").write_text(
        json.dumps(maidata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    stats = chart_stats(chart)
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Also create a tiny raw dataset so `dataset ingest` can be smoke-tested.
    raw_song = out_dir / "raw" / "demo"
    raw_song.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_path, raw_song / "audio.wav")
    shutil.copy2(maidata_path, raw_song / "maidata.txt")

    summary = {
        "audio": str(audio_path),
        "maidata": str(maidata_path),
        "bpm": round(timing.bpm, 3),
        "bpm_confidence": round(timing.bpm_confidence, 3),
        "chart_notes": stats["notes_total"],
        "notes_per_second": stats["notes_per_second"],
        "validation_warnings": result.warnings,
        "duration_seconds": round(timing.duration, 2),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
