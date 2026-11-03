"""Precompute log-mel features for training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from ..audio.io import load_audio
from .features import log_mel_spectrogram


def prepare_features(
    processed_dir: str | Path,
    out_dir: str | Path,
    *,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop: int = 256,
    overwrite: bool = False,
    limit: int | None = None,
    songs: Iterable[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = sorted(processed_dir.glob("*/meta.json"))
    wanted = {str(song) for song in songs} if songs is not None else None
    if limit is not None:
        metas = metas[:limit]

    written = 0
    skipped = 0
    errors: list[str] = []
    for meta_path in metas:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{meta_path}: {exc}")
            continue
        song_id = str(meta.get("song_id") or meta_path.parent.name)
        if wanted is not None and song_id not in wanted:
            skipped += 1
            continue
        audio_path = meta.get("audio")
        if not audio_path or not Path(audio_path).exists():
            if log:
                log(f"[{song_id}] skip: no audio")
            skipped += 1
            continue
        target = out_dir / f"{song_id}.npz"
        if target.exists() and not overwrite:
            if log:
                log(f"[{song_id}] exists, skip")
            skipped += 1
            continue
        try:
            audio = load_audio(audio_path, sr=sr)
            mel, fps = log_mel_spectrogram(
                audio.samples,
                audio.sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop=hop,
            )
            import numpy as np

            np.savez_compressed(
                target,
                mel=mel,
                fps=np.float32(fps),
                sr=np.int32(audio.sr),
                n_mels=np.int32(n_mels),
                n_fft=np.int32(n_fft),
                hop=np.int32(hop),
                duration=np.float32(audio.duration),
                song_id=np.array(song_id),
            )
            written += 1
            if log:
                log(f"[{song_id}] {mel.shape} -> {target}")
        except Exception as exc:  # noqa: BLE001 - batch boundary
            errors.append(f"{song_id}: {exc}")
            if log:
                log(f"[{song_id}] ERROR: {exc}")

    summary = {
        "processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "sr": sr,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop": hop,
        "songs": sorted(wanted) if wanted is not None else None,
    }
    (out_dir / "features_index.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
