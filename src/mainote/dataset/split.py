"""Leakage-safe dataset splits by song."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def split_dataset(
    processed_dir: str | Path,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
) -> dict[str, Any]:
    if train < 0 or val < 0 or test < 0:
        raise ValueError("split ratios must be non-negative")
    total = train + val + test
    if total <= 0:
        raise ValueError("split ratios must sum to a positive value")
    train, val, test = train / total, val / total, test / total

    processed_dir = Path(processed_dir)
    songs: list[dict[str, str]] = []
    for meta_path in sorted(processed_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        song_id = str(meta.get("song_id") or meta_path.parent.name)
        artist = str(meta.get("artist") or "")
        songs.append({"song_id": song_id, "artist": artist})

    # Stable hash-based ordering: deterministic across machines and independent
    # of filesystem order.  Splitting by song prevents window-level leakage.
    def bucket(song_id: str) -> str:
        digest = hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).digest()
        return digest.hex()

    songs.sort(key=lambda s: (bucket(s["song_id"]), s["song_id"]))
    n = len(songs)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # Guarantee at least one val/test sample when the dataset is large enough.
    if n >= 3:
        if val > 0 and n_val == 0:
            n_val = 1
        if test > 0 and n - n_train - n_val <= 0:
            n_val = max(0, n_val - 1)
    n_test = max(0, n - n_train - n_val)

    splits = {
        "train": songs[:n_train],
        "val": songs[n_train : n_train + n_val],
        "test": songs[n_train + n_val : n_train + n_val + n_test],
    }
    return {
        "processed_dir": str(processed_dir),
        "seed": seed,
        "ratios": {"train": train, "val": val, "test": test},
        "counts": {name: len(items) for name, items in splits.items()},
        "splits": {
            name: [item["song_id"] for item in items] for name, items in splits.items()
        },
        "artists": {
            name: sorted({item["artist"] for item in items if item["artist"]})
            for name, items in splits.items()
        },
    }
