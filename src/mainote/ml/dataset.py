"""Windowed chart dataset for P3 training."""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..maidata.model import TICKS_PER_BAR, Chart
from .tokenizer import EventTokenizer


@dataclass
class WindowSpec:
    row: dict[str, Any]
    sample_path: Path
    start_bar: int
    end_bar: int


class ChartWindowDataset(Dataset):
    """One item = a fixed-bar audio/chart window for one difficulty."""

    def __init__(
        self,
        manifest_path: str | Path,
        features_dir: str | Path,
        split: str,
        tokenizer: EventTokenizer,
        *,
        window_bars: int = 16,
        stride_bars: int = 8,
        context_bars: int = 2,
        min_events: int = 1,
        max_windows_per_chart: int | None = None,
        max_audio_frames: int | None = 4096,
        min_alignment: float = 0.0,
        augment: bool = False,
        spec_augment: dict[str, Any] | None = None,
    ):
        if window_bars <= 0 or stride_bars <= 0:
            raise ValueError("window_bars and stride_bars must be positive")
        self.manifest_path = Path(manifest_path)
        self.features_dir = Path(features_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.window_bars = window_bars
        self.stride_bars = stride_bars
        self.context_bars = context_bars
        self.min_events = min_events
        self.max_audio_frames = max_audio_frames
        self.augment = augment
        self.spec_augment = spec_augment or {}
        self._cache: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._rng = random.Random(0)
        self.windows: list[WindowSpec] = []

        rows = self._read_rows(min_alignment=min_alignment)
        for row in rows:
            sample_path = row.get("sample")
            if sample_path and Path(sample_path).exists():
                sample_path = Path(sample_path)
            else:
                sample_path = (
                    self.manifest_path.parent
                    / str(row["song_id"])
                    / f"inote_{int(row['chart_index'])}.json"
                )
            if not sample_path.exists():
                continue
            try:
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
                chart = Chart.from_dict(sample["chart"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            max_tick = max((e.tick for e in chart.events), default=0)
            max_bar = max(1, math.ceil((max_tick + 1) / TICKS_PER_BAR))
            starts = list(range(0, max_bar, stride_bars))
            windows: list[WindowSpec] = []
            for start in starts:
                end = min(start + window_bars, max_bar)
                if end <= start:
                    continue
                start_tick = start * TICKS_PER_BAR
                end_tick = end * TICKS_PER_BAR
                count = sum(1 for e in chart.events if start_tick <= e.tick < end_tick)
                if count >= min_events:
                    windows.append(
                        WindowSpec(
                            row=row,
                            sample_path=sample_path,
                            start_bar=start,
                            end_bar=end,
                        )
                    )
            if max_windows_per_chart and len(windows) > max_windows_per_chart:
                caps = max_windows_per_chart
                step = max(1, len(windows) // caps)
                windows = windows[::step][:caps]
            self.windows.extend(windows)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.windows[index]
        sample = json.loads(spec.sample_path.read_text(encoding="utf-8"))
        chart = Chart.from_dict(sample["chart"])
        row = spec.row
        start_tick = spec.start_bar * TICKS_PER_BAR
        end_tick = spec.end_bar * TICKS_PER_BAR
        difficulty_index = int(row.get("chart_index") or chart.index)
        level = row.get("level_num")
        if level is None:
            level = row.get("level")
        token_ids = self.tokenizer.encode_chart(
            chart,
            start_tick=start_tick,
            end_tick=end_tick,
            difficulty_index=difficulty_index,
            level=level,
        )

        song_id = str(row["song_id"])
        mel, fps = self._load_features(song_id)
        context_start_tick = max(0, start_tick - self.context_bars * TICKS_PER_BAR)
        max_tick = max((e.tick for e in chart.events), default=end_tick)
        context_end_tick = min(
            max_tick + TICKS_PER_BAR,
            end_tick + self.context_bars * TICKS_PER_BAR,
        )
        start_sec = chart.first + chart.seconds_at_tick(context_start_tick)
        end_sec = chart.first + chart.seconds_at_tick(context_end_tick)
        a = max(0, int(math.floor(start_sec * fps)))
        b = min(len(mel), int(math.ceil(end_sec * fps)))
        if b <= a:
            a = min(max(0, a), max(0, len(mel) - 1))
            b = min(len(mel), a + 1)
        window = mel[a:b].copy()
        if self.max_audio_frames and len(window) > self.max_audio_frames:
            start = max(0, (len(window) - self.max_audio_frames) // 2)
            window = window[start : start + self.max_audio_frames]
        if self.augment and self.spec_augment:
            window = self._spec_augment(window)
        return {
            "mel": window.astype(np.float32),
            "tokens": token_ids,
            "sample_id": row.get("sample_id", ""),
            "song_id": song_id,
            "start_bar": spec.start_bar,
            "end_bar": spec.end_bar,
            "fps": fps,
        }

    # ------------------------------------------------------------------
    def _read_rows(self, *, min_alignment: float) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") != self.split:
                continue
            if not row.get("has_audio", False):
                continue
            if not row.get("valid", True):
                continue
            confidence = row.get("alignment_confidence")
            if confidence is not None and float(confidence) < min_alignment:
                continue
            rows.append(row)
        return rows

    def _load_features(self, song_id: str) -> tuple[np.ndarray, float]:
        if song_id in self._cache:
            value = self._cache.pop(song_id)
            self._cache[song_id] = value
            return value
        path = self.features_dir / f"{song_id}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"missing features for {song_id}: {path}. "
                "Run `mainote ml prepare-features <processed_dir> --out <features_dir>`."
            )
        with np.load(path, allow_pickle=False) as data:
            mel = np.asarray(data["mel"], dtype=np.float32)
            fps = float(data["fps"])
        self._cache[song_id] = (mel, fps)
        while len(self._cache) > 8:
            self._cache.popitem(last=False)
        return mel, fps

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        cfg = self.spec_augment
        prob = float(cfg.get("prob", 0.5))
        if self._rng.random() >= prob:
            return mel
        out = mel.copy()
        n_frames, n_mels = out.shape
        freq_mask = int(cfg.get("freq_mask", 8))
        time_mask = int(cfg.get("time_mask", 16))
        if freq_mask > 0 and n_mels > 1:
            width = self._rng.randint(1, min(freq_mask, n_mels - 1))
            start = self._rng.randint(0, n_mels - width)
            out[:, start : start + width] = 0.0
        if time_mask > 0 and n_frames > 1:
            width = self._rng.randint(1, min(time_mask, n_frames - 1))
            start = self._rng.randint(0, n_frames - width)
            out[start : start + width, :] = 0.0
        return out


def collate_windows(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    max_frames = max((item["mel"].shape[0] for item in batch), default=1)
    max_tokens = max((len(item["tokens"]) for item in batch), default=1)
    n_mels = batch[0]["mel"].shape[1]
    mel = torch.zeros((len(batch), max_frames, n_mels), dtype=torch.float32)
    mel_mask = torch.ones((len(batch), max_frames), dtype=torch.bool)
    tokens = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
    token_mask = torch.ones((len(batch), max_tokens), dtype=torch.bool)
    for i, item in enumerate(batch):
        frames = item["mel"].shape[0]
        mel[i, :frames] = torch.from_numpy(item["mel"])
        mel_mask[i, :frames] = False
        length = len(item["tokens"])
        tokens[i, :length] = torch.tensor(item["tokens"], dtype=torch.long)
        token_mask[i, :length] = False
    return {
        "mel": mel,
        "mel_mask": mel_mask,
        "tokens": tokens,
        "token_mask": token_mask,
    }
