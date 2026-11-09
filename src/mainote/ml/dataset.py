"""Windowed chart dataset for P3 training."""

from __future__ import annotations

import json
import math
import random
import zlib
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
        max_tokens: int | None = None,
        min_alignment: float = 0.0,
        augment: bool = False,
        spec_augment: dict[str, Any] | None = None,
        shuffle_same_tick_events: bool = False,
        chart_cache_size: int = 512,
        feature_cache_size: int = 64,
    ):
        if window_bars <= 0 or stride_bars <= 0:
            raise ValueError("window_bars and stride_bars must be positive")
        if chart_cache_size < 1 or feature_cache_size < 1:
            raise ValueError("cache sizes must be positive")
        self.manifest_path = Path(manifest_path)
        self.features_dir = Path(features_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.window_bars = window_bars
        self.stride_bars = stride_bars
        self.context_bars = context_bars
        self.min_events = min_events
        self.max_audio_frames = max_audio_frames
        self.max_tokens = max_tokens
        self.augment = augment
        self.spec_augment = spec_augment or {}
        self.shuffle_same_tick_events = shuffle_same_tick_events
        self.chart_cache_size = chart_cache_size
        self.feature_cache_size = feature_cache_size
        self._feature_cache: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._chart_cache: OrderedDict[Path, Chart] = OrderedDict()
        self._feature_meta_cache: dict[str, tuple[float, int] | None] = {}
        self._rng = random.Random(0)
        self.windows: list[WindowSpec] = []
        # Reasons windows were dropped during construction.  Training logs this
        # once instead of crashing a DataLoader worker mid-epoch.
        self.skipped_windows: dict[str, int] = {}

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
                chart = self._load_chart(sample_path)
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
                if count < min_events:
                    continue
                spec = WindowSpec(
                    row=row,
                    sample_path=sample_path,
                    start_bar=start,
                    end_bar=end,
                )
                # Reject unusable windows during construction.  Doing this
                # lazily in __getitem__ kills a worker halfway through an epoch
                # (overlong mel, missing features, or too many tokens).
                if self._window_is_usable(spec, chart):
                    windows.append(spec)
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
        chart = self.chart_for(index)
        row = spec.row
        token_ids = self._window_token_ids(spec, chart)

        song_id = str(row["song_id"])
        mel, fps = self._load_features(song_id)
        a, b = self._window_frame_bounds(spec, chart, fps, len(mel))
        window = mel[a:b].copy()
        if self.max_audio_frames and len(window) > self.max_audio_frames:
            # Do not silently center-crop: token labels cover the whole
            # window, so cropping the audio would misalign features and
            # targets.  Reject loudly instead.
            raise ValueError(
                f"window {spec.start_bar}-{spec.end_bar} of "
                f"{row.get('song_id', '?')} has {len(window)} mel frames, "
                f"exceeding max_audio_frames={self.max_audio_frames}; "
                "reduce data.window_bars/context_bars or raise "
                "max_audio_frames to match the model config."
            )
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
    def chart_for(self, index: int) -> Chart:
        """Parsed chart for a window (LRU-cached; shared, do not mutate)."""
        return self._load_chart(self.windows[index].sample_path)

    def window_bounds(self, index: int) -> tuple[int, int]:
        spec = self.windows[index]
        return spec.start_bar * TICKS_PER_BAR, spec.end_bar * TICKS_PER_BAR

    def _skip(self, reason: str) -> None:
        self.skipped_windows[reason] = self.skipped_windows.get(reason, 0) + 1

    def _window_token_ids(
        self, spec: WindowSpec, chart: Chart
    ) -> list[int]:
        row = spec.row
        difficulty_index = int(row.get("chart_index") or chart.index)
        level = row.get("level_num")
        if level is None:
            level = row.get("level")
        shuffle_seed = None
        if self.shuffle_same_tick_events:
            # Stable per window: the same window must tokenize identically in
            # every epoch, in validation, and in the usability check, or the
            # cached token counts and val loss stop being comparable.
            sample_id = str(row.get("sample_id") or row.get("song_id", ""))
            shuffle_seed = zlib.crc32(
                f"{sample_id}:{spec.start_bar}:{spec.end_bar}".encode("utf-8")
            )
        return self.tokenizer.encode_chart(
            chart,
            start_tick=spec.start_bar * TICKS_PER_BAR,
            end_tick=spec.end_bar * TICKS_PER_BAR,
            difficulty_index=difficulty_index,
            level=level,
            shuffle_same_tick=self.shuffle_same_tick_events,
            shuffle_seed=shuffle_seed,
        )

    def _window_frame_bounds(
        self,
        spec: WindowSpec,
        chart: Chart,
        fps: float,
        mel_frames: int,
    ) -> tuple[int, int]:
        """Mel slice for a window, including its audio context bars."""
        start_tick = spec.start_bar * TICKS_PER_BAR
        end_tick = spec.end_bar * TICKS_PER_BAR
        context_start_tick = max(0, start_tick - self.context_bars * TICKS_PER_BAR)
        max_tick = max((e.tick for e in chart.events), default=end_tick)
        context_end_tick = min(
            max_tick + TICKS_PER_BAR,
            end_tick + self.context_bars * TICKS_PER_BAR,
        )
        start_sec = chart.first + chart.seconds_at_tick(context_start_tick)
        end_sec = chart.first + chart.seconds_at_tick(context_end_tick)
        a = max(0, int(math.floor(start_sec * fps)))
        b = min(mel_frames, int(math.ceil(end_sec * fps)))
        if b <= a:
            a = min(max(0, a), max(0, mel_frames - 1))
            b = min(mel_frames, a + 1)
        return a, b

    def _feature_meta(self, song_id: str) -> tuple[float, int] | None:
        """(fps, mel frames) without keeping the full mel in the LRU cache."""
        if not song_id:
            return None
        if song_id in self._feature_meta_cache:
            return self._feature_meta_cache[song_id]
        path = self.features_dir / f"{song_id}.npz"
        if not path.exists():
            self._feature_meta_cache[song_id] = None
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                shape = data["mel"].shape
                fps = float(data["fps"])
        except (OSError, ValueError, KeyError, TypeError):
            self._feature_meta_cache[song_id] = None
            return None
        if len(shape) != 2 or fps <= 0:
            self._feature_meta_cache[song_id] = None
            return None
        meta = (fps, int(shape[0]))
        self._feature_meta_cache[song_id] = meta
        return meta

    def _window_is_usable(self, spec: WindowSpec, chart: Chart) -> bool:
        song_id = str(spec.row.get("song_id", ""))
        meta = self._feature_meta(song_id)
        if meta is None:
            self._skip("missing_features")
            return False
        fps, mel_frames = meta
        a, b = self._window_frame_bounds(spec, chart, fps, mel_frames)
        if b <= a:
            self._skip("empty_audio")
            return False
        if self.max_audio_frames and (b - a) > self.max_audio_frames:
            self._skip("overlong_audio")
            return False
        if self.max_tokens:
            try:
                token_ids = self._window_token_ids(spec, chart)
            except (KeyError, TypeError, ValueError):
                self._skip("tokenize_error")
                return False
            # Training feeds tokens[:, :-1]; that prefix must fit the model's
            # positional encoding.  Skip instead of failing on the dense
            # chart that first exceeds max_tokens.
            if len(token_ids) - 1 > self.max_tokens:
                self._skip("overlong_tokens")
                return False
        return True

    def _load_chart(self, path: Path) -> Chart:
        cached = self._chart_cache.pop(path, None)
        if cached is not None:
            self._chart_cache[path] = cached
            return cached
        sample = json.loads(path.read_text(encoding="utf-8"))
        chart = Chart.from_dict(sample["chart"])
        self._chart_cache[path] = chart
        while len(self._chart_cache) > self.chart_cache_size:
            self._chart_cache.popitem(last=False)
        return chart

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
        cached = self._feature_cache.pop(song_id, None)
        if cached is not None:
            self._feature_cache[song_id] = cached
            return cached
        path = self.features_dir / f"{song_id}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"missing features for {song_id}: {path}. "
                "Run `mainote ml prepare-features <processed_dir> --out <features_dir>`."
            )
        with np.load(path, allow_pickle=False) as data:
            mel = np.asarray(data["mel"], dtype=np.float32)
            fps = float(data["fps"])
        self._feature_cache[song_id] = (mel, fps)
        while len(self._feature_cache) > self.feature_cache_size:
            self._feature_cache.popitem(last=False)
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
