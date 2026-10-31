"""Log-mel features for the P3 audio encoder (numpy only)."""

from __future__ import annotations

from typing import Any

import math


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "numpy is required for feature extraction. Install with: "
            "python3 -m pip install numpy"
        ) from exc
    return np


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    *,
    fmin: float = 20.0,
    fmax: float | None = None,
):
    np = _numpy()
    fmax = fmax if fmax is not None else sr / 2.0
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = np.array([mel_to_hz(x) for x in mel_points], dtype=np.float64)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        for j in range(left, center):
            if center > left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filters[i, j] = (right - j) / (right - center)
    return filters


def log_mel_spectrogram(
    samples: Any,
    sr: int,
    *,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop: int = 256,
    fmin: float = 20.0,
    fmax: float | None = None,
    normalize: bool = True,
):
    """Return ``(mel, fps)`` with shape ``(frames, n_mels)``."""
    np = _numpy()
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        raise ValueError("empty audio")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    pad = n_fft // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (padded.size - n_fft) // hop
    if n_frames <= 0:
        raise ValueError("audio too short for the chosen FFT/hop")
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    frames = frames[:n_frames]
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    filters = mel_filterbank(sr, n_fft, n_mels, fmin=fmin, fmax=fmax)
    mel = spectrum @ filters.T
    mel = np.log1p(mel).astype(np.float32)
    if normalize:
        mean = float(mel.mean())
        std = float(mel.std())
        if std < 1e-6:
            std = 1.0
        mel = (mel - mean) / std
    return mel.astype(np.float32), sr / float(hop)
