"""Lightweight onset / tempo analysis (numpy only)."""

from __future__ import annotations

from typing import Any

import math


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "numpy is required for audio analysis. Install it with: "
            "python3 -m pip install numpy"
        ) from exc
    return np


def onset_envelope(
    samples: Any,
    sr: int,
    *,
    n_fft: int = 1024,
    hop: int = 256,
):
    """Return ``(times, strength)`` for a spectral-flux onset envelope."""
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
    if n_frames < 2:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]
    frames = frames[:n_frames]
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    # Log compression makes quiet high-frequency onsets visible without letting
    # a single loud bass note dominate everything.
    spectrum = np.log1p(spectrum * 10.0)
    diff = np.diff(spectrum, axis=0)
    diff = np.maximum(diff, 0.0).sum(axis=1)
    strength = np.concatenate(
        [np.zeros(1, dtype=np.float32), diff.astype(np.float32)]
    )
    peak = float(strength.max())
    if peak > 0:
        strength /= peak
    times = (np.arange(n_frames, dtype=np.float32) * hop + n_fft / 2.0) / sr
    return times, strength


def pick_onsets(
    times,
    strength,
    *,
    threshold_factor: float = 0.6,
    min_gap: float = 0.05,
):
    """Pick local maxima above an adaptive threshold."""
    np = _numpy()
    times = np.asarray(times, dtype=np.float32)
    strength = np.asarray(strength, dtype=np.float32)
    if strength.size < 3:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    mean = float(strength.mean())
    std = float(strength.std())
    threshold = max(mean + threshold_factor * std, 0.03)

    candidates: list[int] = []
    for i in range(1, strength.size - 1):
        if strength[i] < threshold:
            continue
        if strength[i] >= strength[i - 1] and strength[i] > strength[i + 1]:
            candidates.append(i)

    # Greedy strongest-first, then enforce a minimum spacing.
    candidates.sort(key=lambda i: float(strength[i]), reverse=True)
    accepted: list[int] = []
    for i in candidates:
        t = float(times[i])
        if all(abs(t - float(times[j])) >= min_gap for j in accepted):
            accepted.append(i)
    accepted.sort()
    idx = np.array(accepted, dtype=np.int64)
    return times[idx], strength[idx]


def interpolate_strength(times, strength, query_times):
    np = _numpy()
    return np.interp(
        np.asarray(query_times, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
        np.asarray(strength, dtype=np.float64),
        left=0.0,
        right=0.0,
    )


def estimate_bpm(
    times,
    strength,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 200.0,
):
    """Autocorrelation tempo estimate with a broad 130-BPM prior."""
    np = _numpy()
    times = np.asarray(times, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    if times.size < 4 or strength.size < 4:
        return 120.0, 0.0

    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return 120.0, 0.0
    x = np.maximum(strength - float(strength.mean()), 0.0)
    if float(x.sum()) <= 0:
        return 120.0, 0.0

    min_lag = max(1, int(round(60.0 / max_bpm / dt)))
    max_lag = min(int(round(60.0 / min_bpm / dt)), x.size - 2)
    if max_lag <= min_lag:
        return 120.0, 0.0

    best_lag = min_lag
    best_score = -1.0
    scores: dict[int, float] = {}
    for lag in range(min_lag, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
        corr = float(np.dot(a, b) / denom)
        bpm = 60.0 / (lag * dt)
        prior = math.exp(-0.5 * (math.log2(max(bpm, 1e-6) / 130.0) / 0.7) ** 2)
        score = corr * prior
        scores[lag] = corr
        if score > best_score:
            best_score = score
            best_lag = lag

    # Parabolic refinement around the best integer lag.
    if best_lag - 1 in scores and best_lag + 1 in scores:
        left = scores[best_lag - 1]
        center = scores[best_lag]
        right = scores[best_lag + 1]
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-9:
            offset = 0.5 * (left - right) / denom
            offset = max(-0.5, min(0.5, offset))
            best_lag_float = best_lag + offset
        else:
            best_lag_float = float(best_lag)
    else:
        best_lag_float = float(best_lag)

    bpm = 60.0 / (best_lag_float * dt)
    corr_values = list(scores.values())
    mean_corr = float(np.mean(corr_values)) if corr_values else 0.0
    confidence = max(0.0, min(1.0, (scores[best_lag] - mean_corr) / 0.5))
    return float(bpm), confidence


def track_beats(
    times,
    strength,
    bpm: float,
    duration: float,
    *,
    phase_step: float = 0.005,
):
    """Return ``(beat_times, downbeat_times, phase, downbeat_phase)``."""
    np = _numpy()
    if bpm <= 0 or duration <= 0:
        return [], [], 0.0, 0
    period = 60.0 / bpm
    times = np.asarray(times, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    if times.size < 2:
        beats = list(np.arange(0.0, duration + period, period))
        return [float(b) for b in beats], [float(b) for b in beats[::4]], 0.0, 0

    best_phase = 0.0
    best_score = -1.0
    for phase in np.arange(0.0, period, phase_step):
        beat_times = np.arange(phase, duration + period, period)
        if beat_times.size == 0:
            continue
        score = float(
            interpolate_strength(times, strength, beat_times).mean()
        )
        if score > best_score:
            best_score = score
            best_phase = float(phase)

    beats = np.arange(best_phase, duration + period, period)
    beats = beats[(beats >= 0.0) & (beats <= duration + period)]
    beat_list = [float(b) for b in beats]
    if not beat_list:
        beat_list = [0.0]

    # Pick the downbeat phase (0..3) with the strongest average onset support.
    downbeat_phase = 0
    best_db_score = -1.0
    for offset in range(4):
        subset = beat_list[offset::4]
        if not subset:
            continue
        score = float(interpolate_strength(times, strength, subset).mean())
        if score > best_db_score:
            best_db_score = score
            downbeat_phase = offset
    downbeats = beat_list[downbeat_phase::4]
    return beat_list, downbeats, best_phase, downbeat_phase


def analyze_audio(
    path: str,
    sr: int = 22050,
    *,
    bpm: float | None = None,
    first: float = 0.0,
):
    """Analyze an audio file and return a :class:`TimingInfo`.

    ``bpm`` overrides the estimated tempo and re-tracks the beat grid; this is
    the manual-correction path when the frontend lands on a half/double tempo.
    """
    from .io import load_audio
    from .timing import TimingInfo

    audio = load_audio(path, sr=sr)
    times, strength = onset_envelope(audio.samples, audio.sr)
    estimated_bpm, bpm_conf = estimate_bpm(times, strength)
    if bpm is not None:
        if bpm <= 0:
            raise ValueError("bpm override must be positive")
        used_bpm = float(bpm)
    else:
        used_bpm = estimated_bpm
    beats, downbeats, phase, db_phase = track_beats(
        times, strength, used_bpm, audio.duration
    )
    onset_times, onset_strengths = pick_onsets(times, strength)
    return TimingInfo(
        path=audio.path,
        sr=audio.sr,
        duration=audio.duration,
        bpm=used_bpm,
        bpm_confidence=bpm_conf,
        beat_times=beats,
        downbeat_times=downbeats,
        onset_times=[float(x) for x in onset_times],
        onset_strengths=[float(x) for x in onset_strengths],
        first=float(first),
        beat_phase=phase,
        downbeat_phase=db_phase,
        tags=audio.tags,
    )
