"""Lightweight onset / tempo analysis (numpy only)."""

from __future__ import annotations

from typing import Any

import math


# Audio autocorrelation cannot distinguish a tempo from its metrical relatives
# (95/190, 126.33/190, ...).  Maimai's notated ``&wholebpm`` clusters around
# 165 BPM (median of the local chart corpus), so a log-normal prior over the
# candidate tempo resolves the octave while still letting 60 BPM and 250+ BPM
# outliers win when their autocorrelation peak is strong enough.
BPM_PRIOR_CENTER = 165.0
BPM_PRIOR_SIGMA = 0.6
# Below this margin the estimate is ambiguous enough that callers should
# surface the candidates and suggest a manual ``--bpm`` override.
BPM_LOW_CONFIDENCE = 0.2
_TEMPO_MULTIPLIERS = (1.0, 2.0, 0.5, 1.5, 2.0 / 3.0, 4.0 / 3.0, 3.0 / 4.0)


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


def _autocorrelation(times, strength, min_bpm: float, max_bpm: float):
    np = _numpy()
    times = np.asarray(times, dtype=np.float64)
    strength = np.asarray(strength, dtype=np.float64)
    if times.size < 4 or strength.size < 4:
        return {}, 0.0

    dt = float(np.median(np.diff(times)))
    if dt <= 0:
        return {}, 0.0
    x = np.maximum(strength - float(strength.mean()), 0.0)
    if float(x.sum()) <= 0:
        return {}, 0.0

    min_lag = max(1, int(round(60.0 / max_bpm / dt)))
    max_lag = min(int(round(60.0 / min_bpm / dt)), x.size - 2)
    if max_lag <= min_lag:
        return {}, dt

    scores: dict[int, float] = {}
    for lag in range(min_lag, max_lag + 1):
        a = x[:-lag]
        b = x[lag:]
        denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) + 1e-12
        scores[lag] = float(np.dot(a, b) / denom)
    return scores, dt


def _correlation_at(scores: dict[int, float], lag: float) -> float:
    """Linearly interpolate the autocorrelation at a fractional lag."""
    first = min(scores)
    last = max(scores)
    lag = max(float(first), min(float(last), lag))
    lo = int(math.floor(lag))
    hi = min(lo + 1, last)
    if lo == hi:
        return scores[lo]
    frac = lag - lo
    return scores[lo] * (1.0 - frac) + scores[hi] * frac


def _tempo_peaks(scores: dict[int, float]) -> list[tuple[float, int]]:
    lags = sorted(scores)
    peaks: list[tuple[float, int]] = []
    for index in range(1, len(lags) - 1):
        lag = lags[index]
        if (
            scores[lag] >= scores[lags[index - 1]]
            and scores[lag] >= scores[lags[index + 1]]
        ):
            peaks.append((scores[lag], lag))
    if not peaks:
        peaks = [(scores[lag], lag) for lag in lags]
    peaks.sort(reverse=True)
    return peaks


def _refine_bpm(scores: dict[int, float], dt: float, bpm: float) -> float:
    """Parabolic interpolation around the nearest autocorrelation lag."""
    center = int(round(60.0 / (bpm * dt)))
    if center - 1 not in scores or center + 1 not in scores:
        return bpm
    left = scores[center - 1]
    middle = scores[center]
    right = scores[center + 1]
    denom = left - 2.0 * middle + right
    if abs(denom) <= 1e-9:
        return bpm
    offset = max(-0.5, min(0.5, 0.5 * (left - right) / denom))
    refined = 60.0 / ((center + offset) * dt)
    # A candidate produced by a metrical multiplier may not sit on a peak;
    # only accept the refinement when it stays near the original tempo.
    if abs(refined / bpm - 1.0) > 0.03:
        return bpm
    return refined


def estimate_tempo(
    times,
    strength,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 260.0,
    candidate_limit: int = 5,
) -> tuple[float, float, list[dict[str, float]]]:
    """Estimate BPM, confidence, and ranked metrical candidates.

    Autocorrelation finds the periodicities, but the notated tempo is
    ambiguous between metrical relatives (e.g. 126.33 and 190 BPM describe the
    same onsets).  Each autocorrelation peak is therefore expanded by common
    tempo ratios and scored against the autocorrelation curve plus a
    chart-tempo prior.  Confidence is the relative score margin to the next
    tempo that is at least 8% away; low values mean the caller should ask for
    ``--bpm``.
    """
    scores, dt = _autocorrelation(times, strength, min_bpm, max_bpm)
    if not scores:
        return 120.0, 0.0, []

    expanded: list[tuple[float, float, float]] = []
    for peak_corr, lag in _tempo_peaks(scores)[:12]:
        peak_bpm = 60.0 / (lag * dt)
        for multiplier in _TEMPO_MULTIPLIERS:
            bpm = peak_bpm * multiplier
            if bpm < min_bpm or bpm > max_bpm:
                continue
            lag_f = 60.0 / (bpm * dt)
            if lag_f < min(scores) or lag_f > max(scores):
                continue
            corr = _correlation_at(scores, lag_f)
            prior = math.exp(
                -0.5
                * (
                    math.log2(max(bpm, 1e-6) / BPM_PRIOR_CENTER)
                    / BPM_PRIOR_SIGMA
                )
                ** 2
            )
            expanded.append((corr * prior, corr, bpm))

    # Keep one candidate per tempo cluster; adjacent lags and multiplier
    # aliases otherwise produce misleadingly tiny confidence margins.
    expanded.sort(reverse=True)
    candidates: list[tuple[float, float, float]] = []
    for score, corr, bpm in expanded:
        if all(abs(bpm / existing[2] - 1.0) > 0.06 for existing in candidates):
            candidates.append((score, corr, bpm))
    if not candidates:
        return 120.0, 0.0, []

    best_score, best_corr, best_bpm = candidates[0]
    best_bpm = _refine_bpm(scores, dt, best_bpm)

    second = next(
        (
            candidate
            for candidate in candidates[1:]
            if abs(candidate[2] / best_bpm - 1.0) > 0.08
        ),
        None,
    )
    if best_corr < 0.05:
        confidence = 0.0
    elif second is None or best_score <= 0:
        confidence = 1.0
    else:
        confidence = (best_score - second[0]) / best_score
    confidence = min(confidence, best_corr / 0.15)
    confidence = max(0.0, min(1.0, confidence))

    reported = [
        {
            "bpm": round(float(bpm), 3),
            "score": round(float(score), 4),
            "correlation": round(float(corr), 4),
        }
        for score, corr, bpm in candidates[: max(1, candidate_limit)]
    ]
    if reported:
        reported[0]["bpm"] = round(float(best_bpm), 3)
    return float(best_bpm), float(confidence), reported


def estimate_bpm(
    times,
    strength,
    *,
    min_bpm: float = 60.0,
    max_bpm: float = 260.0,
):
    """Return ``(bpm, confidence)``; see :func:`estimate_tempo`."""
    bpm, confidence, _ = estimate_tempo(
        times, strength, min_bpm=min_bpm, max_bpm=max_bpm
    )
    return bpm, confidence


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
    estimated_bpm, bpm_conf, bpm_candidates = estimate_tempo(times, strength)
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
        bpm_candidates=bpm_candidates,
        beat_times=beats,
        downbeat_times=downbeats,
        onset_times=[float(x) for x in onset_times],
        onset_strengths=[float(x) for x in onset_strengths],
        first=float(first),
        beat_phase=phase,
        downbeat_phase=db_phase,
        tags=audio.tags,
    )
