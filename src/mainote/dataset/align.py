"""Audio-chart alignment scoring (P2 data quality)."""

from __future__ import annotations

from typing import Any

from ..audio.timing import TimingInfo
from ..maidata.model import Chart


def score_chart_against_timing(
    timing: TimingInfo,
    chart: Chart,
    *,
    offset_range: float = 0.35,
    step: float = 0.01,
    tolerance: float = 0.06,
) -> dict[str, Any]:
    """Score how well a chart's note times line up with detected onsets.

    Returns the best global offset (seconds), matched ratio, and mean onset
    strength at matched notes.  A low confidence means the audio version or
    ``first`` value is probably wrong; such pairs should not enter training.
    """
    note_times = [chart.event_time_seconds(e) for e in chart.events]
    if not note_times or not timing.onset_times:
        return {
            "score": 0.0,
            "matched_ratio": 0.0,
            "best_offset_seconds": 0.0,
            "confidence": 0.0,
            "note_count": len(note_times),
            "onset_count": len(timing.onset_times),
        }

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for alignment scoring") from exc

    onsets = np.asarray(timing.onset_times, dtype=np.float64)
    strengths = np.asarray(timing.onset_strengths, dtype=np.float64)
    notes = np.asarray(note_times, dtype=np.float64)

    best = {
        "score": -1.0,
        "matched_ratio": 0.0,
        "best_offset_seconds": 0.0,
    }
    offset_scores: list[float] = []
    offsets = np.arange(-offset_range, offset_range + step / 2, step)
    for offset in offsets:
        shifted = notes + offset
        idx = np.searchsorted(onsets, shifted)
        left = np.clip(idx - 1, 0, onsets.size - 1)
        right = np.clip(idx, 0, onsets.size - 1)
        left_delta = np.abs(onsets[left] - shifted)
        right_delta = np.abs(onsets[right] - shifted)
        use_left = left_delta <= right_delta
        nearest = np.where(use_left, left, right)
        delta = np.where(use_left, left_delta, right_delta)
        matched = delta <= tolerance
        ratio = float(matched.mean())
        if not matched.any():
            score = 0.0
        else:
            score = float(strengths[nearest][matched].mean()) * ratio
        offset_scores.append(score)
        better = score > best["score"] + 1e-9
        tied_but_closer = (
            abs(score - best["score"]) <= 1e-9
            and abs(float(offset)) < abs(best["best_offset_seconds"])
        )
        if better or tied_but_closer:
            best = {
                "score": score,
                "matched_ratio": ratio,
                "best_offset_seconds": float(offset),
            }

    # Confidence combines how many notes matched with how strongly the best
    # offset stands out from other offsets (a random chart has little contrast).
    median_score = float(np.median(offset_scores)) if offset_scores else 0.0
    separation = (best["score"] - median_score) / max(1e-9, best["score"])
    separation = max(0.0, min(1.0, separation))
    confidence = best["matched_ratio"] * (0.5 + 0.5 * separation)
    best["confidence"] = round(max(0.0, min(1.0, confidence)), 4)
    best["score"] = round(max(0.0, best["score"]), 4)
    best["matched_ratio"] = round(best["matched_ratio"], 4)
    best["note_count"] = len(notes)
    best["onset_count"] = len(onsets)
    return best
