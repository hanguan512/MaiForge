"""Note-level evaluation against a reference chart.

These metrics are for comparing a generated chart with an existing maidata
chart.  They are deliberately tolerant in time (default 60 ms) and strict about
position/type, so a model is not rewarded for merely matching the density.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..maidata.model import Chart, Event
from ..maidata.stats import chart_stats


@dataclass
class _Match:
    ref: Event
    pred: Event
    delta_seconds: float


def evaluate_chart(
    pred: Chart,
    ref: Chart,
    *,
    tolerance_seconds: float = 0.06,
) -> dict[str, Any]:
    ref_events = sorted(ref.events, key=lambda e: e.sort_key())
    pred_events = sorted(pred.events, key=lambda e: e.sort_key())
    ref_times = [ref.event_time_seconds(e) for e in ref_events]
    pred_times = [pred.event_time_seconds(e) for e in pred_events]

    matches: list[_Match] = []
    used: set[int] = set()
    for i, ref_event in enumerate(ref_events):
        ref_time = ref_times[i]
        best_j: int | None = None
        best_score: tuple[float, float, float] | None = None
        for j, pred_event in enumerate(pred_events):
            if j in used:
                continue
            dt = abs(pred_times[j] - ref_time)
            if dt > tolerance_seconds:
                continue
            score = (
                0.0 if pred_event.position == ref_event.position else 1.0,
                0.0 if pred_event.kind == ref_event.kind else 1.0,
                dt,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_j = j
        if best_j is not None:
            used.add(best_j)
            matches.append(
                _Match(
                    ref=ref_event,
                    pred=pred_events[best_j],
                    delta_seconds=abs(pred_times[best_j] - ref_time),
                )
            )

    matched = len(matches)
    pred_count = len(pred_events)
    ref_count = len(ref_events)
    precision = matched / pred_count if pred_count else 0.0
    recall = matched / ref_count if ref_count else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    position_ok = sum(1 for m in matches if m.ref.position == m.pred.position)
    type_ok = sum(1 for m in matches if m.ref.kind == m.pred.kind)
    exact_ok = sum(
        1
        for m in matches
        if m.ref.position == m.pred.position
        and m.ref.kind == m.pred.kind
        and _flags(m.ref) == _flags(m.pred)
    )
    duration_errors = []
    for match in matches:
        if match.ref.kind in ("hold", "touch_hold") and match.pred.kind == match.ref.kind:
            ref_dur = match.ref.end_tick() - match.ref.tick
            pred_dur = match.pred.end_tick() - match.pred.tick
            duration_errors.append(abs(ref_dur - pred_dur))

    slide_matches = [
        m for m in matches if m.ref.kind == "slide" and m.pred.kind == "slide"
    ]
    slide_end_ok = sum(
        1
        for m in slide_matches
        if _slide_ends(m.ref) == _slide_ends(m.pred)
    )

    pred_stats = chart_stats(pred)
    ref_stats = chart_stats(ref)
    return {
        "tolerance_seconds": tolerance_seconds,
        "counts": {
            "reference": ref_count,
            "predicted": pred_count,
            "matched": matched,
        },
        "onset": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_time_error_ms": round(
                1000.0 * sum(m.delta_seconds for m in matches) / matched, 3
            )
            if matches
            else 0.0,
        },
        "position_accuracy": round(position_ok / matched, 4) if matched else 0.0,
        "type_accuracy": round(type_ok / matched, 4) if matched else 0.0,
        "exact_accuracy": round(exact_ok / matched, 4) if matched else 0.0,
        "duration_mae_ticks": round(
            sum(duration_errors) / len(duration_errors), 2
        )
        if duration_errors
        else None,
        "slide": {
            "matched": len(slide_matches),
            "endpoint_accuracy": round(slide_end_ok / len(slide_matches), 4)
            if slide_matches
            else None,
        },
        "stats": {
            "notes_per_second": {
                "reference": ref_stats["notes_per_second"],
                "predicted": pred_stats["notes_per_second"],
                "delta": round(
                    pred_stats["notes_per_second"] - ref_stats["notes_per_second"], 4
                ),
            },
            "max_simultaneous": {
                "reference": ref_stats["max_simultaneous"],
                "predicted": pred_stats["max_simultaneous"],
                "delta": pred_stats["max_simultaneous"]
                - ref_stats["max_simultaneous"],
            },
            "type_ratios": {
                kind: {
                    "reference": ref_stats["ratios"].get(kind, 0.0),
                    "predicted": pred_stats["ratios"].get(kind, 0.0),
                }
                for kind in sorted(
                    set(ref_stats["ratios"]) | set(pred_stats["ratios"])
                )
            },
        },
    }


def _flags(event: Event) -> list[str]:
    return sorted(set(event.flags))


def _slide_ends(event: Event) -> list[str]:
    return [segment.end for segment in event.slide]
