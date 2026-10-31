"""Aggregate processed dataset metadata into a difficulty report."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..maidata.stats import DIFFICULTY_BY_INDEX


def report_dataset(processed_dir: str | Path) -> dict[str, Any]:
    processed_dir = Path(processed_dir)
    metas: list[dict[str, Any]] = []
    for path in sorted(processed_dir.glob("*/meta.json")):
        try:
            metas.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue

    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    alignments: list[dict[str, Any]] = []
    audio_songs = 0
    for meta in metas:
        if meta.get("has_audio"):
            audio_songs += 1
        for chart in meta.get("charts", []):
            name = chart.get("difficulty") or DIFFICULTY_BY_INDEX.get(
                chart.get("index"), "unknown"
            )
            by_difficulty[name].append(chart)
            if chart.get("alignment"):
                alignments.append(
                    {
                        "song_id": meta.get("song_id"),
                        "index": chart.get("index"),
                        **chart["alignment"],
                    }
                )

    difficulty_reports = {
        name: _aggregate_charts(charts)
        for name, charts in sorted(
            by_difficulty.items(),
            key=lambda kv: min(
                (c.get("index", 99) for c in kv[1]), default=99
            ),
        )
    }
    low_alignment = [a for a in alignments if a.get("confidence", 0) < 0.5]
    return {
        "processed_dir": str(processed_dir),
        "song_count": len(metas),
        "audio_song_count": audio_songs,
        "audio_coverage": round(audio_songs / max(1, len(metas)), 3),
        "chart_count": sum(len(m.get("charts", [])) for m in metas),
        "song_ids": sorted(m.get("song_id", "") for m in metas),
        "difficulties": difficulty_reports,
        "alignment": {
            "measured_charts": len(alignments),
            "mean_confidence": round(
                statistics.fmean(a.get("confidence", 0.0) for a in alignments), 4
            )
            if alignments
            else 0.0,
            "low_confidence_charts": len(low_alignment),
            "low_confidence_samples": low_alignment[:20],
        },
    }


def _aggregate_charts(charts: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        out = []
        for chart in charts:
            value = chart.get(key)
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    levels = []
    for chart in charts:
        try:
            levels.append(float(chart.get("level")))
        except (TypeError, ValueError):
            pass

    counts = defaultdict(int)
    for chart in charts:
        for kind, value in chart.get("counts", {}).items():
            counts[kind] += int(value)

    total_notes = sum(int(c.get("notes_total", 0)) for c in charts)
    return {
        "chart_count": len(charts),
        "levels": _summary(levels),
        "notes_total": _summary(values("notes_total")),
        "notes_per_second": _summary(values("notes_per_second")),
        "peak_notes_per_second_1s": _summary(values("peak_notes_per_second_1s")),
        "touch_ratio": _summary(values("touch_ratio")),
        "max_simultaneous": _summary(values("max_simultaneous")),
        "finest_division": _summary(values("finest_division")),
        "bpm_min": _summary(values("bpm_min")),
        "bpm_max": _summary(values("bpm_max")),
        "type_counts": dict(sorted(counts.items())),
        "type_ratios": {
            kind: round(value / max(1, total_notes), 4)
            for kind, value in sorted(counts.items())
        },
    }


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    values = sorted(values)
    return {
        "min": round(values[0], 4),
        "p25": round(_quantile(values, 0.25), 4),
        "median": round(_quantile(values, 0.5), 4),
        "p75": round(_quantile(values, 0.75), 4),
        "max": round(values[-1], 4),
        "mean": round(statistics.fmean(values), 4),
    }


def _quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    fraction = pos - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction
