"""Chart statistics and difficulty-profile helpers."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .model import EVENT_KINDS, Chart, Event, Maidata, TICKS_PER_BAR

# Rough target ranges (notes per second of chart time).  These are deliberately
# broad; the dataset report should replace them with empirical quantiles.
DIFFICULTY_PROFILES: dict[str, dict[str, Any]] = {
    "easy": {"notes_per_second": (0.4, 1.0), "touch_ratio": (0.0, 0.1)},
    "basic": {"notes_per_second": (0.7, 1.4), "touch_ratio": (0.0, 0.15)},
    "advanced": {"notes_per_second": (1.2, 2.1), "touch_ratio": (0.0, 0.2)},
    "expert": {"notes_per_second": (1.8, 3.0), "touch_ratio": (0.0, 0.25)},
    "master": {"notes_per_second": (2.6, 4.5), "touch_ratio": (0.0, 0.35)},
    "remaster": {"notes_per_second": (3.5, 6.0), "touch_ratio": (0.0, 0.45)},
}

# Piecewise target density anchored on the sample charts and typical maimai
# progression.  This is a prior for the rule baseline; P2/P3 should replace it
# with empirical dataset quantiles.
LEVEL_DENSITY_ANCHORS = [
    (1.0, 0.55),
    (3.0, 1.20),
    (6.0, 1.85),
    (10.0, 2.65),
    (13.0, 3.90),
    (15.0, 5.20),
]


def density_for_level(level: float) -> float:
    level = float(level)
    anchors = LEVEL_DENSITY_ANCHORS
    if level <= anchors[0][0]:
        return anchors[0][1]
    if level >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= level <= x1:
            t = (level - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t
    return anchors[-1][1]

DIFFICULTY_BY_INDEX = {
    1: "easy",
    2: "basic",
    3: "advanced",
    4: "expert",
    5: "master",
    6: "remaster",
    7: "original",
}


def chart_stats(chart: Chart) -> dict[str, Any]:
    counts = Counter(e.kind for e in chart.events)
    flags = Counter(flag for e in chart.events for flag in set(e.flags))
    slide_shapes = Counter(
        segment.shape for e in chart.events for segment in e.slide
    )
    simultaneous = Counter()
    by_tick: dict[int, int] = defaultdict(int)
    for event in chart.events:
        by_tick[event.tick] += 1
    for count in by_tick.values():
        simultaneous[count] += 1

    duration = max(chart.duration_seconds, 1e-9)
    total = len(chart.events)
    slide_events = counts.get("slide", 0)
    slide_segments = sum(len(e.slide) for e in chart.events)
    # ADX counts each additional same-start (`*`) slide segment as a note, but
    # counts a connected path slide (e.g. 4-1>2-7) as one note.  The wiki count
    # also adds each slide's star tap.
    same_start_extra = 0
    for event in chart.events:
        if event.kind != "slide":
            continue
        if event.chain_type == "same_start":
            same_start_extra += len(event.slide) - 1
        elif event.chain_type == "mixed":
            # Each '*' branch is one slide note, even if the branch is a path.
            same_start_extra += len({s.branch for s in event.slide}) - 1
    note_times = sorted(chart.event_time_seconds(e) for e in chart.events)
    end_times = sorted(chart.event_end_seconds(e) for e in chart.events)

    divisions = sorted({d.division for d in chart.division_changes})
    bpms = [b.bpm for b in chart.bpm_changes]

    return {
        "index": chart.index,
        "difficulty": DIFFICULTY_BY_INDEX.get(chart.index, "unknown"),
        "level": chart.level,
        "designer": chart.designer,
        "first": chart.first,
        "initial_bpm": chart.initial_bpm,
        "bpm_min": min(bpms) if bpms else chart.initial_bpm,
        "bpm_max": max(bpms) if bpms else chart.initial_bpm,
        "bpm_change_count": max(0, len(chart.bpm_changes) - 1),
        "duration_seconds": round(duration, 3),
        "chart_end_tick": chart.end_tick,
        "bars": round(chart.end_tick / TICKS_PER_BAR, 3),
        "notes_total": total,
        # ADX counts extra same-start slide segments; the wiki count also
        # counts each slide's star tap.  Keep both for cross-checking.
        "notes_total_adx": total + same_start_extra,
        "notes_total_wiki": total + slide_events + same_start_extra,
        "notes_per_second": round(total / duration, 3),
        "peak_notes_per_second_1s": _peak_density(note_times, 1.0),
        "peak_notes_per_4s": _peak_density(note_times, 4.0),
        "first_note_seconds": round(note_times[0], 3) if note_times else None,
        "last_note_seconds": round(max(end_times), 3) if end_times else None,
        "counts": {kind: counts.get(kind, 0) for kind in EVENT_KINDS},
        "ratios": {
            kind: round(counts.get(kind, 0) / total, 4) if total else 0.0
            for kind in EVENT_KINDS
        },
        "flags": dict(sorted(flags.items())),
        "slide_segments": slide_segments,
        "slide_shapes": dict(sorted(slide_shapes.items())),
        "max_simultaneous": max(by_tick.values(), default=0),
        "simultaneous_histogram": {
            str(k): v for k, v in sorted(simultaneous.items())
        },
        "divisions": divisions,
        "finest_division": max(divisions, default=4),
        "touch_count": counts.get("touch", 0) + counts.get("touch_hold", 0),
        "touch_ratio": round(
            (counts.get("touch", 0) + counts.get("touch_hold", 0)) / total, 4
        )
        if total
        else 0.0,
        "break_notes": flags.get("break", 0),
        "ex_notes": flags.get("ex", 0),
        "firework_notes": flags.get("firework", 0),
    }


def maidata_stats(maidata: Maidata) -> dict[str, Any]:
    charts = [chart_stats(c) for c in sorted(maidata.charts, key=lambda c: c.index)]
    return {
        "title": maidata.get("title", ""),
        "artist": maidata.get("artist", ""),
        "wholebpm": maidata.get("wholebpm"),
        "first": maidata.get("first"),
        "chart_count": len(charts),
        "charts": charts,
    }


def _peak_density(times: list[float], window: float) -> int:
    if not times:
        return 0
    best = 0
    left = 0
    for right, t in enumerate(times):
        while t - times[left] >= window:
            left += 1
        best = max(best, right - left + 1)
    return best


def profile_for_difficulty(
    difficulty: str | None = None,
    level: float | None = None,
) -> dict[str, Any]:
    """Return a target profile for the rule generator.

    ``level`` nudges the density between the profile bounds.  Without a level we
    use the middle of the range.
    """
    name = (difficulty or "master").lower()
    if name in ("re:master", "remaster"):
        name = "remaster"
    profile = DIFFICULTY_PROFILES.get(name, DIFFICULTY_PROFILES["master"])
    low, high = profile["notes_per_second"]
    if level is not None:
        density = max(low, min(high, density_for_level(float(level))))
        fraction = max(0.0, min(1.0, (float(level) - 1.0) / 14.0))
    else:
        density = (low + high) / 2.0
        fraction = 0.5
    return {
        "difficulty": name,
        "level": level,
        "notes_per_second": density,
        "notes_per_second_range": (low, high),
        "touch_ratio": profile.get("touch_ratio", (0.0, 0.3))[1] * fraction,
    }
