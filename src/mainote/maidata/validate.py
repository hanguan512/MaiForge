"""Validation rules for the maidata IR.

These are *hard* format/playability checks, not style rules.  Difficulty is a
soft target and is handled by the evaluator/statistics layer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .model import (
    EVENT_KINDS,
    POSITIONS,
    SLIDE_SHAPES,
    TOUCH_SENSORS,
    Chart,
    Event,
    Maidata,
    TICKS_PER_BEAT,
)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def validate_maidata(maidata: Maidata) -> ValidationReport:
    report = ValidationReport()
    meta = maidata.metadata
    whole_bpm = meta.get("wholebpm")
    if whole_bpm is not None:
        try:
            if float(whole_bpm) <= 0:
                report.errors.append(f"&wholebpm must be positive, got {whole_bpm!r}")
        except ValueError:
            report.errors.append(f"&wholebpm is not a number: {whole_bpm!r}")
    first = meta.get("first")
    if first is not None:
        try:
            float(first)
        except ValueError:
            report.errors.append(f"&first is not a number: {first!r}")

    seen_indices: set[int] = set()
    for chart in maidata.charts:
        if chart.index in seen_indices:
            report.errors.append(f"duplicate chart index inote_{chart.index}")
        seen_indices.add(chart.index)
        sub = validate_chart(chart)
        report.errors.extend(f"inote_{chart.index}: {e}" for e in sub.errors)
        report.warnings.extend(f"inote_{chart.index}: {w}" for w in sub.warnings)
    return report


def validate_chart(chart: Chart) -> ValidationReport:
    report = ValidationReport()
    if chart.index <= 0:
        report.errors.append(f"invalid difficulty index {chart.index}")
    if chart.initial_bpm <= 0:
        report.errors.append(f"initial BPM must be positive, got {chart.initial_bpm}")
    if not chart.bpm_changes:
        report.errors.append("missing BPM map")
    else:
        if chart.bpm_changes[0].tick != 0:
            report.errors.append("BPM map must start at tick 0")
        for change in chart.bpm_changes:
            if change.tick < 0:
                report.errors.append(f"negative BPM change tick {change.tick}")
            if change.bpm <= 0:
                report.errors.append(f"BPM change at tick {change.tick} is not positive")
    for change in chart.division_changes:
        if change.division <= 0:
            report.errors.append(
                f"division change at tick {change.tick} is not positive"
            )

    by_tick: dict[int, list[Event]] = defaultdict(list)
    for event in chart.events:
        by_tick[event.tick].append(event)
        _validate_event(event, report)

    # Same-position collisions at the same tick.
    for tick, events in by_tick.items():
        occupied: dict[str, Event] = {}
        for event in events:
            key = event.position
            if key in occupied:
                report.errors.append(
                    f"tick {tick}: multiple events on position/sensor {key!r}"
                )
            occupied[key] = event

    # Hold / touch-hold overlap and taps inside active holds.
    for kind in ("hold", "touch_hold"):
        intervals: list[tuple[int, int, str]] = []
        for event in chart.events:
            if event.kind != kind or event.duration is None:
                continue
            bpm = chart.bpm_at(event.tick)
            ticks = event.duration.approx_ticks(bpm)
            if ticks is None:
                continue
            if ticks <= 0:
                report.errors.append(
                    f"tick {event.tick}: {kind} on {event.position!r} has non-positive duration"
                )
                continue
            intervals.append((event.tick, event.tick + ticks, event.position))
        intervals.sort()
        last_by_pos: dict[str, tuple[int, int]] = {}
        for start, end, pos in intervals:
            previous = last_by_pos.get(pos)
            if previous and start < previous[1]:
                report.errors.append(
                    f"{kind} overlap on {pos!r}: [{previous[0]}, {previous[1]}) and [{start}, {end})"
                )
            last_by_pos[pos] = (start, end)

        active = list(intervals)
        for event in chart.events:
            if event.kind in ("hold", "touch_hold"):
                continue
            for start, end, pos in active:
                if pos == event.position and start < event.tick < end:
                    report.errors.append(
                        f"tick {event.tick}: event on {pos!r} occurs inside an active {kind}"
                    )
                    break

    # Simultaneous-note sanity.  maimai can show many touch sensors at once;
    # >12 is unusual but not strictly illegal.
    for tick, events in by_tick.items():
        if len(events) > 12:
            report.warnings.append(
                f"tick {tick}: {len(events)} simultaneous events (>12)"
            )

    if chart.end_tick < chart.max_tick:
        report.warnings.append(
            f"end_tick {chart.end_tick} is before the last event tick {chart.max_tick}"
        )
    return report


def _validate_event(event: Event, report: ValidationReport) -> None:
    if event.tick < 0:
        report.errors.append(f"negative event tick {event.tick}")
    if event.kind not in EVENT_KINDS:
        report.errors.append(f"unknown event kind {event.kind!r}")
        return

    if event.kind in ("tap", "hold", "slide"):
        if event.position not in POSITIONS:
            report.errors.append(
                f"tick {event.tick}: invalid outer position {event.position!r}"
            )
    elif event.kind in ("touch", "touch_hold"):
        if event.position not in TOUCH_SENSORS:
            report.errors.append(
                f"tick {event.tick}: invalid touch sensor {event.position!r}"
            )

    _validate_flags(event, report)
    _validate_duration(event.duration, report, f"tick {event.tick} {event.kind}")

    if event.kind in ("hold", "touch_hold"):
        if event.duration is None:
            report.errors.append(f"tick {event.tick}: {event.kind} missing duration")
        elif event.duration.bare:
            report.warnings.append(
                f"tick {event.tick}: bare {event.kind} (special short hold)"
            )
    elif event.duration is not None:
        # Taps/touches should not have durations; parse_touch uses touch_hold
        # for those, so this indicates a malformed IR.
        if event.kind == "tap":
            report.errors.append(f"tick {event.tick}: tap has a duration")

    if event.kind == "slide":
        if not event.slide:
            report.errors.append(f"tick {event.tick}: slide has no segments")
        if event.chain_type not in ("single", "same_start", "path"):
            report.errors.append(
                f"tick {event.tick}: invalid slide chain_type {event.chain_type!r}"
            )
        if event.chain_type == "path":
            _validate_duration(
                event.duration,
                report,
                f"tick {event.tick} path slide",
                required=True,
            )
        previous_end = event.position
        for i, segment in enumerate(event.slide):
            segment_start = (
                event.position
                if event.chain_type == "same_start"
                else previous_end
            )
            if segment.shape not in SLIDE_SHAPES:
                report.errors.append(
                    f"tick {event.tick}: invalid slide shape {segment.shape!r}"
                )
            if segment.end not in POSITIONS:
                report.errors.append(
                    f"tick {event.tick}: invalid slide end {segment.end!r}"
                )
            if segment.shape == "V":
                if segment.via not in POSITIONS:
                    report.errors.append(
                        f"tick {event.tick}: V slide has invalid via {segment.via!r}"
                    )
                if segment.via == segment_start:
                    report.warnings.append(
                        f"tick {event.tick}: V slide via equals start point {segment.via!r}"
                    )
            elif segment.shape == "-" and segment.end == segment_start:
                report.errors.append(
                    f"tick {event.tick}: straight slide starts and ends at {segment.end!r}"
                )
            if event.chain_type != "path":
                _validate_duration(
                    segment.duration,
                    report,
                    f"tick {event.tick} slide segment {i}",
                    required=True,
                )
            elif segment.duration is not None:
                _validate_duration(
                    segment.duration,
                    report,
                    f"tick {event.tick} slide segment {i}",
                )
            previous_end = segment.end


def _validate_flags(event: Event, report: ValidationReport) -> None:
    flags = set(event.flags)
    unknown = flags - {
        "break",
        "ex",
        "firework",
        "no_star_wait",
        "no_star_move",
        "force_star",
        "rotate_star",
        "force_tap",
        "pseudo_each",
    }
    if unknown:
        report.errors.append(f"tick {event.tick}: unknown flags {sorted(unknown)}")
    if "firework" in flags and event.kind not in ("touch", "touch_hold"):
        report.errors.append(
            f"tick {event.tick}: firework flag is only valid on touch events"
        )
    if ("no_star_wait" in flags or "no_star_move" in flags) and event.kind != "slide":
        report.errors.append(
            f"tick {event.tick}: no-star flags are only valid on slides"
        )
    for segment in event.slide:
        segment_flags = set(segment.flags)
        bad = segment_flags - {"break", "ex", "firework", "pseudo_each"}
        if bad:
            report.errors.append(
                f"tick {event.tick}: unknown slide track flags {sorted(bad)}"
            )
        if "firework" in segment_flags:
            report.errors.append(
                f"tick {event.tick}: firework is not valid on a slide track"
            )


def _validate_duration(
    duration,
    report: ValidationReport,
    label: str,
    *,
    required: bool = False,
) -> None:
    if duration is None:
        if required:
            report.warnings.append(f"{label}: missing duration")
        return
    if duration.bare:
        return
    if duration.ticks is not None and duration.ticks <= 0:
        report.errors.append(f"{label}: duration ticks must be positive")
    if duration.seconds is not None and duration.seconds <= 0:
        report.errors.append(f"{label}: duration seconds must be positive")
    if duration.ticks is None and duration.seconds is None:
        report.errors.append(f"{label}: duration has no ticks or seconds")
    if duration.wait_seconds is not None and duration.wait_seconds < 0:
        report.errors.append(f"{label}: wait seconds must be >= 0")
    if duration.wait_bpm is not None and duration.wait_bpm <= 0:
        report.errors.append(f"{label}: wait BPM must be positive")


def count_events(events: Iterable[Event]) -> dict[str, int]:
    counts = {kind: 0 for kind in EVENT_KINDS}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    return counts
