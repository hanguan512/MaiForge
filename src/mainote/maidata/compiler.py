"""Deterministic IR -> maidata compiler.

The compiler chooses ``{n}`` per bar (the coarsest division that aligns every
event and BPM change), emits commas, converts durations back to ``[a:b]``, and
writes the difficulty blocks.  It never guesses musical content.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .model import (
    BEATS_PER_BAR,
    TICKS_PER_BAR,
    Chart,
    Duration,
    Event,
    Maidata,
    SlideSegment,
    ticks_to_duration_fraction,
)

_DIVISORS = [n for n in range(1, TICKS_PER_BAR + 1) if TICKS_PER_BAR % n == 0]
_DIVISORS.sort()


def compile_file(maidata: Maidata) -> str:
    return compile_maidata(maidata)


def compile_maidata(maidata: Maidata, *, trailing_newline: bool = True) -> str:
    lines: list[str] = []
    global_meta = list(maidata.metadata_items)
    global_first = _parse_optional_float(dict(global_meta).get("first"))

    # If there is no global &first but all charts share a non-zero first, add it.
    firsts = {round(c.first, 6) for c in maidata.charts}
    if global_first is None and len(firsts) == 1 and next(iter(firsts)) != 0.0:
        global_meta.append(("first", _fmt_num(next(iter(firsts)))))

    for key, value in global_meta:
        lines.append(f"&{key}={value}")

    for chart in sorted(maidata.charts, key=lambda c: c.index):
        lines.append("")
        lines.extend(_compile_chart_metadata(chart, global_first))
        lines.append(f"&inote_{chart.index}={compile_chart_body(chart)}")

    text = "\n".join(lines).rstrip()
    if trailing_newline:
        text += "\n"
    return text


def _compile_chart_metadata(chart: Chart, global_first: float | None) -> list[str]:
    lines: list[str] = []
    known = {"lv", "des", "first"}
    for key, value in chart.metadata.items():
        if key in known:
            continue
        lines.append(f"&{key}_{chart.index}={value}")

    if chart.level or "lv" in chart.metadata:
        lines.append(f"&lv_{chart.index}={chart.level or '-'}")
    if chart.designer or "des" in chart.metadata:
        lines.append(f"&des_{chart.index}={chart.designer or '-'}")

    if "first" in chart.metadata or (
        global_first is not None and abs(chart.first - global_first) > 1e-9
    ):
        lines.append(f"&first_{chart.index}={_fmt_num(chart.first)}")
    return lines


def compile_chart_body(chart: Chart) -> str:
    if not chart.events:
        bpm = _fmt_num(chart.initial_bpm)
        return f"({bpm}){{1}},\nE"

    max_tick = max(e.tick for e in chart.events)
    last_bar = max_tick // TICKS_PER_BAR
    events_by_tick: dict[int, list[Event]] = defaultdict(list)
    for event in chart.events:
        events_by_tick[event.tick].append(event)
    for bucket in events_by_tick.values():
        bucket.sort(key=lambda e: e.sort_key())

    bpm_by_tick: dict[int, float] = {}
    for change in chart.bpm_changes:
        bpm_by_tick[change.tick] = change.bpm
    bpm_by_tick.setdefault(0, chart.initial_bpm)

    lines: list[str] = []
    last_bpm_emitted: float | None = None
    last_division: int | None = None

    for bar_index in range(last_bar + 1):
        bar_start = bar_index * TICKS_PER_BAR
        bar_end = bar_start + TICKS_PER_BAR

        critical = {tick for tick in events_by_tick if bar_start <= tick < bar_end}
        critical.update(
            tick for tick in bpm_by_tick if bar_start < tick < bar_end
        )
        division = _choose_division(critical, bar_start)
        slot = TICKS_PER_BAR // division

        parts: list[str] = []

        if bar_start == 0:
            bpm = bpm_by_tick.get(0, chart.initial_bpm)
            parts.append(f"({_fmt_num(bpm)})")
            last_bpm_emitted = bpm
        else:
            bpm = bpm_by_tick.get(bar_start)
            if bpm is not None and (
                last_bpm_emitted is None or abs(bpm - last_bpm_emitted) > 1e-9
            ):
                parts.append(f"({_fmt_num(bpm)})")
                last_bpm_emitted = bpm

        if division != last_division or bar_index == 0:
            parts.append(f"{{{division}}}")
            last_division = division

        for k in range(division):
            tick = bar_start + k * slot
            # A BPM change at bar_start is emitted before the {n}; changes
            # inside the bar are emitted before the slot that starts there.
            if tick != bar_start:
                bpm = bpm_by_tick.get(tick)
                if bpm is not None and (
                    last_bpm_emitted is None or abs(bpm - last_bpm_emitted) > 1e-9
                ):
                    parts.append(f"({_fmt_num(bpm)})")
                    last_bpm_emitted = bpm
            tokens = [_format_event(e) for e in events_by_tick.get(tick, [])]
            parts.append("/".join(tokens) + ",")

        lines.append("".join(parts))

    lines.append("E")
    return "\n".join(lines)


def _choose_division(critical_ticks: Iterable[int], bar_start: int) -> int:
    offsets = {tick - bar_start for tick in critical_ticks}
    if not offsets:
        return 1
    for n in _DIVISORS:
        slot = TICKS_PER_BAR // n
        if all(offset % slot == 0 for offset in offsets):
            return n
    return TICKS_PER_BAR  # 1-tick slots; should be unreachable for integer ticks


def _format_event(event: Event) -> str:
    if event.kind == "tap":
        return _format_tap(event)
    if event.kind == "hold":
        return _format_hold(event)
    if event.kind == "slide":
        return _format_slide(event)
    if event.kind in ("touch", "touch_hold"):
        return _format_touch(event)
    raise ValueError(f"unknown event kind {event.kind!r}")


def _flag_suffix(flags: set[str], *, for_hold: bool = False) -> str:
    out = ""
    if "break" in flags:
        out += "b"
    if "ex" in flags:
        out += "x"
    if "force_star" in flags:
        out += "$"
    if "rotate_star" in flags:
        out += "$$"
    if "force_tap" in flags:
        out += "@"
    if "pseudo_each" in flags:
        out += "`"
    return out


def _format_tap(event: Event) -> str:
    return event.position + _flag_suffix(set(event.flags))


def _format_hold(event: Event) -> str:
    flags = set(event.flags)
    s = event.position
    if "break" in flags and "ex" in flags:
        s += "bxh"
    elif "break" in flags:
        s += "bh"
    elif "ex" in flags:
        s += "hx"
    else:
        s += "h"
    # Non b/x flags are rare on holds; emit them after h.
    other = _flag_suffix(flags - {"break", "ex"})
    s += other
    s += _format_duration(event.duration)
    return s


def _format_slide(event: Event) -> str:
    flags = set(event.flags)
    s = event.position
    if "break" in flags:
        s += "b"
    if "ex" in flags:
        s += "x"
    if "no_star_wait" in flags:
        s += "?"
    if "no_star_move" in flags:
        s += "!"
    if "force_tap" in flags:
        s += "@"
    if "force_star" in flags:
        s += "$"
    if "rotate_star" in flags:
        s += "$$"

    for i, segment in enumerate(event.slide):
        if i > 0 and event.chain_type != "path":
            s += "*"
        s += _format_slide_shape(segment)
        s += _format_duration(segment.duration)
        s += _flag_suffix(set(segment.flags))
    if event.duration is not None:
        s += _format_duration(event.duration)
    return s


def _format_slide_shape(segment: SlideSegment) -> str:
    if segment.shape == "V":
        if not segment.via:
            raise ValueError("V slide missing via point")
        return f"V{segment.via}{segment.end}"
    return f"{segment.shape}{segment.end}"


def _format_touch(event: Event) -> str:
    flags = set(event.flags)
    s = event.position
    if "break" in flags:
        s += "b"
    if "ex" in flags:
        s += "x"
    if event.kind == "touch_hold":
        s += "h"
    if "firework" in flags:
        s += "f"
    if "force_star" in flags:
        s += "$"
    if "force_tap" in flags:
        s += "@"
    if "pseudo_each" in flags:
        s += "`"
    s += _format_duration(event.duration)
    return s


def _format_duration(duration: Duration | None) -> str:
    if duration is None:
        return ""
    if duration.raw:
        return f"[{duration.raw}]"
    if duration.bare:
        return ""
    if duration.ticks is not None:
        a, b = ticks_to_duration_fraction(duration.ticks)
        return f"[{a}:{b}]"
    if duration.seconds is not None:
        if duration.wait_seconds is not None:
            return f"[{_fmt_num(duration.wait_seconds)}##{_fmt_num(duration.seconds)}]"
        if duration.wait_bpm is not None:
            return f"[{_fmt_num(duration.wait_bpm)}#{_fmt_num(duration.seconds)}]"
        return f"[#{_fmt_num(duration.seconds)}]"
    return ""


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return "0"
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
