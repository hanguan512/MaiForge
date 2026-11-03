"""Core IR for maidata / simai charts.

The IR keeps *semantic* information (tick, kind, position, duration, slide path)
and enough raw duration text to round-trip rare ``[...]`` forms.  Formatting
(commas, ``{n}`` choices, line breaks) is deliberately left to the compiler.

Time base
---------
``TICKS_PER_BEAT = 960`` (like MIDI PPQ), so a 4/4 bar is 3840 ticks.  This
supports every common simai division (``{1}`` ... ``{3840}``) as an integer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

TICKS_PER_BEAT = 960
BEATS_PER_BAR = 4
TICKS_PER_BAR = TICKS_PER_BEAT * BEATS_PER_BAR  # 3840
DEFAULT_BPM = 120.0

POSITIONS = tuple(str(i) for i in range(1, 9))
SLIDE_SHAPES = ("pp", "qq", "-", "^", "<", ">", "v", "p", "q", "s", "z", "w", "V")


def position_distance(a: str, b: str) -> int:
    """Shortest distance around the 8-button ring (0..4)."""
    delta = abs(int(a) - int(b))
    return min(delta, 8 - delta)
TOUCH_SENSORS = tuple(
    [f"A{i}" for i in range(1, 9)]
    + [f"B{i}" for i in range(1, 9)]
    + ["C", "C1", "C2"]
    + [f"D{i}" for i in range(1, 9)]
    + [f"E{i}" for i in range(1, 9)]
)

EVENT_KINDS = ("tap", "hold", "slide", "touch", "touch_hold")
FLAG_NAMES = (
    "break",
    "ex",
    "firework",
    "no_star_wait",   # ?
    "no_star_move",   # !
    "force_star",      # $
    "rotate_star",     # $$
    "force_tap",       # @
    "pseudo_each",     # `
)


class MainoteError(Exception):
    """Base error for the toolkit."""


class ParseError(MainoteError):
    """Raised when maidata cannot be parsed."""


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _sorted_flags(flags: Iterable[str]) -> list[str]:
    order = {name: i for i, name in enumerate(FLAG_NAMES)}
    return sorted(set(flags), key=lambda f: (order.get(f, 999), f))


@dataclass
class Duration:
    """A hold/slide/touch-hold duration.

    Exactly one of ``ticks`` / ``seconds`` / ``bare`` is normally meaningful.
    ``raw`` preserves forms such as ``"4:1"``, ``"#1.5"``, ``"160#2"`` and
    ``"3##1.5"`` so the compiler can reproduce them.
    """

    ticks: int | None = None
    seconds: float | None = None
    raw: str | None = None
    wait_bpm: float | None = None
    wait_seconds: float | None = None
    bare: bool = False

    def signature(self) -> dict[str, Any]:
        # ``raw`` is formatting, not semantics: the compiler may normalize
        # ``[8:2]`` to ``[4:1]`` while the duration is unchanged.
        return {
            "ticks": self.ticks,
            "seconds": _round_float(self.seconds),
            "wait_bpm": _round_float(self.wait_bpm),
            "wait_seconds": _round_float(self.wait_seconds),
            "bare": self.bare,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.signature(), "raw": self.raw}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Duration | None":
        if data is None:
            return None
        return cls(
            ticks=data.get("ticks"),
            seconds=data.get("seconds"),
            raw=data.get("raw"),
            wait_bpm=data.get("wait_bpm"),
            wait_seconds=data.get("wait_seconds"),
            bare=bool(data.get("bare", False)),
        )

    def approx_ticks(self, bpm: float) -> int | None:
        if self.ticks is not None:
            return int(self.ticks)
        if self.seconds is not None and bpm > 0:
            return int(round(self.seconds * bpm / 60.0 * TICKS_PER_BEAT))
        return None

    def approx_seconds(self, bpm: float) -> float | None:
        if self.seconds is not None:
            return float(self.seconds)
        if self.ticks is not None and bpm > 0:
            return self.ticks / TICKS_PER_BEAT * 60.0 / bpm
        return None


@dataclass
class SlideSegment:
    shape: str
    end: str
    via: str | None = None
    duration: Duration | None = None
    flags: list[str] = field(default_factory=list)
    # Same-start branch index.  Always 0 except in a ``mixed`` chain, where a
    # single star branches into several slides and at least one branch is a
    # connected path (e.g. ``2<2[4:17]*<2>2[2:17]``).
    branch: int = 0
    raw: str = ""

    def signature(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "end": self.end,
            "via": self.via,
            "duration": self.duration.signature() if self.duration else None,
            "flags": _sorted_flags(self.flags),
            "branch": self.branch,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.signature(), "raw": self.raw}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SlideSegment":
        return cls(
            shape=data["shape"],
            end=str(data["end"]),
            via=data.get("via"),
            duration=Duration.from_dict(data.get("duration")),
            flags=list(data.get("flags", [])),
            branch=int(data.get("branch", 0)),
            raw=data.get("raw", ""),
        )


@dataclass
class Event:
    tick: int
    kind: str
    position: str
    flags: list[str] = field(default_factory=list)
    duration: Duration | None = None
    slide: list[SlideSegment] = field(default_factory=list)
    # "single": one segment; "same_start": simai `*` chain; "path": connected
    # multi-joint slide such as ``4-1>2-7[16:7]``; "mixed": several `*` branches
    # where at least one branch is itself a connected path.
    chain_type: str = "single"
    raw: str = ""

    def signature(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "kind": self.kind,
            "position": self.position,
            "flags": _sorted_flags(self.flags),
            "duration": self.duration.signature() if self.duration else None,
            "slide": [s.signature() for s in self.slide],
            "chain_type": self.chain_type,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.signature(),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            tick=int(data["tick"]),
            kind=data["kind"],
            position=str(data["position"]),
            flags=list(data.get("flags", [])),
            duration=Duration.from_dict(data.get("duration")),
            slide=[SlideSegment.from_dict(s) for s in data.get("slide", [])],
            chain_type=data.get("chain_type", "single"),
            raw=data.get("raw", ""),
        )

    def sort_key(self) -> tuple:
        return (self.tick, self.kind, self.position, self.raw)

    def _slide_durations(self) -> list[Duration]:
        """Per-branch movement durations for a slide event."""
        if self.chain_type == "path" and self.duration is not None:
            return [self.duration]
        if self.chain_type == "mixed":
            branches: dict[int, Duration] = {}
            for segment in self.slide:
                if segment.duration is not None:
                    branches[segment.branch] = segment.duration
            return list(branches.values())
        return [s.duration for s in self.slide if s.duration is not None]

    def end_tick(self) -> int:
        """Approximate end tick for stats/validation.

        Holds use their written duration.  Slides include simai's implicit
        1-beat star wait plus the longest movement duration.
        """
        if self.kind in ("hold", "touch_hold") and self.duration:
            d = self.duration.ticks
            return self.tick + (d if d is not None else 0)
        if self.kind == "slide":
            durations = [
                d.ticks
                for d in self._slide_durations()
                if d.ticks is not None
            ]
            movement = max(durations) if durations else 0
            return self.tick + TICKS_PER_BEAT + (movement or 0)
        return self.tick

    def duration_seconds_at(self, bpm: float) -> float | None:
        if self.kind in ("hold", "touch_hold") and self.duration:
            return self.duration.approx_seconds(bpm)
        if self.kind == "slide":
            seconds = [
                d.approx_seconds(bpm)
                for d in self._slide_durations()
                if d.approx_seconds(bpm) is not None
            ]
            movement = max(seconds) if seconds else None
            return (60.0 / bpm if bpm > 0 else 0.0) + (movement or 0.0)
        return None


@dataclass
class BpmChange:
    tick: int
    bpm: float

    def signature(self) -> dict[str, Any]:
        return {"tick": int(self.tick), "bpm": _round_float(self.bpm)}

    def to_dict(self) -> dict[str, Any]:
        return self.signature()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BpmChange":
        return cls(tick=int(data["tick"]), bpm=float(data["bpm"]))


@dataclass
class DivisionChange:
    tick: int
    division: int

    def signature(self) -> dict[str, Any]:
        return {"tick": int(self.tick), "division": int(self.division)}

    def to_dict(self) -> dict[str, Any]:
        return self.signature()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DivisionChange":
        return cls(tick=int(data["tick"]), division=int(data["division"]))


@dataclass
class Chart:
    index: int
    level: str = ""
    designer: str = ""
    first: float = 0.0
    initial_bpm: float = DEFAULT_BPM
    events: list[Event] = field(default_factory=list)
    bpm_changes: list[BpmChange] = field(default_factory=list)
    division_changes: list[DivisionChange] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    end_tick: int = 0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.events.sort(key=lambda e: e.sort_key())
        self.bpm_changes.sort(key=lambda b: b.tick)
        self.division_changes.sort(key=lambda d: d.tick)
        if not self.bpm_changes:
            self.bpm_changes = [BpmChange(0, self.initial_bpm)]

    # ---- tempo helpers -------------------------------------------------
    def bpm_at(self, tick: int) -> float:
        bpm = self.initial_bpm
        for change in self.bpm_changes:
            if change.tick <= tick:
                bpm = change.bpm
            else:
                break
        return bpm

    def seconds_at_tick(self, tick: int) -> float:
        """Seconds from chart tick 0 to ``tick`` under the tempo map."""
        if tick <= 0:
            return 0.0
        bpm = self.initial_bpm
        last_tick = 0
        seconds = 0.0
        for change in self.bpm_changes:
            if change.tick <= 0:
                bpm = change.bpm
                continue
            if change.tick >= tick:
                break
            seconds += (change.tick - last_tick) / TICKS_PER_BEAT * 60.0 / bpm
            last_tick = change.tick
            bpm = change.bpm
        seconds += (tick - last_tick) / TICKS_PER_BEAT * 60.0 / bpm
        return seconds

    def ticks_at_seconds(self, seconds: float) -> int:
        """Inverse of :meth:`seconds_at_tick` under the tempo map."""
        if seconds <= 0:
            return 0
        bpm = self.initial_bpm
        last_tick = 0
        elapsed = 0.0
        for change in self.bpm_changes:
            if change.tick <= 0:
                bpm = change.bpm
                continue
            if bpm <= 0:
                return last_tick
            segment = (
                (change.tick - last_tick) / TICKS_PER_BEAT * 60.0 / bpm
            )
            if elapsed + segment >= seconds:
                break
            elapsed += segment
            last_tick = change.tick
            bpm = change.bpm
        if bpm <= 0:
            return last_tick
        return last_tick + int(
            round((seconds - elapsed) * bpm / 60.0 * TICKS_PER_BEAT)
        )

    def event_time_seconds(self, event: Event) -> float:
        return self.first + self.seconds_at_tick(event.tick)

    def event_end_seconds(self, event: Event) -> float:
        """Absolute audio end time (approximate for slides)."""
        start = self.event_time_seconds(event)
        if event.kind == "slide":
            bpm = self.bpm_at(event.tick)
            movement = [
                d.approx_seconds(bpm)
                for d in event._slide_durations()
                if d.approx_seconds(bpm) is not None
            ]
            wait = 60.0 / bpm if bpm > 0 else 0.0
            return start + wait + (max(movement) if movement else 0.0)
        if event.duration is not None:
            d = event.duration.approx_seconds(self.bpm_at(event.tick))
            if d is not None:
                return start + d
        return start

    @property
    def duration_seconds(self) -> float:
        """Chart time-axis duration, excluding ``first``."""
        return self.seconds_at_tick(max(self.end_tick, self.max_tick))

    @property
    def max_tick(self) -> int:
        return max((e.tick for e in self.events), default=0)

    @property
    def note_count(self) -> int:
        return len(self.events)

    def signature(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "level": self.level,
            "designer": self.designer,
            "first": _round_float(self.first),
            "initial_bpm": _round_float(self.initial_bpm),
            "metadata": dict(sorted(self.metadata.items())),
            "bpm_changes": [b.signature() for b in self.bpm_changes],
            "events": [e.signature() for e in sorted(self.events, key=lambda e: e.sort_key())],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.signature(),
            "division_changes": [d.to_dict() for d in self.division_changes],
            "end_tick": self.end_tick,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chart":
        chart = cls(
            index=int(data["index"]),
            level=str(data.get("level", "")),
            designer=str(data.get("designer", "")),
            first=float(data.get("first", 0.0)),
            initial_bpm=float(data.get("initial_bpm", DEFAULT_BPM)),
            events=[Event.from_dict(e) for e in data.get("events", [])],
            bpm_changes=[BpmChange.from_dict(b) for b in data.get("bpm_changes", [])],
            division_changes=[
                DivisionChange.from_dict(d) for d in data.get("division_changes", [])
            ],
            metadata={str(k): str(v) for k, v in data.get("metadata", {}).items()},
            end_tick=int(data.get("end_tick", 0)),
            warnings=list(data.get("warnings", [])),
        )
        return chart


@dataclass
class Maidata:
    metadata_items: list[tuple[str, str]] = field(default_factory=list)
    charts: list[Chart] = field(default_factory=list)
    source_path: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.metadata_items)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.metadata.get(key, default)

    def chart(self, index: int) -> Chart | None:
        for chart in self.charts:
            if chart.index == index:
                return chart
        return None

    def signature(self) -> dict[str, Any]:
        return {
            "metadata": dict(sorted(self.metadata.items())),
            "charts": [c.signature() for c in sorted(self.charts, key=lambda c: c.index)],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "mainote.maidata.ir",
            "version": 1,
            "metadata": [{"key": k, "value": v} for k, v in self.metadata_items],
            "charts": [c.to_dict() for c in sorted(self.charts, key=lambda c: c.index)],
            "source_path": self.source_path,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Maidata":
        items: list[tuple[str, str]] = []
        if "metadata" in data:
            raw = data["metadata"]
            if isinstance(raw, dict):
                items = [(str(k), str(v)) for k, v in raw.items()]
            else:
                items = [(str(item["key"]), str(item["value"])) for item in raw]
        return cls(
            metadata_items=items,
            charts=[Chart.from_dict(c) for c in data.get("charts", [])],
            source_path=data.get("source_path"),
            warnings=list(data.get("warnings", [])),
        )


def ticks_to_duration_fraction(ticks: int) -> tuple[int, int]:
    """Return ``(a, b)`` such that ``ticks = b/a`` whole notes (4/4)."""
    if ticks < 0:
        raise ValueError("duration ticks must be non-negative")
    if ticks == 0:
        # ``[1:0]`` is accepted by simai and means an instant/zero duration.
        return 1, 0
    divisor = math.gcd(int(ticks), TICKS_PER_BAR)
    return TICKS_PER_BAR // divisor, int(ticks) // divisor


def duration_fraction_to_ticks(a: int, b: int) -> int:
    """Convert ``b`` notes of ``1/a`` to ticks in a 4/4 bar."""
    if a <= 0 or b < 0:
        raise ValueError("duration components must be non-negative")
    return int(round(TICKS_PER_BAR * b / a))


def beats_to_ticks(beats: float) -> int:
    return int(round(beats * TICKS_PER_BEAT))


def seconds_to_ticks(seconds: float, bpm: float) -> int:
    return int(round(seconds * bpm / 60.0 * TICKS_PER_BEAT))


def ticks_to_beats(ticks: int) -> float:
    return ticks / TICKS_PER_BEAT


def ticks_to_seconds(ticks: int, bpm: float) -> float:
    return ticks / TICKS_PER_BEAT * 60.0 / bpm
