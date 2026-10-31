"""Rule-based chart generator (P1 baseline).

This is intentionally a musical *baseline*, not the final model: it uses the
audio's onset/beat grid plus deterministic patterns, and it always emits valid
IR.  It gives the neural model (P3) something concrete to beat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..audio.timing import TimingInfo
from ..maidata.model import (
    TICKS_PER_BAR,
    TICKS_PER_BEAT,
    BpmChange,
    Chart,
    DivisionChange,
    Duration,
    Event,
    SlideSegment,
)
from ..maidata.stats import DIFFICULTY_BY_INDEX, profile_for_difficulty
from ..maidata.validate import validate_chart

DIFFICULTY_RULES: dict[str, dict[str, Any]] = {
    "basic": {
        "grid": 4,
        "max_simultaneous": 2,
        "hold_prob": 0.12,
        "slide_prob": 0.01,
        "touch_prob": 0.0,
        "break_prob": 0.02,
        "ex_prob": 0.02,
        "each_prob": 0.05,
        "min_gap_beats": 1.0,
        "slide_shapes": ["-"],
    },
    "advanced": {
        "grid": 8,
        "max_simultaneous": 2,
        "hold_prob": 0.10,
        "slide_prob": 0.03,
        "touch_prob": 0.04,
        "break_prob": 0.02,
        "ex_prob": 0.03,
        "each_prob": 0.10,
        "min_gap_beats": 0.5,
        "slide_shapes": ["-"],
    },
    "expert": {
        "grid": 8,
        "max_simultaneous": 2,
        "hold_prob": 0.14,
        "slide_prob": 0.06,
        "touch_prob": 0.08,
        "break_prob": 0.04,
        "ex_prob": 0.08,
        "each_prob": 0.14,
        "min_gap_beats": 0.5,
        "slide_shapes": ["-", "<", ">", "q"],
    },
    "master": {
        "grid": 16,
        "max_simultaneous": 3,
        "hold_prob": 0.10,
        "slide_prob": 0.10,
        "touch_prob": 0.14,
        "break_prob": 0.05,
        "ex_prob": 0.14,
        "each_prob": 0.18,
        "min_gap_beats": 0.25,
        "slide_shapes": ["-", "<", ">", "q", "v", "p", "w", "s", "z"],
    },
    "remaster": {
        "grid": 16,
        "max_simultaneous": 3,
        "hold_prob": 0.08,
        "slide_prob": 0.12,
        "touch_prob": 0.18,
        "break_prob": 0.06,
        "ex_prob": 0.18,
        "each_prob": 0.22,
        "min_gap_beats": 0.125,
        "slide_shapes": ["-", "<", ">", "q", "v", "p", "w", "s", "z", "pp", "qq"],
    },
}

DEFAULT_LEVELS = {
    "easy": 1.0,
    "basic": 3.0,
    "advanced": 6.0,
    "expert": 9.0,
    "master": 12.0,
    "remaster": 14.0,
    "original": 12.0,
}

PATTERNS = [
    [1, 5, 2, 6, 3, 7, 4, 8],
    [1, 3, 5, 7, 2, 4, 6, 8],
    [8, 1, 7, 2, 6, 3, 5, 4],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 8, 2, 7, 3, 6, 4, 5],
]

TOUCH_POOL = [
    "A1", "A2", "A5", "A6", "B1", "B2", "B5", "B6",
    "C", "C1", "C2", "D1", "D2", "D5", "D6", "E1", "E2", "E5", "E6",
]


@dataclass
class RuleConfig:
    difficulty: str = "master"
    level: float | None = None
    seed: int = 0
    grid: int | None = None
    target_density: float | None = None
    max_simultaneous: int | None = None
    hold_prob: float | None = None
    slide_prob: float | None = None
    touch_prob: float | None = None
    break_prob: float | None = None
    ex_prob: float | None = None
    each_prob: float | None = None
    min_gap_beats: float | None = None
    designer: str = "mainote-rule"


@dataclass
class RuleResult:
    chart: Chart
    config: RuleConfig
    profile: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def generate_chart(
    timing: TimingInfo,
    *,
    difficulty: str = "master",
    level: float | None = None,
    seed: int = 0,
    config: RuleConfig | None = None,
    title: str | None = None,
    artist: str | None = None,
) -> RuleResult:
    cfg = config or RuleConfig(difficulty=difficulty, level=level, seed=seed)
    cfg.difficulty = difficulty.lower()
    if cfg.level is None:
        cfg.level = level
    name = cfg.difficulty
    if name in ("re:master", "remaster"):
        name = "remaster"
    if cfg.level is None:
        cfg.level = DEFAULT_LEVELS.get(name, 12.0)
    base = dict(DIFFICULTY_RULES.get(name, DIFFICULTY_RULES["master"]))
    for key in (
        "grid",
        "max_simultaneous",
        "hold_prob",
        "slide_prob",
        "touch_prob",
        "break_prob",
        "ex_prob",
        "each_prob",
        "min_gap_beats",
        "target_density",
    ):
        value = getattr(cfg, key, None)
        if value is not None:
            base[key] = value
    profile = profile_for_difficulty(name, cfg.level)
    if cfg.target_density is None:
        base["target_density"] = profile["notes_per_second"]
    grid = int(base["grid"])
    grid_ticks = max(1, TICKS_PER_BAR // grid)
    rng = random.Random(cfg.seed)

    # Each selected tick can produce a primary note plus EACH/touch extras.
    # Select fewer ticks so the final notes/sec lands near the target.
    avg_notes_per_tick = (
        1.0
        + float(base["each_prob"]) * 1.4
        + float(base["touch_prob"]) * 1.15
        + (float(base["touch_prob"]) * 0.25 if base["max_simultaneous"] >= 3 else 0.0)
    )
    tick_density = float(base["target_density"]) / max(1.0, avg_notes_per_tick)

    candidates = _candidate_ticks(timing, grid_ticks, tick_density)
    effective_duration = max(0.0, timing.duration - timing.first)
    selected = _select_ticks(
        candidates,
        target_density=tick_density,
        duration=effective_duration,
        min_gap_ticks=int(round(base["min_gap_beats"] * TICKS_PER_BEAT)),
    )
    if not selected:
        selected = [0]

    events: list[Event] = []
    active_holds: dict[str, int] = {}
    last_pos: str | None = None
    downbeats = set(timing.downbeat_ticks(quantize=grid_ticks))

    for index, tick in enumerate(selected):
        # Expire holds.
        for pos in list(active_holds):
            if active_holds[pos] <= tick:
                del active_holds[pos]

        bar_index = tick // TICKS_PER_BAR
        pattern = PATTERNS[(bar_index // 4 + cfg.seed) % len(PATTERNS)]
        pos = _choose_position(pattern, index, last_pos, active_holds)
        if pos is None:
            continue

        strength = candidates.get(tick, 0.0)
        used = {pos}
        is_downbeat = tick in downbeats or tick % TICKS_PER_BAR == 0
        flags: list[str] = []
        if rng.random() < base["break_prob"] and is_downbeat:
            flags.append("break")
        if rng.random() < base["ex_prob"] and not is_downbeat:
            flags.append("ex")

        primary: Event
        if rng.random() < base["slide_prob"]:
            shape = rng.choice(base["slide_shapes"])
            end = _choose_end(pos, shape, used, rng)
            duration_ticks = 480 if strength > 0.6 else 960
            primary = Event(
                tick=tick,
                kind="slide",
                position=pos,
                flags=flags,
                slide=[
                    SlideSegment(
                        shape=shape,
                        end=end,
                        duration=Duration(ticks=duration_ticks),
                    )
                ],
                chain_type="single",
            )
            # Slides can coexist with a tap on another key.
            if (
                len(used) < base["max_simultaneous"]
                and rng.random() < base["each_prob"] * 0.5
            ):
                partner = _choose_partner(pos, used, active_holds, rng)
                if partner:
                    used.add(partner)
                    events.append(
                        Event(tick=tick, kind="tap", position=partner, flags=[])
                    )
        elif rng.random() < base["hold_prob"]:
            duration_ticks = 1920 if strength > 0.8 else 960
            primary = Event(
                tick=tick,
                kind="hold",
                position=pos,
                flags=flags,
                duration=Duration(ticks=duration_ticks),
            )
            active_holds[pos] = tick + duration_ticks
        else:
            primary = Event(tick=tick, kind="tap", position=pos, flags=flags)
        events.append(primary)

        # EACH on strong accents.
        if (
            len(used) < base["max_simultaneous"]
            and rng.random() < base["each_prob"] * (1.0 + strength)
        ):
            partner = _choose_partner(pos, used, active_holds, rng)
            if partner:
                used.add(partner)
                events.append(
                    Event(tick=tick, kind="tap", position=partner, flags=[])
                )

        # Touch layer.
        if base["touch_prob"] > 0 and rng.random() < base["touch_prob"]:
            n_touch = 1
            if base["max_simultaneous"] >= 3 and rng.random() < 0.25:
                n_touch = 2
            for _ in range(n_touch):
                if len(used) >= base["max_simultaneous"]:
                    break
                sensor = _choose_touch_sensor(used, rng)
                if sensor is None:
                    break
                used.add(sensor)
                touch_flags: list[str] = []
                if rng.random() < 0.12:
                    touch_flags.append("firework")
                if rng.random() < 0.08:
                    duration_ticks = 960
                    events.append(
                        Event(
                            tick=tick,
                            kind="touch_hold",
                            position=sensor,
                            flags=touch_flags,
                            duration=Duration(ticks=duration_ticks),
                        )
                    )
                else:
                    events.append(
                        Event(
                            tick=tick,
                            kind="touch",
                            position=sensor,
                            flags=touch_flags,
                        )
                    )

        last_pos = pos

    events.sort(key=lambda e: e.sort_key())
    events = _repair_events(events, grid_ticks)

    end_tick = _round_up_bar(
        max([e.end_tick() for e in events] + [timing.tick_at(timing.duration) or 0])
    )
    title = title or timing.tags.get("title") or "untitled"
    artist = artist or timing.tags.get("artist") or ""
    chart = Chart(
        index=_difficulty_index(name),
        level="" if cfg.level is None else _fmt_level(cfg.level),
        designer=cfg.designer,
        first=timing.first,
        initial_bpm=timing.bpm,
        events=events,
        bpm_changes=[BpmChange(0, timing.bpm)],
        division_changes=[DivisionChange(0, grid)],
        metadata={
            "lv": "" if cfg.level is None else _fmt_level(cfg.level),
            "des": cfg.designer,
        },
        end_tick=end_tick,
    )
    report = validate_chart(chart)
    warnings = list(report.warnings)
    if report.errors:
        warnings.extend(f"validation error: {e}" for e in report.errors)
        chart.warnings.extend(warnings)
    return RuleResult(
        chart=chart,
        config=cfg,
        profile=profile,
        warnings=warnings,
    )


def _candidate_ticks(
    timing: TimingInfo,
    grid_ticks: int,
    target_density: float,
) -> dict[int, float]:
    """Map tick -> onset strength, with beats/downbeats and a weak grid floor."""
    candidates: dict[int, float] = {}
    for t, s in zip(timing.onset_times, timing.onset_strengths):
        tick = timing.tick_at(t, quantize=grid_ticks)
        if tick is None:
            continue
        candidates[tick] = max(candidates.get(tick, 0.0), float(s))
    for t in timing.beat_times:
        tick = timing.tick_at(t, quantize=grid_ticks)
        if tick is None:
            continue
        candidates[tick] = max(candidates.get(tick, 0.0), 0.55)
    for t in timing.downbeat_times:
        tick = timing.tick_at(t, quantize=grid_ticks)
        if tick is None:
            continue
        candidates[tick] = max(candidates.get(tick, 0.0), 0.75)

    if candidates:
        start = max(0, min(candidates) - grid_ticks * 2)
        end = max(candidates) + grid_ticks * 2
    else:
        start = 0
        end = timing.tick_at(timing.duration, quantize=grid_ticks) or 0
    # A weak floor lets the selector fill the target density even when the
    # onset detector is conservative.  Musical onsets still win the greedy pass.
    for tick in range(start, end + 1, grid_ticks):
        candidates.setdefault(tick, 0.05)
    return candidates


def _select_ticks(
    candidates: dict[int, float],
    *,
    target_density: float,
    duration: float,
    min_gap_ticks: int,
) -> list[int]:
    if not candidates:
        return []
    target_ticks = max(1, int(round(target_density * duration)))
    target_ticks = min(target_ticks, len(candidates))
    ordered = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    selected: list[int] = []
    for tick, _strength in ordered:
        if len(selected) >= target_ticks:
            break
        if all(abs(tick - other) >= min_gap_ticks for other in selected):
            selected.append(tick)
    # If spacing was too strict, relax once.
    if len(selected) < target_ticks * 0.75:
        gap = max(1, min_gap_ticks // 2)
        for tick, _strength in ordered:
            if len(selected) >= target_ticks:
                break
            if tick in selected:
                continue
            if all(abs(tick - other) >= gap for other in selected):
                selected.append(tick)
    selected.sort()
    return selected


def _choose_position(
    pattern: list[int],
    index: int,
    last_pos: str | None,
    active_holds: dict[str, int],
) -> str | None:
    for offset in range(len(pattern)):
        pos = str(pattern[(index + offset) % len(pattern)])
        if pos in active_holds:
            continue
        if pos == last_pos and len(pattern) > 1:
            continue
        return pos
    return None


def _choose_partner(
    pos: str,
    used: set[str],
    active_holds: dict[str, int],
    rng: random.Random,
) -> str | None:
    candidates = [str((int(pos) + delta - 1) % 8 + 1) for delta in (3, 4, 2, 5)]
    rng.shuffle(candidates)
    for candidate in candidates:
        if candidate not in used and candidate not in active_holds:
            return candidate
    return None


def _choose_end(
    pos: str,
    shape: str,
    used: set[str],
    rng: random.Random,
) -> str:
    candidates = [str(i) for i in range(1, 9) if str(i) != pos]
    rng.shuffle(candidates)
    for candidate in candidates:
        if shape == "-" and candidate == pos:
            continue
        return candidate
    return "5" if pos != "5" else "1"


def _choose_touch_sensor(used: set[str], rng: random.Random) -> str | None:
    pool = [s for s in TOUCH_POOL if s not in used]
    if not pool:
        return None
    return rng.choice(pool)


def _repair_events(events: list[Event], grid_ticks: int) -> list[Event]:
    """Remove exact duplicates and enforce non-overlapping holds."""
    result: list[Event] = []
    used_at_tick: dict[int, set[str]] = {}
    active_holds: dict[str, int] = {}
    for event in sorted(events, key=lambda e: e.sort_key()):
        for pos in list(active_holds):
            if active_holds[pos] <= event.tick:
                del active_holds[pos]
        tick_used = used_at_tick.setdefault(event.tick, set())
        if event.position in tick_used:
            continue
        if event.position in active_holds and event.kind != "hold":
            continue
        if event.kind == "hold" and event.duration is not None:
            duration = event.duration.ticks or 0
            if duration <= 0:
                event.duration = Duration(ticks=grid_ticks)
                duration = grid_ticks
            if event.position in active_holds:
                continue
            active_holds[event.position] = event.tick + duration
        if event.kind == "slide":
            for segment in event.slide:
                if segment.shape == "-" and segment.end == event.position:
                    segment.end = "5" if event.position != "5" else "1"
        tick_used.add(event.position)
        result.append(event)
    return result


def _round_up_bar(tick: int) -> int:
    if tick <= 0:
        return TICKS_PER_BAR
    return ((tick + TICKS_PER_BAR - 1) // TICKS_PER_BAR) * TICKS_PER_BAR


def _difficulty_index(name: str) -> int:
    reverse = {v: k for k, v in DIFFICULTY_BY_INDEX.items()}
    return reverse.get(name, 5)


def _fmt_level(level: float) -> str:
    if abs(level - round(level)) < 1e-9:
        return str(int(round(level)))
    return f"{level:.1f}"
