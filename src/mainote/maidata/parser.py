"""maidata / simai parser.

The parser is deliberately independent from audio and ML dependencies.  It
produces the semantic IR in :mod:`mainote.maidata.model`.
"""

from __future__ import annotations

import re
from pathlib import Path

from .model import (
    DEFAULT_BPM,
    TICKS_PER_BAR,
    BpmChange,
    Chart,
    DivisionChange,
    Duration,
    Event,
    Maidata,
    ParseError,
    SlideSegment,
    duration_fraction_to_ticks,
)
from .model import SLIDE_SHAPES, TOUCH_SENSORS

_TOUCH_RE = re.compile(r"^(A[1-8]|B[1-8]|C[12]?|D[1-8]|E[1-8])")
_SLIDE_FLAG_CHARS = set("bx?!@$`")
_TRACK_FLAG_CHARS = set("bx?!@$`")
_HOLD_FLAG_CHARS = set("bx$@`")
_TAP_FLAG_CHARS = set("bx$@`")


def parse_file(path: str | Path) -> Maidata:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return parse_maidata_text(text, source_path=str(path))


def parse_maidata_text(text: str, source_path: str | None = None) -> Maidata:
    lines = text.splitlines()
    global_items: list[tuple[str, str]] = []
    raw_charts: dict[int, str] = {}
    warnings: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("&"):
            i += 1
            continue
        body = line[1:]
        key, sep, value = body.partition("=")
        key = key.strip()
        value = value.strip() if sep else ""
        if key.startswith("inote_"):
            suffix = key[len("inote_") :].strip()
            try:
                index = int(suffix)
            except ValueError as exc:
                raise ParseError(f"invalid difficulty key {key!r}") from exc
            body_lines = [value]
            i += 1
            while i < len(lines) and not lines[i].startswith("&"):
                body_lines.append(lines[i])
                i += 1
            if index in raw_charts:
                warnings.append(f"duplicate &inote_{index}; using the last block")
            raw_charts[index] = "\n".join(body_lines)
            continue

        global_items.append((key, value))
        i += 1

    # Split &lv_N / &des_N / &first_N / any other &foo_N keys into per-chart
    # metadata, so the compiler can emit them in the difficulty block.
    chart_indices = set(raw_charts)
    global_meta: list[tuple[str, str]] = []
    per_chart: dict[int, dict[str, str]] = {n: {} for n in chart_indices}
    for key, value in global_items:
        prefix, sep, suffix = key.rpartition("_")
        if sep and suffix.isdigit() and int(suffix) in chart_indices:
            per_chart[int(suffix)][prefix] = value
        else:
            global_meta.append((key, value))

    meta = dict(global_meta)
    try:
        whole_bpm = float(meta.get("wholebpm", DEFAULT_BPM))
    except ValueError:
        warnings.append(f"invalid &wholebpm={meta.get('wholebpm')!r}; using {DEFAULT_BPM}")
        whole_bpm = DEFAULT_BPM
    global_first = _parse_float(meta.get("first", "0"), 0.0, warnings, "&first")

    charts: list[Chart] = []
    for index in sorted(raw_charts):
        chart_meta = per_chart[index]
        level = chart_meta.get("lv", "")
        designer = chart_meta.get("des", "")
        first = _parse_float(chart_meta.get("first", ""), global_first, warnings, f"&first_{index}")
        chart = _parse_chart_body(
            raw_charts[index],
            index=index,
            level=level,
            designer=designer,
            first=first,
            initial_bpm=whole_bpm,
            metadata=chart_meta,
        )
        charts.append(chart)
        warnings.extend(f"inote_{index}: {w}" for w in chart.warnings)

    return Maidata(
        metadata_items=global_meta,
        charts=charts,
        source_path=source_path,
        warnings=warnings,
    )


def _parse_float(value: str, default: float, warnings: list[str], label: str) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        warnings.append(f"invalid {label}={value!r}; using {default}")
        return default


def _parse_chart_body(
    body: str,
    *,
    index: int,
    level: str,
    designer: str,
    first: float,
    initial_bpm: float,
    metadata: dict[str, str],
) -> Chart:
    # Strip line comments.  ``||`` is the current form; ``>>`` is legacy.
    cleaned_lines = []
    for line in body.splitlines():
        cut = len(line)
        for marker in ("||", ">>"):
            pos = line.find(marker)
            if pos >= 0:
                cut = min(cut, pos)
        cleaned_lines.append(line[:cut])
    text = "\n".join(cleaned_lines)

    tick = 0
    division = 4
    bpm = initial_bpm
    slot: list[str] = []
    events: list[Event] = []
    bpm_changes: list[BpmChange] = []
    division_changes: list[DivisionChange] = []
    warnings: list[str] = []
    ended = False
    final_slot = False

    def slot_ticks(div: int) -> int:
        if div <= 0:
            raise ParseError(f"inote_{index}: invalid division {{{div}}}")
        if TICKS_PER_BAR % div == 0:
            return TICKS_PER_BAR // div
        value = round(TICKS_PER_BAR / div)
        warnings.append(
            f"division {{{div}}} does not divide 4/4 bar; rounded slot to {value} ticks"
        )
        return value

    def record_bpm(tick_value: int, bpm_value: float) -> None:
        if bpm_value <= 0:
            raise ParseError(f"inote_{index}: invalid BPM {bpm_value}")
        bpm_changes.append(BpmChange(tick_value, bpm_value))

    def record_division(tick_value: int, div_value: int) -> None:
        if div_value <= 0:
            raise ParseError(f"inote_{index}: invalid division {{{div_value}}}")
        division_changes.append(DivisionChange(tick_value, div_value))

    def flush_slot() -> None:
        nonlocal slot, ended
        text_slot = "".join(slot).strip()
        slot = []
        if not text_slot:
            return
        if text_slot == "E":
            ended = True
            return
        for part in text_slot.split("/"):
            part = part.strip()
            if not part:
                continue
            if part == "E":
                ended = True
                break
            parsed = _parse_token(part, tick, bpm)
            if parsed is None:
                raise ParseError(
                    f"inote_{index}: unknown note token {part!r} at tick {tick}"
                )
            events.extend(parsed)
            if ended:
                break

    i = 0
    while i < len(text) and not ended:
        c = text[i]
        if c == "(":
            j = text.find(")", i + 1)
            if j < 0:
                raise ParseError(f"inote_{index}: unterminated (bpm) at offset {i}")
            raw = text[i + 1 : j].strip()
            try:
                bpm = float(raw)
            except ValueError as exc:
                raise ParseError(f"inote_{index}: invalid BPM ({raw})") from exc
            record_bpm(tick, bpm)
            i = j + 1
            continue
        if c == "{":
            j = text.find("}", i + 1)
            if j < 0:
                raise ParseError(f"inote_{index}: unterminated {{n}} at offset {i}")
            raw = text[i + 1 : j].strip()
            try:
                division = int(raw)
            except ValueError as exc:
                raise ParseError(f"inote_{index}: invalid division {{{raw}}}") from exc
            record_division(tick, division)
            i = j + 1
            continue
        if c == ",":
            flush_slot()
            tick += slot_ticks(division)
            i += 1
            continue
        if c.isspace():
            i += 1
            continue
        slot.append(c)
        i += 1

    if not ended and slot:
        flush_slot()
        final_slot = True

    # Deduplicate BPM/division changes at the same tick (last wins).
    bpm_changes.sort(key=lambda b: b.tick)
    dedup_bpm: list[BpmChange] = []
    for change in bpm_changes:
        if dedup_bpm and dedup_bpm[-1].tick == change.tick:
            dedup_bpm[-1] = change
        else:
            dedup_bpm.append(change)
    if not dedup_bpm or dedup_bpm[0].tick != 0:
        dedup_bpm.insert(0, BpmChange(0, initial_bpm))
    effective_initial_bpm = dedup_bpm[0].bpm

    division_changes.sort(key=lambda d: d.tick)
    dedup_div: list[DivisionChange] = []
    for change in division_changes:
        if dedup_div and dedup_div[-1].tick == change.tick:
            dedup_div[-1] = change
        else:
            dedup_div.append(change)
    if not dedup_div or dedup_div[0].tick != 0:
        dedup_div.insert(0, DivisionChange(0, 4))

    end_tick = tick + (slot_ticks(division) if final_slot else 0)

    return Chart(
        index=index,
        level=level,
        designer=designer,
        first=first,
        initial_bpm=effective_initial_bpm,
        events=events,
        bpm_changes=dedup_bpm,
        division_changes=dedup_div,
        metadata=dict(metadata),
        end_tick=end_tick,
        warnings=warnings,
    )


def _parse_token(token: str, tick: int, bpm: float) -> list[Event] | None:
    if not token:
        return None

    touch = _parse_touch(token, tick)
    if touch is not None:
        return [touch]

    slide = _parse_slide(token, tick)
    if slide is not None:
        return [slide]

    hold = _parse_hold(token, tick)
    if hold is not None:
        return [hold]

    tap = _parse_tap(token, tick)
    if tap is not None:
        return tap

    return None


def _parse_bracket(token: str, pos: int) -> tuple[str, int]:
    """Read ``[...]`` starting at ``pos``; return (content, next_pos)."""
    if pos >= len(token) or token[pos] != "[":
        raise ParseError(f"expected '[' at {pos} in {token!r}")
    end = token.find("]", pos + 1)
    if end < 0:
        raise ParseError(f"unterminated duration in {token!r}")
    return token[pos + 1 : end], end + 1


def _parse_duration(raw: str) -> Duration:
    raw = raw.strip()
    if not raw:
        raise ParseError("empty duration []")
    if "#" in raw:
        parts = raw.split("#")
        seconds: float | None = None
        wait_bpm: float | None = None
        wait_seconds: float | None = None
        try:
            if len(parts) == 2:
                # "#1.5" or "160#2"
                if parts[0].strip():
                    wait_bpm = float(parts[0])
                if parts[1].strip():
                    seconds = float(parts[1])
            elif len(parts) == 3:
                # "3##1.5" (wait seconds, then move seconds)
                if parts[0].strip():
                    wait_seconds = float(parts[0])
                if parts[2].strip():
                    seconds = float(parts[2])
            else:
                raise ValueError
        except ValueError as exc:
            raise ParseError(f"invalid duration [{raw}]") from exc
        return Duration(
            seconds=seconds,
            raw=raw,
            wait_bpm=wait_bpm,
            wait_seconds=wait_seconds,
        )
    if ":" in raw:
        a_text, sep, b_text = raw.partition(":")
        if not sep or not a_text.isdigit() or not b_text.isdigit():
            raise ParseError(f"invalid duration [{raw}]")
        try:
            ticks = duration_fraction_to_ticks(int(a_text), int(b_text))
        except ValueError as exc:
            raise ParseError(f"invalid duration [{raw}]") from exc
        return Duration(ticks=ticks, raw=raw)
    raise ParseError(f"invalid duration [{raw}]")


def _add_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _parse_touch(token: str, tick: int) -> Event | None:
    match = _TOUCH_RE.match(token)
    if not match:
        return None
    sensor = match.group(1)
    if sensor == "E" and len(token) == 1:
        return None
    rest = token[match.end() :]
    flags: list[str] = []
    is_hold = False
    duration: Duration | None = None
    seen_bracket = False
    i = 0
    while i < len(rest):
        c = rest[i]
        if c == "b":
            _add_flag(flags, "break")
            i += 1
        elif c == "x":
            _add_flag(flags, "ex")
            i += 1
        elif c == "f":
            _add_flag(flags, "firework")
            i += 1
        elif c == "$":
            _add_flag(flags, "force_star")
            i += 1
        elif c == "@":
            _add_flag(flags, "force_tap")
            i += 1
        elif c == "`":
            _add_flag(flags, "pseudo_each")
            i += 1
        elif c == "h":
            is_hold = True
            i += 1
        elif c == "[":
            if seen_bracket:
                raise ParseError(f"duplicate duration in touch {token!r}")
            raw, i = _parse_bracket(rest, i)
            duration = _parse_duration(raw)
            seen_bracket = True
        else:
            raise ParseError(f"invalid touch token {token!r}")
    if seen_bracket and not is_hold:
        raise ParseError(f"touch token {token!r} has a duration but no 'h'")
    if is_hold and duration is None:
        duration = Duration(bare=True)
    return Event(
        tick=tick,
        kind="touch_hold" if is_hold else "touch",
        position=sensor,
        flags=flags,
        duration=duration,
        raw=token,
    )


def _parse_hold(token: str, tick: int) -> Event | None:
    if not token or token[0] not in "12345678":
        return None
    position = token[0]
    rest = token[1:]
    flags: list[str] = []
    i = 0
    while i < len(rest) and rest[i] in _HOLD_FLAG_CHARS:
        if rest[i] == "$":
            if i + 1 < len(rest) and rest[i + 1] == "$":
                _add_flag(flags, "rotate_star")
                i += 2
                continue
            _add_flag(flags, "force_star")
        elif rest[i] == "b":
            _add_flag(flags, "break")
        elif rest[i] == "x":
            _add_flag(flags, "ex")
        elif rest[i] == "@":
            _add_flag(flags, "force_tap")
        elif rest[i] == "`":
            _add_flag(flags, "pseudo_each")
        i += 1
    if i >= len(rest) or rest[i] != "h":
        return None
    i += 1
    # Flags may also follow 'h' (e.g. 1hx[4:1]).
    while i < len(rest) and rest[i] in _HOLD_FLAG_CHARS:
        if rest[i] == "$":
            if i + 1 < len(rest) and rest[i + 1] == "$":
                _add_flag(flags, "rotate_star")
                i += 2
                continue
            _add_flag(flags, "force_star")
        elif rest[i] == "b":
            _add_flag(flags, "break")
        elif rest[i] == "x":
            _add_flag(flags, "ex")
        elif rest[i] == "@":
            _add_flag(flags, "force_tap")
        elif rest[i] == "`":
            _add_flag(flags, "pseudo_each")
        i += 1
    duration: Duration | None = None
    if i < len(rest) and rest[i] == "[":
        raw, i = _parse_bracket(rest, i)
        duration = _parse_duration(raw)
    else:
        duration = Duration(bare=True)
    # Allow trailing flags after the bracket.
    while i < len(rest) and rest[i] in _HOLD_FLAG_CHARS:
        if rest[i] == "b":
            _add_flag(flags, "break")
        elif rest[i] == "x":
            _add_flag(flags, "ex")
        elif rest[i] == "$":
            _add_flag(flags, "force_star")
        elif rest[i] == "@":
            _add_flag(flags, "force_tap")
        elif rest[i] == "`":
            _add_flag(flags, "pseudo_each")
        i += 1
    if i != len(rest):
        raise ParseError(f"invalid hold token {token!r}")
    return Event(
        tick=tick,
        kind="hold",
        position=position,
        flags=flags,
        duration=duration,
        raw=token,
    )


def _read_slide_shape(text: str, pos: int) -> tuple[str, int] | None:
    if text.startswith("pp", pos):
        return "pp", pos + 2
    if text.startswith("qq", pos):
        return "qq", pos + 2
    if pos < len(text) and text[pos] in "-^<>vpqszwV":
        return text[pos], pos + 1
    return None


def _read_digit(text: str, pos: int) -> tuple[str, int] | None:
    if pos < len(text) and text[pos] in "12345678":
        return text[pos], pos + 1
    return None


def _parse_track_flags(text: str, pos: int) -> tuple[list[str], int, bool]:
    """Parse flags after a slide duration.  Returns (flags, pos, saw_star)."""
    flags: list[str] = []
    saw_star = False
    while pos < len(text):
        c = text[pos]
        if c == "*":
            saw_star = True
            pos += 1
            break
        if c == "b":
            _add_flag(flags, "break")
            pos += 1
        elif c == "x":
            _add_flag(flags, "ex")
            pos += 1
        elif c == "$":
            _add_flag(flags, "force_star")
            pos += 1
        elif c == "@":
            _add_flag(flags, "force_tap")
            pos += 1
        elif c == "?":
            _add_flag(flags, "no_star_wait")
            pos += 1
        elif c == "!":
            _add_flag(flags, "no_star_move")
            pos += 1
        elif c == "`":
            _add_flag(flags, "pseudo_each")
            pos += 1
        else:
            break
    return flags, pos, saw_star


def _parse_slide_piece(
    text: str, pos: int
) -> tuple[SlideSegment | None, int, bool]:
    """Parse one slide segment.

    Returns ``(segment, next_pos, saw_star)``.  Leading flags are track flags
    for chained segments; for the first segment the caller has already consumed
    star flags.  A trailing ``*`` is consumed and reported via ``saw_star``.
    """
    leading_flags: list[str] = []
    while pos < len(text) and text[pos] in _TRACK_FLAG_CHARS:
        c = text[pos]
        if c == "b":
            _add_flag(leading_flags, "break")
        elif c == "x":
            _add_flag(leading_flags, "ex")
        elif c == "?":
            _add_flag(leading_flags, "no_star_wait")
        elif c == "!":
            _add_flag(leading_flags, "no_star_move")
        elif c == "@":
            _add_flag(leading_flags, "force_tap")
        elif c == "$":
            if pos + 1 < len(text) and text[pos + 1] == "$":
                _add_flag(leading_flags, "rotate_star")
                pos += 1
            else:
                _add_flag(leading_flags, "force_star")
        elif c == "`":
            _add_flag(leading_flags, "pseudo_each")
        pos += 1

    shape_info = _read_slide_shape(text, pos)
    if shape_info is None:
        return None, pos, False
    shape, pos = shape_info
    via: str | None = None
    if shape == "V":
        via_info = _read_digit(text, pos)
        if via_info is None:
            raise ParseError(f"V slide missing via point in {text!r}")
        via, pos = via_info
        if pos < len(text) and text[pos] == "-":
            pos += 1
        end_info = _read_digit(text, pos)
        if end_info is None:
            raise ParseError(f"V slide missing end point in {text!r}")
        end, pos = end_info
    else:
        end_info = _read_digit(text, pos)
        if end_info is None:
            raise ParseError(f"slide missing end point in {text!r}")
        end, pos = end_info

    duration: Duration | None = None
    if pos < len(text) and text[pos] == "[":
        raw, pos = _parse_bracket(text, pos)
        duration = _parse_duration(raw)

    track_flags, pos, saw_star = _parse_track_flags(text, pos)
    flags = leading_flags + track_flags
    segment = SlideSegment(
        shape=shape,
        end=end,
        via=via,
        duration=duration,
        flags=flags,
        raw=text[:pos],
    )
    return segment, pos, saw_star


def _parse_slide(token: str, tick: int) -> Event | None:
    if not token or token[0] not in "12345678":
        return None
    start = token[0]
    pos = 1
    star_flags: list[str] = []
    while pos < len(token) and token[pos] in _SLIDE_FLAG_CHARS:
        c = token[pos]
        if c == "$":
            if pos + 1 < len(token) and token[pos + 1] == "$":
                _add_flag(star_flags, "rotate_star")
                pos += 2
                continue
            _add_flag(star_flags, "force_star")
        elif c == "b":
            _add_flag(star_flags, "break")
        elif c == "x":
            _add_flag(star_flags, "ex")
        elif c == "?":
            _add_flag(star_flags, "no_star_wait")
        elif c == "!":
            _add_flag(star_flags, "no_star_move")
        elif c == "@":
            _add_flag(star_flags, "force_tap")
        elif c == "`":
            _add_flag(star_flags, "pseudo_each")
        pos += 1

    segments: list[SlideSegment] = []
    chain_type = "single"
    saw_same_start = False
    saw_path = False

    first, pos, saw_star = _parse_slide_piece(token, pos)
    if first is None:
        return None
    segments.append(first)
    if saw_star:
        saw_same_start = True

    while pos < len(token):
        if saw_star:
            # The previous piece ended in '*' (same-start chain).  Parse the
            # next segment directly; it starts from the same star tap.
            segment, pos, saw_star = _parse_slide_piece(token, pos)
            if segment is None:
                raise ParseError(f"invalid slide chain {token!r}")
            segments.append(segment)
            continue

        if token[pos] == "*":
            # Defensive: normally '*' is consumed by _parse_track_flags.
            pos += 1
            if pos >= len(token):
                raise ParseError(f"slide chain ends with '*' in {token!r}")
            saw_same_start = True
            saw_star = True
            continue

        if _read_slide_shape(token, pos) is not None:
            saw_path = True
            segment, pos, saw_star = _parse_slide_piece(token, pos)
            if segment is None:
                raise ParseError(f"invalid path slide {token!r}")
            segments.append(segment)
            continue

        raise ParseError(f"invalid slide token {token!r} at offset {pos}")

    if saw_star:
        raise ParseError(f"slide chain ends with '*' in {token!r}")
    if saw_same_start and saw_path:
        raise ParseError(f"mixed '*' and path slide syntax in {token!r}")
    if saw_same_start:
        chain_type = "same_start"
    elif saw_path:
        chain_type = "path"

    event_duration: Duration | None = None
    if chain_type == "path":
        # A connected path has one duration for the whole path.  Move the last
        # duration from the final segment to the event.
        if segments[-1].duration is not None:
            event_duration = segments[-1].duration
            segments[-1].duration = None

    return Event(
        tick=tick,
        kind="slide",
        position=start,
        flags=star_flags,
        duration=event_duration,
        slide=segments,
        chain_type=chain_type,
        raw=token,
    )


def _parse_tap(token: str, tick: int) -> list[Event] | None:
    if not token or token[0] not in "12345678":
        return None
    positions: list[str] = []
    pos = 0
    while pos < len(token) and token[pos] in "12345678":
        positions.append(token[pos])
        pos += 1
    flags: list[str] = []
    while pos < len(token):
        c = token[pos]
        if c == "$":
            if pos + 1 < len(token) and token[pos + 1] == "$":
                _add_flag(flags, "rotate_star")
                pos += 2
                continue
            _add_flag(flags, "force_star")
        elif c == "b":
            _add_flag(flags, "break")
        elif c == "x":
            _add_flag(flags, "ex")
        elif c == "@":
            _add_flag(flags, "force_tap")
        elif c == "`":
            _add_flag(flags, "pseudo_each")
        else:
            raise ParseError(f"invalid tap token {token!r}")
        pos += 1
    return [
        Event(tick=tick, kind="tap", position=p, flags=list(flags), raw=token)
        for p in positions
    ]
