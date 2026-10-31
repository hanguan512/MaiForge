"""Chart event tokenizer for the P3 sequence model.

Token grammar (one event per ``<END_EVENT>``):

    <BOS> <DIFF_n> <LEVEL_k> <BAR>* <T_k> <TYPE> <POS>
    [<DUR_n> | <DUR_NONE>]                         # holds / touch holds
    [<CHAIN_*> <SHAPE> [<POS via>] <POS end> <DUR>
     [<TRACK_BREAK>|<TRACK_EX>]* <SHAPE ...> ...]  # slides
    [<FLAG_*>*] <END_EVENT>
    ...
    <EOS>

Time is encoded as ``<BAR>`` advances plus a position within the current bar.
One time unit is 20 ticks (1/48 beat); a 4/4 bar has 192 units.  That supports
1/64 notes and finer while keeping the vocabulary small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..maidata.model import (
    TICKS_PER_BAR,
    TICKS_PER_BEAT,
    Chart,
    Duration,
    Event,
    SlideSegment,
)
from ..maidata.model import TOUCH_SENSORS

TIME_UNIT = 20
UNITS_PER_BAR = TICKS_PER_BAR // TIME_UNIT  # 192
DUR_UNIT = TICKS_PER_BEAT // 48            # 1/48 beat = 20 ticks
MAX_DUR_UNITS = 2048
# level_bucket = round(level * 2); bucket 31 represents 15.5 (15+).
MAX_LEVEL_BUCKET = 31

TYPE_TOKENS = {
    "tap": "TAP",
    "hold": "HOLD",
    "slide": "SLIDE",
    "touch": "TOUCH",
    "touch_hold": "TOUCH_HOLD",
}
TYPE_FROM_TOKEN = {v: k for k, v in TYPE_TOKENS.items()}

SHAPE_TOKENS = {
    "-": "SHAPE_MINUS",
    "^": "SHAPE_CARET",
    "<": "SHAPE_LT",
    ">": "SHAPE_GT",
    "v": "SHAPE_v",
    "p": "SHAPE_p",
    "q": "SHAPE_q",
    "s": "SHAPE_s",
    "z": "SHAPE_z",
    "w": "SHAPE_w",
    "pp": "SHAPE_pp",
    "qq": "SHAPE_qq",
    "V": "SHAPE_V",
}
SHAPE_FROM_TOKEN = {v: k for k, v in SHAPE_TOKENS.items()}

CHAIN_TOKENS = {
    "single": "CHAIN_SINGLE",
    "same_start": "CHAIN_SAME",
    "path": "CHAIN_PATH",
}
CHAIN_FROM_TOKEN = {v: k for k, v in CHAIN_TOKENS.items()}

FLAG_TOKENS = {
    "break": "FLAG_BREAK",
    "ex": "FLAG_EX",
    "firework": "FLAG_FIREWORK",
    "no_star_wait": "FLAG_NO_STAR_WAIT",
    "no_star_move": "FLAG_NO_STAR_MOVE",
    "force_star": "FLAG_FORCE_STAR",
    "rotate_star": "FLAG_ROTATE_STAR",
    "force_tap": "FLAG_FORCE_TAP",
    "pseudo_each": "FLAG_PSEUDO_EACH",
}
FLAG_FROM_TOKEN = {v: k for k, v in FLAG_TOKENS.items()}
FLAG_ORDER = list(FLAG_TOKENS)

TRACK_FLAG_TOKENS = {
    "break": "TRACK_BREAK",
    "ex": "TRACK_EX",
}
TRACK_FLAG_FROM_TOKEN = {v: k for k, v in TRACK_FLAG_TOKENS.items()}


def token_category(token: str) -> str:
    """Classify a token for weighted losses and per-class accuracy."""
    if token.startswith("DUR_"):
        return "duration"
    if token in TYPE_FROM_TOKEN:
        return "type"
    if token.startswith(("SHAPE_", "CHAIN_", "TRACK_")):
        return "structure"
    if token.startswith(("P_", "S_")):
        return "position"
    if token == "BAR" or token.startswith("T_"):
        return "time"
    if token.startswith("FLAG_"):
        return "flag"
    if token.startswith(("DIFF_", "LEVEL_")):
        return "condition"
    return "special"


class TokenizeError(ValueError):
    pass


def _parse_level_value(value: str | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in ("-", "?"):
        return None
    try:
        if text.endswith("+"):
            return float(text[:-1]) + 0.5
        if text.endswith("-"):
            return float(text[:-1]) - 0.5
        return float(text)
    except ValueError:
        return None


def level_bucket(level: str | float | None) -> int:
    value = _parse_level_value(level)
    if value is None:
        return 0
    return max(0, min(MAX_LEVEL_BUCKET, int(round(value * 2))))


class ChartVocabulary:
    """Token strings <-> integer ids."""

    def __init__(self, tokens: list[str]):
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary contains duplicate tokens")
        self.tokens = list(tokens)
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        for required in ("PAD", "BOS", "EOS", "UNK"):
            if required not in self.token_to_id:
                raise ValueError(f"vocabulary missing {required}")

    @classmethod
    def default(cls) -> "ChartVocabulary":
        tokens: list[str] = [
            "PAD",
            "BOS",
            "EOS",
            "UNK",
            "END_EVENT",
            "BAR",
        ]
        tokens += [f"T_{i}" for i in range(UNITS_PER_BAR)]
        tokens += [f"DIFF_{i}" for i in range(1, 8)]
        tokens += [f"LEVEL_{i}" for i in range(MAX_LEVEL_BUCKET + 1)]
        tokens += list(TYPE_TOKENS.values())
        tokens += [f"P_{i}" for i in range(1, 9)]
        tokens += [f"S_{sensor}" for sensor in TOUCH_SENSORS]
        tokens += ["DUR_NONE"] + [f"DUR_{i}" for i in range(1, MAX_DUR_UNITS + 1)]
        tokens += list(SHAPE_TOKENS.values())
        tokens += list(CHAIN_TOKENS.values())
        tokens += list(FLAG_TOKENS.values())
        tokens += list(TRACK_FLAG_TOKENS.values())
        return cls(tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["PAD"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["BOS"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["EOS"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["UNK"]

    @property
    def end_event_id(self) -> int:
        return self.token_to_id["END_EVENT"]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        out = []
        for token in tokens:
            if token not in self.token_to_id:
                raise TokenizeError(f"unknown token {token!r}")
            out.append(self.token_to_id[token])
        return out

    def decode(self, ids: Iterable[int]) -> list[str]:
        out = []
        for idx in ids:
            token = self.id_to_token.get(int(idx))
            if token is None:
                raise TokenizeError(f"unknown token id {idx}")
            out.append(token)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChartVocabulary":
        return cls([str(t) for t in data["tokens"]])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ChartVocabulary":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ParsedSequence:
    events: list[Event] = field(default_factory=list)
    difficulty_index: int | None = None
    level_bucket: int | None = None
    done: bool = False


class ChartTokenGrammar:
    """Incremental parser and constrained-decoding state machine.

    ``strict=True`` additionally enforces musical constraints that greedy
    decoding otherwise violates: no two events at the same tick/position, no
    new event inside an active hold, and no degenerate straight slide whose
    end equals its start.  Tokenizing and round-tripping existing charts uses
    the default permissive mode so unusual but legal source data still parses.
    """

    def __init__(self, vocab: ChartVocabulary, *, strict: bool = False):
        self.vocab = vocab
        self.strict = strict
        self.parsed = ParsedSequence()
        self.state = "start"
        self.bar = 0
        self.last_event_tick = -1
        self.current_tick: int | None = None
        self.current_type: str | None = None
        self.current_pos: str | None = None
        self.current_duration: Duration | None = None
        self.current_chain: str | None = None
        self.current_flags: list[str] = []
        self.current_segments: list[SlideSegment] = []
        self.current_segment: SlideSegment | None = None
        self.used_flags: set[str] = set()
        # Positions already occupied at ``tick_positions_tick``.
        self.tick_positions_tick: int | None = None
        self.tick_positions: set[str] = set()
        # Outer position / touch sensor -> end tick of the active hold.
        self.active_holds: dict[str, int] = {}

    # ------------------------------------------------------------------
    def feed(self, token: str) -> None:
        if token == "PAD":
            return
        if token == "BOS":
            if self.state != "start":
                raise TokenizeError("BOS in the middle of a sequence")
            return
        if token == "EOS":
            if self.state not in ("start", "time"):
                raise TokenizeError(f"EOS in unexpected state {self.state!r}")
            self.parsed.done = True
            self.state = "done"
            return
        if token == "UNK":
            raise TokenizeError("UNK token")
        if token.startswith("DIFF_"):
            if self.state != "start":
                raise TokenizeError("difficulty token after chart started")
            self.parsed.difficulty_index = int(token[len("DIFF_") :])
            return
        if token.startswith("LEVEL_"):
            if self.state != "start":
                raise TokenizeError("level token after chart started")
            self.parsed.level_bucket = int(token[len("LEVEL_") :])
            return
        if token == "BAR":
            if self.state not in ("start", "time"):
                raise TokenizeError("BAR outside a time position")
            self.bar += 1
            self.state = "time"
            return
        if token.startswith("T_"):
            if self.state not in ("start", "time"):
                raise TokenizeError("time token outside a time position")
            unit = int(token[len("T_") :])
            if not 0 <= unit < UNITS_PER_BAR:
                raise TokenizeError(f"time unit out of range: {token}")
            tick = self.bar * TICKS_PER_BAR + unit * TIME_UNIT
            if tick < self.last_event_tick:
                raise TokenizeError("time moved backwards")
            if self.strict and tick != self.tick_positions_tick:
                self.tick_positions.clear()
                self.tick_positions_tick = tick
            self.current_tick = tick
            self.state = "type"
            return

        if token in TYPE_FROM_TOKEN:
            if self.state != "type":
                raise TokenizeError(f"type token in state {self.state!r}")
            self.current_type = TYPE_FROM_TOKEN[token]
            self.current_flags = []
            self.current_segments = []
            self.current_segment = None
            self.used_flags = set()
            self.state = "pos"
            return

        if token.startswith("P_") or token.startswith("S_"):
            if self.state == "pos":
                self._set_position(token)
                if self.current_type == "slide":
                    self.state = "slide_chain"
                elif self.current_type in ("hold", "touch_hold"):
                    self.state = "hold_dur"
                else:
                    self.state = "tap_flags"
                return
            if self.state == "slide_via":
                if not token.startswith("P_"):
                    raise TokenizeError("slide via must be an outer position")
                self._current_segment_require().via = token[len("P_") :]
                self.state = "slide_end"
                return
            if self.state == "slide_end":
                if not token.startswith("P_"):
                    raise TokenizeError("slide end must be an outer position")
                end = token[len("P_") :]
                segment = self._current_segment_require()
                if self.strict and segment.shape == "-" and end == self._segment_start():
                    raise TokenizeError(
                        "straight slide cannot end at its start position"
                    )
                segment.end = end
                self.state = "slide_dur"
                return
            raise TokenizeError(f"position token in state {self.state!r}")

        if token.startswith("DUR_"):
            if self.state == "hold_dur":
                self.current_duration = self._duration_from_token(token)
                self.state = "hold_flags"
                return
            if self.state == "slide_dur":
                segment = self._current_segment_require()
                segment.duration = self._duration_from_token(token)
                self.state = "slide_after_dur"
                return
            raise TokenizeError(f"duration token in state {self.state!r}")

        if token in CHAIN_FROM_TOKEN:
            if self.state != "slide_chain":
                raise TokenizeError("chain token outside a slide")
            self.current_chain = CHAIN_FROM_TOKEN[token]
            self.state = "slide_shape"
            return

        if token in SHAPE_FROM_TOKEN:
            if self.state == "slide_shape":
                if self.current_segment is not None:
                    self.current_segments.append(self.current_segment)
                self.current_segment = SlideSegment(
                    shape=SHAPE_FROM_TOKEN[token], end="", raw=""
                )
                if self.current_segment.shape == "V":
                    self.state = "slide_via"
                else:
                    self.state = "slide_end"
                return
            if self.state == "slide_after_dur":
                if self.current_chain == "single":
                    raise TokenizeError("single slide cannot have another segment")
                if self.current_segment is not None:
                    self.current_segments.append(self.current_segment)
                self.current_segment = SlideSegment(
                    shape=SHAPE_FROM_TOKEN[token], end="", raw=""
                )
                if self.current_segment.shape == "V":
                    self.state = "slide_via"
                else:
                    self.state = "slide_end"
                return
            raise TokenizeError(f"shape token in state {self.state!r}")

        if token in TRACK_FLAG_FROM_TOKEN:
            if self.state != "slide_after_dur":
                raise TokenizeError("track flag outside a slide segment")
            segment = self._current_segment_require()
            flag = TRACK_FLAG_FROM_TOKEN[token]
            if flag not in segment.flags:
                segment.flags.append(flag)
            return

        if token in FLAG_FROM_TOKEN:
            if self.state not in ("tap_flags", "hold_flags", "slide_after_dur", "slide_flags"):
                raise TokenizeError(f"flag token in state {self.state!r}")
            flag = FLAG_FROM_TOKEN[token]
            if flag not in self._allowed_event_flags():
                raise TokenizeError(
                    f"flag {flag!r} is not valid for {self.current_type!r}"
                )
            if flag not in self.used_flags:
                self.used_flags.add(flag)
                self.current_flags.append(flag)
            if self.state == "slide_after_dur":
                self.state = "slide_flags"
            return

        if token == "END_EVENT":
            if self.state not in ("tap_flags", "hold_flags", "slide_after_dur", "slide_flags"):
                raise TokenizeError(f"END_EVENT in state {self.state!r}")
            self._finish_event()
            return

        raise TokenizeError(f"unexpected token {token!r} in state {self.state!r}")

    # ------------------------------------------------------------------
    def allowed_tokens(self) -> set[str]:
        state = self.state
        if state == "done":
            return set()
        if state == "start":
            allowed = {"BOS", "EOS"}
            allowed |= {f"DIFF_{i}" for i in range(1, 8)}
            allowed |= {f"LEVEL_{i}" for i in range(MAX_LEVEL_BUCKET + 1)}
            allowed.add("BAR")
            allowed |= self._allowed_time_tokens()
            return allowed
        if state == "time":
            allowed = {"BAR", "EOS"}
            allowed |= self._allowed_time_tokens()
            return allowed
        if state == "type":
            return set(TYPE_TOKENS.values())
        if state == "pos":
            if self.current_type in ("tap", "hold", "slide"):
                allowed = {f"P_{i}" for i in range(1, 9)}
            elif self.current_type in ("touch", "touch_hold"):
                allowed = {f"S_{sensor}" for sensor in TOUCH_SENSORS}
            else:
                return set()
            if self.strict:
                blocked = self._blocked_positions()
                allowed = {
                    token
                    for token in allowed
                    if token.split("_", 1)[1] not in blocked
                }
            return allowed
        if state == "hold_dur":
            return {"DUR_NONE"} | {f"DUR_{i}" for i in range(1, MAX_DUR_UNITS + 1)}
        if state in ("tap_flags", "hold_flags"):
            return self._allowed_flags() | {"END_EVENT"}
        if state == "slide_chain":
            return set(CHAIN_TOKENS.values())
        if state == "slide_shape":
            return set(SHAPE_TOKENS.values())
        if state in ("slide_via", "slide_end"):
            allowed = {f"P_{i}" for i in range(1, 9)}
            if (
                state == "slide_end"
                and self.strict
                and self.current_segment is not None
                and self.current_segment.shape == "-"
            ):
                allowed.discard(f"P_{self._segment_start()}")
            return allowed
        if state == "slide_dur":
            return {"DUR_NONE"} | {f"DUR_{i}" for i in range(1, MAX_DUR_UNITS + 1)}
        if state == "slide_after_dur":
            allowed = {"END_EVENT"}
            allowed |= {token for token, flag in TRACK_FLAG_FROM_TOKEN.items() if flag not in self.current_segment.flags} if self.current_segment else set()
            if self.current_chain != "single":
                allowed |= set(SHAPE_TOKENS.values())
            allowed |= self._allowed_flags()
            return allowed
        if state == "slide_flags":
            return self._allowed_flags() | {"END_EVENT"}
        return set()

    def finish(self) -> ParsedSequence:
        if self.state not in ("time", "done"):
            raise TokenizeError(f"incomplete sequence in state {self.state!r}")
        return self.parsed

    # ------------------------------------------------------------------
    def _allowed_time_tokens(self) -> set[str]:
        allowed = set()
        for unit in range(UNITS_PER_BAR):
            tick = self.bar * TICKS_PER_BAR + unit * TIME_UNIT
            if tick >= self.last_event_tick:
                allowed.add(f"T_{unit}")
        return allowed

    def _allowed_event_flags(self) -> set[str]:
        allowed = set(FLAG_TOKENS)
        if self.current_type != "slide":
            allowed.discard("no_star_wait")
            allowed.discard("no_star_move")
        if self.current_type not in ("touch", "touch_hold"):
            allowed.discard("firework")
        if self.current_type in ("hold", "touch_hold"):
            allowed.discard("force_star")
            allowed.discard("rotate_star")
            allowed.discard("force_tap")
        return allowed

    def _allowed_flags(self) -> set[str]:
        allowed = self._allowed_event_flags()
        return {
            token
            for token, flag in FLAG_FROM_TOKEN.items()
            if flag in allowed and flag not in self.used_flags
        }

    def _set_position(self, token: str) -> None:
        if self.current_type in ("tap", "hold", "slide"):
            if not token.startswith("P_"):
                raise TokenizeError(f"outer position expected, got {token}")
            self.current_pos = token[len("P_") :]
        else:
            if not token.startswith("S_"):
                raise TokenizeError(f"touch sensor expected, got {token}")
            self.current_pos = token[len("S_") :]
        if self.strict:
            self._expire_holds()
            if self.current_pos in self.tick_positions:
                raise TokenizeError(
                    f"position {self.current_pos} already used at tick "
                    f"{self.current_tick}"
                )
            hold_end = self.active_holds.get(self.current_pos)
            if hold_end is not None and hold_end > (self.current_tick or 0):
                raise TokenizeError(
                    f"position {self.current_pos} is inside an active hold "
                    f"until tick {hold_end}"
                )
            self.tick_positions.add(self.current_pos)

    def _segment_start(self) -> str | None:
        """Start position of the slide segment currently being built."""
        if not self.current_segments or self.current_chain == "same_start":
            return self.current_pos
        return self.current_segments[-1].end

    def _expire_holds(self) -> None:
        tick = self.current_tick if self.current_tick is not None else 0
        expired = [pos for pos, end in self.active_holds.items() if end <= tick]
        for pos in expired:
            del self.active_holds[pos]

    def _blocked_positions(self) -> set[str]:
        self._expire_holds()
        tick = self.current_tick if self.current_tick is not None else 0
        blocked = set(self.tick_positions)
        blocked |= {
            pos for pos, end in self.active_holds.items() if end > tick
        }
        return blocked

    def _current_segment_require(self) -> SlideSegment:
        if self.current_segment is None:
            raise TokenizeError("slide segment missing")
        return self.current_segment

    def _duration_from_token(self, token: str) -> Duration | None:
        if token == "DUR_NONE":
            return None
        units = int(token[len("DUR_") :])
        if not 1 <= units <= MAX_DUR_UNITS:
            raise TokenizeError(f"duration out of range: {token}")
        return Duration(ticks=units * DUR_UNIT)

    def _finish_event(self) -> None:
        if self.current_tick is None or self.current_type is None or self.current_pos is None:
            raise TokenizeError("event missing time/type/position")
        if self.current_type == "slide":
            if self.current_segment is not None:
                self.current_segments.append(self.current_segment)
            if not self.current_segments:
                raise TokenizeError("slide has no segments")
            chain = self.current_chain or "single"
            event_duration = None
            if chain == "path":
                last = self.current_segments[-1]
                event_duration = last.duration
                last.duration = None
            event = Event(
                tick=self.current_tick,
                kind="slide",
                position=self.current_pos,
                flags=list(self.current_flags),
                duration=event_duration,
                slide=list(self.current_segments),
                chain_type=chain,
                raw="",
            )
        else:
            event = Event(
                tick=self.current_tick,
                kind=self.current_type,
                position=self.current_pos,
                flags=list(self.current_flags),
                duration=self.current_duration
                if self.current_type in ("hold", "touch_hold")
                else None,
                raw="",
            )
        self.parsed.events.append(event)
        if (
            self.strict
            and self.current_type in ("hold", "touch_hold")
            and self.current_duration is not None
            and self.current_duration.ticks
        ):
            self.active_holds[self.current_pos] = (
                self.current_tick + self.current_duration.ticks
            )
        self.last_event_tick = max(self.last_event_tick, self.current_tick)
        self.current_tick = None
        self.current_type = None
        self.current_pos = None
        self.current_duration = None
        self.current_chain = None
        self.current_flags = []
        self.current_segments = []
        self.current_segment = None
        self.used_flags = set()
        self.state = "time"


class EventTokenizer:
    """Serialize chart windows into model tokens and parse them back."""

    def __init__(self, vocab: ChartVocabulary | None = None):
        self.vocab = vocab or ChartVocabulary.default()

    def tokenize_chart(
        self,
        chart: Chart,
        *,
        start_tick: int = 0,
        end_tick: int | None = None,
        difficulty_index: int | None = None,
        level: str | float | None = None,
        include_special: bool = True,
    ) -> list[str]:
        if start_tick % TICKS_PER_BAR != 0:
            raise TokenizeError("window start_tick must be bar-aligned")
        end_tick = end_tick if end_tick is not None else chart.max_tick + TICKS_PER_BAR
        tokens: list[str] = []
        if include_special:
            tokens.append("BOS")
            tokens.append(f"DIFF_{difficulty_index or chart.index}")
            tokens.append(f"LEVEL_{level_bucket(level if level is not None else chart.level)}")
        last_bar = 0
        for event in sorted(chart.events, key=lambda e: e.sort_key()):
            if event.tick < start_tick or event.tick >= end_tick:
                continue
            bar, unit = self._relative_bar_unit(event.tick, start_tick)
            while last_bar < bar:
                tokens.append("BAR")
                last_bar += 1
            tokens.append(f"T_{unit}")
            tokens.extend(self._tokenize_event(event, chart))
        if include_special:
            tokens.append("EOS")
        return tokens

    def encode_chart(
        self,
        chart: Chart,
        *,
        start_tick: int = 0,
        end_tick: int | None = None,
        difficulty_index: int | None = None,
        level: str | float | None = None,
        include_special: bool = True,
    ) -> list[int]:
        return self.vocab.encode(
            self.tokenize_chart(
                chart,
                start_tick=start_tick,
                end_tick=end_tick,
                difficulty_index=difficulty_index,
                level=level,
                include_special=include_special,
            )
        )

    def detokenize(
        self,
        tokens: Iterable[str],
        *,
        start_tick: int = 0,
    ) -> ParsedSequence:
        grammar = ChartTokenGrammar(self.vocab)
        for token in tokens:
            grammar.feed(token)
            if grammar.parsed.done:
                break
        parsed = grammar.finish()
        if start_tick:
            for event in parsed.events:
                event.tick += start_tick
        return parsed

    def decode_ids(
        self,
        ids: Iterable[int],
        *,
        start_tick: int = 0,
    ) -> ParsedSequence:
        return self.detokenize(self.vocab.decode(ids), start_tick=start_tick)

    # ------------------------------------------------------------------
    def _tokenize_event(self, event: Event, chart: Chart) -> list[str]:
        tokens = [TYPE_TOKENS[event.kind], self._position_token(event)]
        if event.kind in ("hold", "touch_hold"):
            tokens.append(self._duration_token(event.duration, event.tick, chart))
        elif event.kind == "slide":
            if not event.slide:
                raise TokenizeError(f"slide at tick {event.tick} has no segments")
            tokens.append(CHAIN_TOKENS.get(event.chain_type, "CHAIN_SINGLE"))
            for index, segment in enumerate(event.slide):
                tokens.append(SHAPE_TOKENS[segment.shape])
                if segment.shape == "V":
                    if not segment.via:
                        raise TokenizeError("V slide missing via point")
                    tokens.append(f"P_{segment.via}")
                tokens.append(f"P_{segment.end}")
                if event.chain_type == "path":
                    duration = event.duration if index == len(event.slide) - 1 else None
                else:
                    duration = segment.duration
                    if duration is None and index == len(event.slide) - 1:
                        duration = event.duration
                tokens.append(self._duration_token(duration, event.tick, chart))
                for flag in sorted(set(segment.flags)):
                    if flag in TRACK_FLAG_TOKENS:
                        tokens.append(TRACK_FLAG_TOKENS[flag])
        for flag in sorted(set(event.flags), key=lambda f: FLAG_ORDER.index(f) if f in FLAG_ORDER else 99):
            if flag not in FLAG_TOKENS:
                raise TokenizeError(f"unsupported event flag {flag!r}")
            tokens.append(FLAG_TOKENS[flag])
        tokens.append("END_EVENT")
        return tokens

    @staticmethod
    def _position_token(event: Event) -> str:
        if event.kind in ("touch", "touch_hold"):
            return f"S_{event.position}"
        return f"P_{event.position}"

    @staticmethod
    def _relative_bar_unit(tick: int, start_tick: int) -> tuple[int, int]:
        offset = tick - start_tick
        bar = offset // TICKS_PER_BAR
        unit = int(round((offset % TICKS_PER_BAR) / TIME_UNIT))
        if unit >= UNITS_PER_BAR:
            bar += unit // UNITS_PER_BAR
            unit %= UNITS_PER_BAR
        return bar, unit

    @staticmethod
    def _duration_token(
        duration: Duration | None,
        event_tick: int,
        chart: Chart,
    ) -> str:
        if duration is None or duration.bare:
            return "DUR_NONE"
        if duration.ticks is not None:
            ticks = duration.ticks
        elif duration.seconds is not None:
            bpm = chart.bpm_at(event_tick)
            ticks = int(round(duration.seconds * bpm / 60.0 * TICKS_PER_BEAT))
        else:
            return "DUR_NONE"
        units = int(round(ticks / DUR_UNIT))
        if units <= 0:
            return "DUR_NONE"
        return f"DUR_{max(1, min(MAX_DUR_UNITS, units))}"
