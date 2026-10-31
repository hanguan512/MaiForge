"""Timing information shared by the audio frontend and chart generator."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..maidata.model import TICKS_PER_BEAT


@dataclass
class TimingInfo:
    path: str
    sr: int
    duration: float
    bpm: float
    bpm_confidence: float
    beat_times: list[float]
    downbeat_times: list[float]
    onset_times: list[float]
    onset_strengths: list[float]
    first: float = 0.0
    beat_phase: float = 0.0
    downbeat_phase: int = 0
    tags: dict[str, str] = field(default_factory=dict)
    # Optional manual tempo map: [{"tick": 0, "bpm": 180}, ...].  Empty means
    # constant ``bpm``.  Unlike beat_times this maps chart ticks directly.
    bpm_changes: list[dict[str, float]] = field(default_factory=list)

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def tick_at(self, seconds: float, *, quantize: int = 1) -> int | None:
        """Convert absolute audio seconds to chart ticks.

        ``quantize`` is a tick grid (e.g. 240 for a 16th note); 0/1 means no
        quantization.  Returns ``None`` for times before chart tick 0.
        """
        chart_seconds = seconds - self.first
        if chart_seconds < -1e-6:
            return None
        ticks = chart_seconds / self.seconds_per_beat * TICKS_PER_BEAT
        if quantize and quantize > 1:
            ticks = round(ticks / quantize) * quantize
        return int(round(ticks))

    def beat_ticks(self, *, quantize: int = 1) -> list[int]:
        return [
            t
            for t in (self.tick_at(x, quantize=quantize) for x in self.beat_times)
            if t is not None
        ]

    def downbeat_ticks(self, *, quantize: int = 1) -> list[int]:
        return [
            t
            for t in (self.tick_at(x, quantize=quantize) for x in self.downbeat_times)
            if t is not None
        ]

    @property
    def bar_count(self) -> int:
        if not self.beat_times:
            return 0
        return max(0, len(self.beat_times) // 4)

    def retime(self, bpm: float) -> None:
        """Rebuild a constant-tempo beat grid from the current phase."""
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.bpm = float(bpm)
        period = 60.0 / self.bpm
        start = float(self.beat_phase)
        if start < 0:
            start = start % period
        beats: list[float] = []
        t = start
        while t <= self.duration + period:
            if t >= 0:
                beats.append(round(t, 6))
            t += period
        if not beats:
            beats = [0.0]
        self.beat_times = beats
        self.downbeat_times = beats[self.downbeat_phase :: 4]
        self.beat_phase = beats[0]
        self.bpm_changes = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sr": self.sr,
            "duration": round(float(self.duration), 6),
            "bpm": round(float(self.bpm), 6),
            "bpm_confidence": round(float(self.bpm_confidence), 4),
            "first": round(float(self.first), 6),
            "beat_phase": round(float(self.beat_phase), 6),
            "downbeat_phase": int(self.downbeat_phase),
            "tags": dict(self.tags),
            "beat_times": [round(float(x), 6) for x in self.beat_times],
            "downbeat_times": [round(float(x), 6) for x in self.downbeat_times],
            "onset_times": [round(float(x), 6) for x in self.onset_times],
            "onset_strengths": [round(float(x), 6) for x in self.onset_strengths],
            "bpm_changes": [
                {"tick": int(change["tick"]), "bpm": round(float(change["bpm"]), 6)}
                for change in self.bpm_changes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingInfo":
        return cls(
            path=str(data.get("path", "")),
            sr=int(data.get("sr", 22050)),
            duration=float(data.get("duration", 0.0)),
            bpm=float(data.get("bpm", 120.0)),
            bpm_confidence=float(data.get("bpm_confidence", 0.0)),
            beat_times=[float(x) for x in data.get("beat_times", [])],
            downbeat_times=[float(x) for x in data.get("downbeat_times", [])],
            onset_times=[float(x) for x in data.get("onset_times", [])],
            onset_strengths=[float(x) for x in data.get("onset_strengths", [])],
            first=float(data.get("first", 0.0)),
            beat_phase=float(data.get("beat_phase", 0.0)),
            downbeat_phase=int(data.get("downbeat_phase", 0)),
            tags={str(k): str(v) for k, v in data.get("tags", {}).items()},
            bpm_changes=[
                {"tick": int(change["tick"]), "bpm": float(change["bpm"])}
                for change in data.get("bpm_changes", [])
            ],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TimingInfo":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
