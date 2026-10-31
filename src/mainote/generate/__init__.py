"""Generation entry points."""

from __future__ import annotations

from ..audio.timing import TimingInfo
from ..maidata.model import Chart, Maidata
from .rule import RuleConfig, RuleResult, generate_chart

__all__ = [
    "RuleConfig",
    "RuleResult",
    "generate_chart",
    "generate_from_audio",
    "maidata_for_chart",
]


def generate_from_audio(
    audio_path: str,
    *,
    difficulty: str = "master",
    level: float | None = None,
    seed: int = 0,
    sr: int = 22050,
    title: str | None = None,
    artist: str | None = None,
    bpm: float | None = None,
    first: float = 0.0,
    config: RuleConfig | None = None,
) -> tuple[TimingInfo, RuleResult]:
    from ..audio.features import analyze_audio

    timing = analyze_audio(audio_path, sr=sr, bpm=bpm, first=first)
    result = generate_chart(
        timing,
        difficulty=difficulty,
        level=level,
        seed=seed,
        config=config,
        title=title,
        artist=artist,
    )
    return timing, result


def maidata_for_chart(
    chart: Chart,
    *,
    title: str,
    artist: str = "",
    wholebpm: float | None = None,
    clock_count: int | None = None,
) -> Maidata:
    # Keep the generated metadata minimal: title/artist/BPM are enough to load
    # the chart.  Emit &first only when it is non-zero and &clock_count only
    # when explicitly requested, matching common maidata files.
    meta = [
        ("title", title),
        ("artist", artist),
        ("wholebpm", _fmt(wholebpm if wholebpm is not None else chart.initial_bpm)),
    ]
    if abs(chart.first) > 1e-9:
        meta.insert(2, ("first", _fmt(chart.first)))
    if clock_count is not None:
        meta.append(("clock_count", str(clock_count)))
    return Maidata(metadata_items=meta, charts=[chart])


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")
