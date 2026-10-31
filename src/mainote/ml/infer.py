"""Run a trained P3 checkpoint on an audio file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..audio.timing import TimingInfo
from ..generate.rule import DEFAULT_LEVELS
from ..maidata.compiler import compile_maidata
from ..maidata.model import (
    TICKS_PER_BAR,
    TICKS_PER_BEAT,
    BpmChange,
    Chart,
    DivisionChange,
    Event,
)
from ..maidata.stats import DIFFICULTY_BY_INDEX, chart_stats
from ..maidata.validate import validate_chart
from .decode import greedy_decode
from .features import log_mel_spectrogram
from .model import ChartModel, ChartModelConfig, build_model
from .tokenizer import ChartVocabulary, EventTokenizer


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ChartModel, ChartVocabulary, EventTokenizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = ChartModelConfig.from_dict(checkpoint.get("model_config"))
    tokens = checkpoint.get("vocab_tokens")
    if not tokens:
        raise RuntimeError("checkpoint has no vocab_tokens")
    vocab = ChartVocabulary([str(t) for t in tokens])
    tokenizer = EventTokenizer(vocab)
    model = build_model(
        len(vocab.tokens), model_cfg, pad_id=vocab.pad_id
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocab, tokenizer, checkpoint


def infer(
    *,
    audio_path: str,
    checkpoint_path: str,
    output_path: str,
    difficulty: str = "master",
    level: float | None = None,
    timing_path: str | None = None,
    features_path: str | None = None,
    device_name: str | None = None,
    window_bars: int | None = None,
    stride_bars: int | None = None,
    context_bars: int | None = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
    top_k: int | None = None,
    seed: int = 0,
    title: str | None = None,
    artist: str | None = None,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    model, vocab, tokenizer, checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = checkpoint.get("config") or {}
    data_cfg = cfg.get("data", {})
    window_bars = int(window_bars or data_cfg.get("window_bars", 16))
    stride_bars = int(stride_bars or data_cfg.get("stride_bars", 8))
    context_bars = int(context_bars or data_cfg.get("context_bars", 2))

    if timing_path:
        timing = TimingInfo.load(timing_path)
    else:
        from ..audio.features import analyze_audio

        timing = analyze_audio(audio_path)
    if timing.bpm <= 0:
        raise ValueError("timing BPM must be positive")

    mel, fps = _load_or_compute_mel(
        audio_path, features_path, model.config.n_mels
    )
    difficulty_index = _difficulty_index(difficulty)
    level_value = level
    if level_value is None:
        level_value = DEFAULT_LEVELS.get(_normalize_difficulty(difficulty), 12.0)

    sec_per_bar = 4.0 * 60.0 / timing.bpm
    first_onset = min(timing.onset_times) if timing.onset_times else timing.first
    last_onset = max(timing.onset_times) if timing.onset_times else timing.duration
    start_bar = max(0, int(math.floor((first_onset - timing.first) / sec_per_bar)))
    end_bar = max(
        start_bar + window_bars,
        int(math.ceil((last_onset - timing.first + sec_per_bar) / sec_per_bar)),
    )
    max_tick = int(round((timing.duration - timing.first) / (60.0 / timing.bpm) * TICKS_PER_BEAT))
    max_bar = max(1, math.ceil(max_tick / TICKS_PER_BAR))

    events: list[Event] = []
    warnings: list[str] = []
    for start in range(start_bar, end_bar, stride_bars):
        if start >= max_bar:
            break
        win_end_bar = min(start + window_bars, max_bar)
        start_tick = start * TICKS_PER_BAR
        end_tick = win_end_bar * TICKS_PER_BAR
        context_start_tick = max(0, start_tick - context_bars * TICKS_PER_BAR)
        context_end_tick = min(max_tick, end_tick + context_bars * TICKS_PER_BAR)
        a = max(
            0,
            int(
                math.floor(
                    (timing.first + _tick_seconds(context_start_tick, timing.bpm)) * fps
                )
            ),
        )
        b = min(
            len(mel),
            int(
                math.ceil(
                    (timing.first + _tick_seconds(context_end_tick, timing.bpm)) * fps
                )
            ),
        )
        if b <= a:
            continue
        window = mel[a:b]
        mel_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
        mel_mask = torch.zeros((1, window.shape[0]), dtype=torch.bool, device=device)
        try:
            _, parsed = greedy_decode(
                model,
                mel_tensor,
                mel_mask,
                tokenizer=tokenizer,
                difficulty_index=difficulty_index,
                level=level_value,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                allow_incomplete=True,
            )
        except Exception as exc:  # noqa: BLE001 - one bad window should not kill the song
            warnings.append(f"window {start}: decode failed: {exc}")
            continue
        if not parsed.done:
            warnings.append(
                f"window {start}: incomplete sequence, kept {len(parsed.events)} events"
            )
        for event in parsed.events:
            absolute_tick = event.tick + start_tick
            if start_tick <= absolute_tick < end_tick:
                event.tick = absolute_tick
                events.append(event)
        print(
            json.dumps(
                {
                    "window": [start, win_end_bar],
                    "events": len(parsed.events),
                    "kept": len(events),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    events, repair_warnings = _repair_events(events)
    warnings.extend(repair_warnings)
    if not events:
        raise RuntimeError("model generated no events")
    chart = Chart(
        index=difficulty_index,
        level=_fmt_level(level_value),
        designer="mainote-p3",
        first=timing.first,
        initial_bpm=timing.bpm,
        events=events,
        bpm_changes=[BpmChange(0, timing.bpm)],
        division_changes=[DivisionChange(0, 48)],
        metadata={"lv": _fmt_level(level_value), "des": "mainote-p3"},
        end_tick=_round_up_bar(max(e.end_tick() for e in events)),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = validate_chart(chart)
    if report.errors:
        (output_path.parent / "p3_ir.json").write_text(
            json.dumps(chart.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path.parent / "p3_validation.json").write_text(
            json.dumps(
                {"errors": report.errors, "warnings": report.warnings},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            "generated chart failed validation: " + "; ".join(report.errors[:10])
        )

    from ..generate import maidata_for_chart

    title = title or timing.tags.get("title") or Path(audio_path).stem
    artist = artist or timing.tags.get("artist") or ""
    maidata = maidata_for_chart(
        chart, title=title, artist=artist, wholebpm=timing.bpm
    )
    text = compile_maidata(maidata)
    output_path.write_text(text, encoding="utf-8")
    (output_path.parent / "timing.json").write_text(
        json.dumps(timing.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path.parent / "p3_ir.json").write_text(
        json.dumps(maidata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "audio": audio_path,
        "checkpoint": checkpoint_path,
        "maidata": str(output_path),
        "notes": chart.note_count,
        "notes_per_second": chart_stats(chart)["notes_per_second"],
        "difficulty": _normalize_difficulty(difficulty),
        "level": _fmt_level(level_value),
        "bpm": timing.bpm,
        "warnings": warnings + report.warnings,
    }
    (output_path.parent / "p3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _load_or_compute_mel(
    audio_path: str,
    features_path: str | None,
    n_mels: int,
) -> tuple[np.ndarray, float]:
    if features_path:
        data = np.load(features_path, allow_pickle=False)
        mel = np.asarray(data["mel"], dtype=np.float32)
        stored_mels = int(data["n_mels"]) if "n_mels" in data else mel.shape[1]
        if stored_mels != n_mels:
            raise ValueError(
                f"feature n_mels={stored_mels} does not match model n_mels={n_mels}"
            )
        return mel, float(data["fps"])
    from ..audio.io import load_audio

    audio = load_audio(audio_path)
    mel, fps = log_mel_spectrogram(
        audio.samples, audio.sr, n_mels=n_mels
    )
    return mel, fps


def _tick_seconds(tick: int, bpm: float) -> float:
    return tick / TICKS_PER_BEAT * 60.0 / bpm


def _resolve_device(value: str | None) -> torch.device:
    if value and value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_difficulty(value: str) -> str:
    name = value.lower()
    if name in ("re:master", "remaster"):
        return "remaster"
    return name


def _difficulty_index(value: str) -> int:
    name = _normalize_difficulty(value)
    reverse = {v: k for k, v in DIFFICULTY_BY_INDEX.items()}
    return reverse.get(name, 5)


def _round_up_bar(tick: int) -> int:
    if tick <= 0:
        return TICKS_PER_BAR
    return ((tick + TICKS_PER_BAR - 1) // TICKS_PER_BAR) * TICKS_PER_BAR


def _repair_events(events: list[Event]) -> tuple[list[Event], list[str]]:
    """Drop duplicate/overlapping events produced by an imperfect model."""
    out: list[Event] = []
    warnings: list[str] = []
    seen: set[tuple[int, str]] = set()
    active: dict[str, int] = {}
    for event in sorted(events, key=lambda e: e.sort_key()):
        for pos in list(active):
            if active[pos] <= event.tick:
                del active[pos]
        key = (event.tick, event.position)
        if key in seen:
            warnings.append(
                f"dropped duplicate {event.kind} {event.position} at tick {event.tick}"
            )
            continue
        if event.position in active and event.kind != "hold":
            warnings.append(
                f"dropped {event.kind} {event.position} at tick {event.tick} inside active hold"
            )
            continue
        if event.kind in ("hold", "touch_hold"):
            if event.duration is None:
                warnings.append(
                    f"dropped {event.kind} {event.position} at tick {event.tick} without duration"
                )
                continue
            end = event.end_tick()
            if end <= event.tick or event.position in active:
                warnings.append(
                    f"dropped overlapping {event.kind} {event.position} at tick {event.tick}"
                )
                continue
            active[event.position] = end
        if event.kind == "slide":
            if not event.slide:
                warnings.append(f"dropped empty slide at tick {event.tick}")
                continue
            for segment in event.slide:
                if segment.shape == "-" and segment.end == event.position:
                    segment.end = "5" if event.position != "5" else "1"
                    warnings.append(
                        f"fixed straight slide at tick {event.tick} to end {segment.end}"
                    )
        seen.add(key)
        out.append(event)
    return out, warnings


def _fmt_level(level: float | None) -> str:
    if level is None:
        return ""
    if abs(level - round(level)) < 1e-9:
        return str(int(round(level)))
    return f"{level:.1f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a chart with a P3 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("-o", "--output", required=True, help="maidata.txt output")
    parser.add_argument("--difficulty", default="master")
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--timing", default=None)
    parser.add_argument("--features", default=None, help="precomputed .npz features")
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-bars", type=int, default=None)
    parser.add_argument("--stride-bars", type=int, default=None)
    parser.add_argument("--context-bars", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--artist", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = infer(
        audio_path=args.audio,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        difficulty=args.difficulty,
        level=args.level,
        timing_path=args.timing,
        features_path=args.features,
        device_name=args.device,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        context_bars=args.context_bars,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        title=args.title,
        artist=args.artist,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
