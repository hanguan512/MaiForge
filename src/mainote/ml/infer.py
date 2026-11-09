"""Run a trained P3 checkpoint on an audio file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..audio.features import BPM_LOW_CONFIDENCE
from ..audio.timing import TimingInfo
from ..generate.rule import DEFAULT_LEVELS
from ..maidata.compiler import compile_maidata
from ..maidata.model import (
    MAX_OUTER_SIMULTANEOUS,
    OUTER_KINDS,
    TICKS_PER_BAR,
    BpmChange,
    Chart,
    DivisionChange,
    Event,
)
from ..maidata.stats import DIFFICULTY_BY_INDEX, chart_stats
from ..maidata.validate import validate_chart
from .decode import constrained_decode
from .features import log_mel_spectrogram
from .model import ChartModel, ChartModelConfig, build_model
from .tokenizer import ChartVocabulary, EventTokenizer

# Local dataset priors: 9.1% of events across all charts and 13.6% across
# master charts are slides (worst single chart 43.5%).  A generated chart far
# above that is a checkpoint/decoding failure, not a style choice.
SLIDE_RATIO_WARNING = 0.30


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
        len(vocab.tokens),
        model_cfg,
        pad_id=vocab.pad_id,
        vocab_tokens=vocab.tokens,
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        raise RuntimeError(
            "checkpoint is not compatible with this architecture. "
            "Architecture v2 (embed_scale/final_norm/audio_stride) requires "
            "retraining; v1 checkpoints cannot be loaded."
        ) from exc
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
    temperature: float = 0.8,
    top_k: int | None = 20,
    seed: int = 0,
    bpm: float | None = None,
    first: float | None = None,
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
    if bpm is not None:
        timing.retime(float(bpm))
    if first is not None:
        timing.first = float(first)
    if timing.bpm <= 0:
        raise ValueError("timing BPM must be positive")

    # Optional manual tempo map.  Training windows were cut with the chart's
    # tempo map; using a constant BPM at inference drifts on BPM-change songs.
    bpm_changes: list[BpmChange] = []
    for change in timing.bpm_changes:
        tick = int(change.get("tick", 0))
        value = float(change.get("bpm", 0.0))
        if value <= 0:
            raise ValueError(f"tempo map BPM must be positive at tick {tick}")
        bpm_changes.append(BpmChange(tick, value))
    initial_bpm = timing.bpm
    for change in bpm_changes:
        if change.tick <= 0:
            initial_bpm = change.bpm
    tempo = Chart(
        index=0,
        first=timing.first,
        initial_bpm=initial_bpm,
        bpm_changes=bpm_changes,
    )

    mel, fps = _load_or_compute_mel(
        audio_path, features_path, model.config.n_mels
    )
    difficulty_index = _difficulty_index(difficulty)
    level_value = level
    if level_value is None:
        level_value = DEFAULT_LEVELS.get(_normalize_difficulty(difficulty), 12.0)

    first_onset = min(timing.onset_times) if timing.onset_times else timing.first
    last_onset = max(timing.onset_times) if timing.onset_times else timing.duration
    first_tick = tempo.ticks_at_seconds(max(0.0, first_onset - timing.first))
    last_tick = tempo.ticks_at_seconds(max(0.0, last_onset - timing.first))
    start_bar = max(0, first_tick // TICKS_PER_BAR)
    end_bar = max(
        start_bar + window_bars,
        int(math.ceil((last_tick + TICKS_PER_BAR) / TICKS_PER_BAR)),
    )
    max_tick = tempo.ticks_at_seconds(max(0.0, timing.duration - timing.first))
    max_bar = max(1, math.ceil(max_tick / TICKS_PER_BAR))

    events: list[Event] = []
    warnings: list[str] = []
    if (
        bpm is None
        and timing.bpm_candidates
        and timing.bpm_confidence < BPM_LOW_CONFIDENCE
    ):
        preview = ", ".join(
            f"{candidate['bpm']:.1f}"
            for candidate in timing.bpm_candidates[:3]
        )
        warnings.append(
            f"low BPM confidence {timing.bpm_confidence:.2f} "
            f"(candidates: {preview}); pass --bpm if the beat grid is wrong"
        )
    covered_until_tick = 0
    for start in range(start_bar, end_bar, stride_bars):
        if start >= max_bar:
            break
        win_end_bar = min(start + window_bars, max_bar)
        start_tick = start * TICKS_PER_BAR
        end_tick = win_end_bar * TICKS_PER_BAR
        # Skip windows whose full range was already covered by earlier ones.
        # The actual coverage is determined after decoding, not by the nominal
        # window length: a model may emit EOS before reaching ``end_tick``.
        if covered_until_tick >= end_tick:
            continue
        context_start_tick = max(0, start_tick - context_bars * TICKS_PER_BAR)
        context_end_tick = min(max_tick, end_tick + context_bars * TICKS_PER_BAR)
        a = max(
            0,
            int(
                math.floor(
                    (timing.first + tempo.seconds_at_tick(context_start_tick))
                    * fps
                )
            ),
        )
        b = min(
            len(mel),
            int(
                math.ceil(
                    (timing.first + tempo.seconds_at_tick(context_end_tick))
                    * fps
                )
            ),
        )
        if b <= a:
            continue
        window = mel[a:b]
        if window.shape[0] > model.config.max_audio_frames:
            warnings.append(
                f"window {start}: {window.shape[0]} mel frames exceed "
                f"max_audio_frames={model.config.max_audio_frames}; "
                "the encoder will truncate the tail"
            )
        mel_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)
        mel_mask = torch.zeros((1, window.shape[0]), dtype=torch.bool, device=device)
        try:
            ids, parsed = constrained_decode(
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
        # Count BAR advances in the generated sequence.  If the model emitted
        # EOS at bar 5 of a 16-bar window, only those 6 bars are safe to merge;
        # the next overlapping window must be allowed to fill the remainder.
        bar_count = sum(
            1 for token_id in ids if vocab.id_to_token[token_id] == "BAR"
        )
        generated_bars = bar_count + 1 if (bar_count or parsed.events) else 0
        generated_bars = max(0, min(win_end_bar - start, generated_bars))
        generated_end_tick = start_tick + generated_bars * TICKS_PER_BAR
        if generated_end_tick < end_tick:
            warnings.append(
                f"window {start}: generated {generated_bars}/{win_end_bar - start} bars"
            )
        keep_start_tick = max(start_tick, covered_until_tick)
        for event in parsed.events:
            absolute_tick = event.tick + start_tick
            if keep_start_tick <= absolute_tick < generated_end_tick:
                event.tick = absolute_tick
                events.append(event)
        covered_until_tick = max(covered_until_tick, generated_end_tick)
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
        initial_bpm=initial_bpm,
        events=events,
        bpm_changes=[BpmChange(change.tick, change.bpm) for change in bpm_changes],
        division_changes=[DivisionChange(0, 48)],
        metadata={"lv": _fmt_level(level_value), "des": "mainote-p3"},
        end_tick=_round_up_bar(max(e.end_tick() for e in events)),
    )
    stats, type_counts, type_ratios, type_warnings = _type_distribution(chart)
    warnings.extend(type_warnings)

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
        chart, title=title, artist=artist, wholebpm=initial_bpm
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
        "notes_per_second": stats["notes_per_second"],
        "type_counts": type_counts,
        "type_ratios": type_ratios,
        "difficulty": _normalize_difficulty(difficulty),
        "level": _fmt_level(level_value),
        "bpm": initial_bpm,
        "bpm_confidence": timing.bpm_confidence,
        "bpm_candidates": timing.bpm_candidates,
        "bpm_changes": timing.bpm_changes,
        "temperature": temperature,
        "top_k": top_k,
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
        with np.load(features_path, allow_pickle=False) as data:
            mel = np.asarray(data["mel"], dtype=np.float32)
            stored_mels = int(data["n_mels"]) if "n_mels" in data else mel.shape[1]
            fps = float(data["fps"])
        if stored_mels != n_mels:
            raise ValueError(
                f"feature n_mels={stored_mels} does not match model n_mels={n_mels}"
            )
        return mel, fps
    from ..audio.io import load_audio

    audio = load_audio(audio_path)
    mel, fps = log_mel_spectrogram(
        audio.samples, audio.sr, n_mels=n_mels
    )
    return mel, fps


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


def _type_distribution(
    chart: Chart,
) -> tuple[dict[str, Any], dict[str, int], dict[str, float], list[str]]:
    """Event-type counts/ratios plus a collapse warning.

    The warning keeps a slide-heavy checkpoint from looking successful just
    because the chart validates and renders.  A healthy chart rarely exceeds
    ~15% slides; the local dataset master prior is 13.6%.
    """
    stats = chart_stats(chart)
    type_counts = {
        kind: int(count) for kind, count in stats["counts"].items() if count
    }
    type_ratios = {
        kind: round(count / max(1, chart.note_count), 4)
        for kind, count in type_counts.items()
    }
    warnings: list[str] = []
    slide_ratio = type_ratios.get("slide", 0.0)
    if chart.note_count >= 50 and slide_ratio > SLIDE_RATIO_WARNING:
        warnings.append(
            f"slide share {slide_ratio:.1%} is far above typical charts "
            f"(about 9-14%); the checkpoint may be type-collapsed. Try "
            "another checkpoint or --temperature 0.6."
        )
    return stats, type_counts, type_ratios, warnings


def _repair_events(events: list[Event]) -> tuple[list[Event], list[str]]:
    """Drop duplicate/overlapping/impossible events from an imperfect model."""
    out: list[Event] = []
    warnings: list[str] = []
    seen: set[tuple[int, str]] = set()
    active: dict[str, int] = {}
    outer_by_tick: dict[int, int] = {}
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
        # Safety net for checkpoints/decoders that do not enforce the
        # grammar's outer-button chord cap.  Touch sensors are excluded: one
        # swipe can legitimately cover many sensors at the same tick.
        if event.kind in OUTER_KINDS:
            count = outer_by_tick.get(event.tick, 0)
            if count >= MAX_OUTER_SIMULTANEOUS:
                warnings.append(
                    f"dropped {event.kind} {event.position} at tick {event.tick}: "
                    f"more than {MAX_OUTER_SIMULTANEOUS} outer notes at one tick"
                )
                continue
            outer_by_tick[event.tick] = count + 1
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
            previous_end = event.position
            for index, segment in enumerate(event.slide):
                segment_start = (
                    event.position
                    if event.chain_type == "same_start"
                    or (
                        event.chain_type == "mixed"
                        and (
                            index == 0
                            or segment.branch != event.slide[index - 1].branch
                        )
                    )
                    else previous_end
                )
                if segment.shape == "-" and segment.end == segment_start:
                    segment.end = "5" if segment_start != "5" else "1"
                    warnings.append(
                        f"fixed straight slide at tick {event.tick} to end {segment.end}"
                    )
                previous_end = segment.end
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
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--first", type=float, default=None)
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
        bpm=args.bpm,
        first=args.first,
        title=args.title,
        artist=args.artist,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
