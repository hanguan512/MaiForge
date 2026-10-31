"""Split processed songs into one training sample per difficulty.

A song keeps a single ``maidata.ir.json`` (all difficulties share the same
audio), but this module flattens it into:

- ``manifest.jsonl``: one line per ``(song_id, inote_N)`` training sample
- ``<song_id>/inote_<N>.json``: self-contained chart sample (events + metadata)
- ``splits.json``: song-level train/val/test assignment
- ``summary.json``: counts by split/difficulty and audio/alignment coverage

Splitting is always by song: every difficulty of a song stays in the same
split, which prevents audio/chart leakage.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..maidata.model import Chart, Maidata
from ..maidata.stats import DIFFICULTY_BY_INDEX, chart_stats
from .split import split_dataset


@dataclass
class PreparedSamples:
    rows: list[dict[str, Any]] = field(default_factory=list)
    charts: dict[str, Chart] = field(default_factory=dict)
    songs: dict[str, Maidata] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def parse_level(value: str | None) -> float | None:
    """Parse ``12``, ``12.9``, ``12+``, ``12-`` into a numeric level."""
    if value is None:
        return None
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


def build_manifest(
    processed_dir: str | Path,
    *,
    splits: str | Path | None = None,
    seed: int = 0,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    require_audio: bool = False,
) -> dict[str, Any]:
    prepared = prepare_samples(
        processed_dir,
        splits=splits,
        seed=seed,
        train=train,
        val=val,
        test=test,
        require_audio=require_audio,
    )
    return {
        "rows": prepared.rows,
        "splits": prepared.splits,
        "summary": prepared.summary,
    }


def export_samples(
    processed_dir: str | Path,
    out_dir: str | Path,
    *,
    splits: str | Path | None = None,
    seed: int = 0,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    require_audio: bool = False,
) -> dict[str, Any]:
    prepared = prepare_samples(
        processed_dir,
        splits=splits,
        seed=seed,
        train=train,
        val=val,
        test=test,
        require_audio=require_audio,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in prepared.rows:
        sample_path = (
            out_dir / row["song_id"] / f"inote_{row['chart_index']}.json"
        )
        row["sample"] = str(sample_path.resolve())

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in prepared.rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in prepared.rows:
        sample_id = row["sample_id"]
        chart = prepared.charts[sample_id]
        song = prepared.songs[row["song_id"]]
        sample = {
            "format": "mainote.chart_sample",
            "version": 1,
            **row,
            "song_metadata": dict(song.metadata_items),
            "chart": chart.to_dict(),
        }
        song_dir = out_dir / row["song_id"]
        song_dir.mkdir(parents=True, exist_ok=True)
        sample_path = Path(row["sample"])
        sample_path.write_text(
            json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    (out_dir / "splits.json").write_text(
        json.dumps(prepared.splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(prepared.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return prepared.summary


def prepare_samples(
    processed_dir: str | Path,
    *,
    splits: str | Path | None = None,
    seed: int = 0,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    require_audio: bool = False,
) -> PreparedSamples:
    processed_dir = Path(processed_dir)
    songs = list(_iter_processed_songs(processed_dir))
    split_result, split_map = _resolve_splits(
        processed_dir,
        splits=splits,
        seed=seed,
        train=train,
        val=val,
        test=test,
        song_ids=[meta.get("song_id") or meta_path.parent.name for meta, _, meta_path in songs],
    )

    prepared = PreparedSamples(splits=split_result)
    difficulty_counts: Counter[str] = Counter()
    split_chart_counts: Counter[str] = Counter()
    alignment_confidences: list[float] = []
    skipped_no_audio: list[str] = []

    for meta, maidata, meta_path in songs:
        song_id = str(meta.get("song_id") or meta_path.parent.name)
        if require_audio and not meta.get("has_audio"):
            skipped_no_audio.append(song_id)
            continue
        prepared.songs[song_id] = maidata
        ir_path = meta_path.parent / "maidata.ir.json"
        stats_by_index = {
            int(item["index"]): item
            for item in meta.get("charts", [])
            if "index" in item
        }
        timing = meta.get("timing") or {}
        audio_duration = timing.get("duration")
        for chart in sorted(maidata.charts, key=lambda c: c.index):
            stats = dict(stats_by_index.get(chart.index) or chart_stats(chart))
            alignment = stats.pop("alignment", None)
            if alignment and isinstance(alignment.get("confidence"), (int, float)):
                alignment_confidences.append(float(alignment["confidence"]))
            difficulty = DIFFICULTY_BY_INDEX.get(chart.index, "unknown")
            level_num = parse_level(chart.level or stats.get("level"))
            sample_id = f"{song_id}:{chart.index}"
            alignment_confidence = (
                alignment.get("confidence") if isinstance(alignment, dict) else None
            )
            row = {
                "sample_id": sample_id,
                "song_id": song_id,
                "chart_index": chart.index,
                "difficulty": difficulty,
                "level": chart.level or stats.get("level", ""),
                "level_num": level_num,
                "charter": chart.designer or stats.get("designer", ""),
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
                "audio": meta.get("audio"),
                "has_audio": bool(meta.get("has_audio")),
                "audio_duration": audio_duration,
                "maidata": meta.get("maidata"),
                "ir": str(ir_path.resolve()) if ir_path.exists() else None,
                "first": chart.first,
                "initial_bpm": chart.initial_bpm,
                "bpm_change_count": max(0, len(chart.bpm_changes) - 1),
                "event_count": chart.note_count,
                "stats": stats,
                "alignment": alignment,
                "alignment_confidence": alignment_confidence,
                "valid": not bool(meta.get("errors")),
                "split": split_map.get(song_id, "unsplit"),
                "warnings": list(meta.get("warnings", [])) + list(chart.warnings),
                "errors": list(meta.get("errors", [])),
            }
            prepared.rows.append(row)
            prepared.charts[sample_id] = chart
            difficulty_counts[difficulty] += 1
            split_chart_counts[row["split"]] += 1

    prepared.rows.sort(key=lambda r: (r["song_id"], r["chart_index"]))
    split_song_counts = {
        name: len(split_result.get("splits", {}).get(name, []))
        for name in ("train", "val", "test")
    }
    prepared.summary = {
        "processed_dir": str(processed_dir),
        "song_count": len(prepared.songs),
        "chart_count": len(prepared.rows),
        "skipped_no_audio": skipped_no_audio,
        "require_audio": require_audio,
        "audio_coverage": round(
            sum(1 for r in prepared.rows if r["has_audio"]) / max(1, len(prepared.rows)),
            3,
        ),
        "split_chart_counts": dict(sorted(split_chart_counts.items())),
        "split_song_counts": split_song_counts,
        "unsplit_chart_count": split_chart_counts.get("unsplit", 0),
        "unsplit_song_count": sum(
            1
            for song_id in prepared.songs
            if split_map.get(song_id) == "unsplit"
        ),
        "difficulty_chart_counts": dict(sorted(difficulty_counts.items())),
        "alignment": {
            "measured": len(alignment_confidences),
            "mean_confidence": round(
                sum(alignment_confidences) / len(alignment_confidences), 4
            )
            if alignment_confidences
            else 0.0,
            "low_confidence": sum(1 for c in alignment_confidences if c < 0.5),
        },
    }
    return prepared


def _resolve_splits(
    processed_dir: Path,
    *,
    splits: str | Path | None,
    seed: int,
    train: float,
    val: float,
    test: float,
    song_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    if splits is not None:
        data = json.loads(Path(splits).read_text(encoding="utf-8"))
    else:
        data = split_dataset(
            processed_dir, train=train, val=val, test=test, seed=seed
        )
    split_map: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for song_id in data.get("splits", {}).get(name, []):
            split_map[str(song_id)] = name
    for song_id in song_ids:
        split_map.setdefault(str(song_id), "unsplit")
    return data, split_map


def _iter_processed_songs(
    processed_dir: Path,
) -> Iterable[tuple[dict[str, Any], Maidata, Path]]:
    for meta_path in sorted(processed_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        ir_path = meta_path.parent / "maidata.ir.json"
        if not ir_path.exists():
            continue
        try:
            maidata = Maidata.from_dict(
                json.loads(ir_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        yield meta, maidata, meta_path
