"""Dataset ingestion: raw song folders -> processed metadata + IR."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..audio.io import find_audio_file, find_maidata_file, write_json
from ..maidata.parser import parse_file
from ..maidata.stats import chart_stats
from ..maidata.validate import validate_maidata


@dataclass
class IngestResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def ingest_dataset(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    align: bool = False,
    sr: int = 22050,
    log: Callable[[str], None] | None = None,
) -> IngestResult:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = IngestResult()
    song_dirs = _find_song_dirs(raw_dir)
    if not song_dirs:
        result.warnings.append(f"no song folders found in {raw_dir}")
        return result

    used_ids: set[str] = set()
    for song_dir in song_dirs:
        song_id = _unique_id(_sanitize_id(song_dir.name), used_ids)
        used_ids.add(song_id)
        if log:
            log(f"[{song_id}] {song_dir}")
        record, warnings, errors = _ingest_song(
            song_dir, song_id, align=align, sr=sr
        )
        result.warnings.extend(warnings)
        result.errors.extend(errors)
        if record is None:
            continue
        record["warnings"] = warnings
        record["errors"] = errors
        song_out = out_dir / song_id
        ir = record.pop("ir", None)
        result.records.append(record)
        write_json(song_out / "meta.json", record)
        if ir is not None:
            write_json(song_out / "maidata.ir.json", ir)
    summary = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "song_count": len(result.records),
        "audio_coverage": round(
            sum(1 for r in result.records if r.get("has_audio"))
            / max(1, len(result.records)),
            3,
        ),
        "warnings": result.warnings,
        "errors": result.errors,
    }
    write_json(out_dir / "ingest_summary.json", summary)
    return result


def _find_song_dirs(raw_dir: Path) -> list[Path]:
    if find_maidata_file(raw_dir) or find_audio_file(raw_dir):
        return [raw_dir]
    return sorted(
        p
        for p in raw_dir.iterdir()
        if p.is_dir() and (find_maidata_file(p) or find_audio_file(p))
    )


def _ingest_song(
    song_dir: Path,
    song_id: str,
    *,
    align: bool,
    sr: int,
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    maidata_path = find_maidata_file(song_dir)
    audio_path = find_audio_file(song_dir)

    if maidata_path is None:
        return None, warnings, [f"{song_id}: no maidata file"]

    try:
        maidata = parse_file(maidata_path)
    except Exception as exc:
        return None, warnings, [f"{song_id}: failed to parse maidata: {exc}"]

    validation = validate_maidata(maidata)
    warnings.extend(f"{song_id}: {w}" for w in validation.warnings)
    errors.extend(f"{song_id}: {e}" for e in validation.errors)

    timing = None
    if audio_path is not None and align:
        try:
            from ..audio.features import analyze_audio

            timing = analyze_audio(str(audio_path), sr=sr)
        except Exception as exc:
            warnings.append(f"{song_id}: alignment analysis failed: {exc}")

    charts: list[dict[str, Any]] = []
    for chart in sorted(maidata.charts, key=lambda c: c.index):
        stats = chart_stats(chart)
        alignment = None
        if timing is not None:
            from .align import score_chart_against_timing

            alignment = score_chart_against_timing(timing, chart)
            if alignment["confidence"] < 0.5:
                warnings.append(
                    f"{song_id}: inote_{chart.index} low alignment confidence "
                    f"{alignment['confidence']} (offset {alignment['best_offset_seconds']}s)"
                )
        charts.append({**stats, "alignment": alignment})

    meta = maidata.metadata
    tags = timing.tags if timing is not None else {}
    title = meta.get("title") or tags.get("title") or song_id
    artist = meta.get("artist") or tags.get("artist") or ""
    record: dict[str, Any] = {
        "song_id": song_id,
        "raw_dir": str(song_dir.resolve()),
        "audio": str(audio_path.resolve()) if audio_path else None,
        "maidata": str(maidata_path.resolve()),
        "has_audio": audio_path is not None,
        "title": title,
        "artist": artist,
        "metadata": meta,
        "chart_count": len(charts),
        "charts": charts,
        "timing": timing.to_dict() if timing is not None else None,
        "ir": maidata.to_dict(),
    }
    return record, warnings, errors


def _sanitize_id(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE).strip("._")
    return cleaned or "song"


def _unique_id(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        return candidate
    n = 2
    while f"{candidate}_{n}" in used:
        n += 1
    return f"{candidate}_{n}"
