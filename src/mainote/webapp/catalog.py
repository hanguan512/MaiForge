"""Dataset browsing: songs, charts, audio and inference-output inspection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..audio.io import AUDIO_EXTENSIONS
from ..maidata.compiler import compile_maidata
from ..maidata.model import (
    TICKS_PER_BAR,
    Chart,
    Maidata,
)
from ..maidata.parser import parse_file
from ..maidata.stats import DIFFICULTY_BY_INDEX, chart_stats
from ..maidata.validate import validate_chart
from .settings import WebSettings


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _audio_in_dir(directory: Path) -> Path | None:
    for extension in AUDIO_EXTENSIONS:
        candidates = sorted(directory.glob(f"*{extension}"))
        if candidates:
            return candidates[0]
    return None


# The MajdataView WebGL build's difficulty dropdown defaults to index 0
# (EASY), so the embedded preview always compiles the selected chart into the
# first difficulty slot.  The surrounding UI keeps showing the real difficulty.
VIEWER_DIFFICULTY_INDEX = 1

BACKGROUND_NAMES = (
    "bg.jpg",
    "bg.jpeg",
    "bg.png",
    "Cover.jpg",
    "Cover.jpeg",
    "Cover.png",
    "cover.jpg",
    "cover.jpeg",
    "cover.png",
)


class Catalog:
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings

    # -- roots -----------------------------------------------------------
    def roots(self) -> list[dict[str, Any]]:
        items = []
        for root in self.settings.data_roots:
            songs = self._song_dirs(root)
            items.append(
                {
                    "value": self.settings.rel(root),
                    "label": self._root_label(root),
                    "exists": root.is_dir(),
                    "song_count": len(songs),
                }
            )
        return items

    def _root_label(self, root: Path) -> str:
        relative = self.settings.rel(root)
        if relative.startswith("data/"):
            return "主数据集"
        if "demo" in relative:
            return "演示数据"
        if "audio-fm" in relative:
            return "音频测试数据"
        return root.name

    def _song_dirs(self, root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        return [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / "meta.json").is_file()
        ]

    # -- songs -----------------------------------------------------------
    def songs(self, root_value: str | None = None) -> list[dict[str, Any]]:
        root = self.settings.data_root_by_value(root_value)
        items = []
        for song_dir in self._song_dirs(root):
            meta = _read_json(song_dir / "meta.json", {}) or {}
            charts = list(meta.get("charts") or [])
            audio = self._audio_path(song_dir, meta)
            items.append(
                {
                    "root": self.settings.rel(root),
                    "id": song_dir.name,
                    "title": str(meta.get("title") or song_dir.name),
                    "artist": str(meta.get("artist") or ""),
                    "bpm": (meta.get("timing") or {}).get("bpm"),
                    "has_audio": bool(audio),
                    "has_ir": (song_dir / "maidata.ir.json").is_file(),
                    "cover_url": self._cover_url(root, song_dir.name, meta),
                    "chart_count": len(charts),
                    "charts": [self._chart_brief(chart) for chart in charts],
                    "warnings": list(meta.get("warnings") or []),
                    "errors": list(meta.get("errors") or []),
                }
            )
        return items

    def song(self, root_value: str | None, song_id: str) -> dict[str, Any]:
        root = self.settings.data_root_by_value(root_value)
        song_dir = self._song_dir(root, song_id)
        meta = _read_json(song_dir / "meta.json", {}) or {}
        audio = self._audio_path(song_dir, meta)
        ir_path = song_dir / "maidata.ir.json"
        maidata = None
        if ir_path.is_file():
            try:
                maidata = Maidata.from_dict(_read_json(ir_path, {}))
            except (OSError, ValueError, TypeError, KeyError):
                maidata = None
        charts = []
        if maidata is not None:
            for chart in maidata.charts:
                charts.append(self._chart_row(chart))
        else:
            charts = [self._chart_brief(chart) for chart in meta.get("charts") or []]
        return {
            "root": self.settings.rel(root),
            "id": song_id,
            "title": str(meta.get("title") or song_id),
            "artist": str(meta.get("artist") or ""),
            "metadata": meta.get("metadata") or {},
            "has_audio": bool(audio),
            "audio_url": self._audio_url(root, song_id) if audio else None,
            "cover_url": self._cover_url(root, song_id, meta),
            "timing": meta.get("timing"),
            "alignment": self._alignment(meta),
            "charts": charts,
            "warnings": list(meta.get("warnings") or []),
            "errors": list(meta.get("errors") or []),
            "ir_available": maidata is not None,
        }

    def chart(
        self,
        root_value: str | None,
        song_id: str,
        index: int,
    ) -> dict[str, Any]:
        root = self.settings.data_root_by_value(root_value)
        song_dir = self._song_dir(root, song_id)
        meta = _read_json(song_dir / "meta.json", {}) or {}
        ir_path = song_dir / "maidata.ir.json"
        if not ir_path.is_file():
            raise ValueError(f"谱面 IR 不存在: {self.settings.rel(ir_path)}")
        maidata = Maidata.from_dict(_read_json(ir_path, {}))
        chart = maidata.chart(index)
        if chart is None:
            available = ", ".join(str(item.index) for item in maidata.charts) or "无"
            raise ValueError(f"难度 {index} 不存在(可用: {available})")
        audio = self._audio_path(song_dir, meta)
        text = compile_maidata(self._single_chart_maidata(maidata, chart))
        return self._chart_payload(
            chart,
            title=str(meta.get("title") or song_id),
            artist=str(meta.get("artist") or ""),
            audio_url=self._audio_url(root, song_id) if audio else None,
            cover_url=self._cover_url(root, song_id, meta),
            source={
                "kind": "song",
                "root": self.settings.rel(root),
                "song_id": song_id,
            },
            maidata_text=text,
            charts=self._viewer_chart_options(maidata.charts),
            audio_duration_seconds=self._audio_duration(meta),
        )

    def chart_file(self, path_value: str, index: int | None = None) -> dict[str, Any]:
        path = self.settings.resolve_under(
            path_value, [self.settings.repo_root], label="谱面", must_exist=True
        )
        if not path.is_file():
            raise ValueError(f"不是文件: {path}")
        maidata = parse_file(path)
        if not maidata.charts:
            raise ValueError("文件里没有难度块")
        chart = maidata.chart(index) if index is not None else maidata.charts[0]
        if chart is None:
            raise ValueError(f"难度 {index} 不存在")
        summary = _read_json(path.parent / "p3_summary.json", {}) or {}
        audio = summary.get("audio") if isinstance(summary, dict) else None
        audio_url = None
        if audio and Path(str(audio)).is_file():
            try:
                audio_path = self.settings.resolve_under(
                    str(audio), [self.settings.repo_root], label="音频"
                )
            except ValueError:
                audio_path = None
            if audio_path is not None:
                audio_url = f"/api/audio?path={audio_path.as_posix()}"
        return self._chart_payload(
            chart,
            title=maidata.get("title") or path.parent.name,
            artist=maidata.get("artist") or "",
            audio_url=audio_url,
            cover_url=None,
            source={"kind": "file", "path": self.settings.rel(path)},
            maidata_text=path.read_text(encoding="utf-8", errors="replace"),
            charts=self._viewer_chart_options(maidata.charts),
            audio_duration_seconds=None,
        )

    # -- MajdataView preview --------------------------------------------
    def viewer_maidata(
        self,
        root_value: str | None,
        song_id: str,
        index: int,
        seek: float = 0.0,
    ) -> str:
        """Compile one chart into the slot the MajdataView build selects.

        ``seek`` rebuilds the chart for playback from that audio time: notes
        that start earlier are dropped and ``first`` is shifted by ``-seek``
        so the WebGL player can start its trimmed track at 0.
        """
        root = self.settings.data_root_by_value(root_value)
        song_dir = self._song_dir(root, song_id)
        ir_path = song_dir / "maidata.ir.json"
        if not ir_path.is_file():
            raise ValueError(f"谱面 IR 不存在: {self.settings.rel(ir_path)}")
        maidata = Maidata.from_dict(_read_json(ir_path, {}))
        chart = maidata.chart(index)
        if chart is None:
            available = ", ".join(str(item.index) for item in maidata.charts) or "无"
            raise ValueError(f"难度 {index} 不存在(可用: {available})")
        return self._viewer_text(maidata, chart, seek)

    def viewer_maidata_file(
        self, path_value: str, index: int | None, seek: float = 0.0
    ) -> str:
        path = self.settings.resolve_under(
            path_value, [self.settings.repo_root], label="谱面", must_exist=True
        )
        if not path.is_file():
            raise ValueError(f"不是文件: {path}")
        maidata = parse_file(path)
        if not maidata.charts:
            raise ValueError("文件里没有难度块")
        chart = maidata.chart(index) if index is not None else maidata.charts[0]
        if chart is None:
            raise ValueError(f"难度 {index} 不存在")
        return self._viewer_text(maidata, chart, seek)

    def _viewer_text(
        self, maidata: Maidata, chart: Chart, seek: float = 0.0
    ) -> str:
        seek = max(0.0, float(seek or 0.0))
        # The build's SimaiProcess only understands a global ``&first`` and
        # ignores ``&first_N``.  Strip any inherited value and let the
        # compiler derive a single global one from the shifted chart.
        metadata_items = [
            (key, value)
            for key, value in maidata.metadata_items
            if key != "first"
        ]
        metadata = {
            key: value for key, value in chart.metadata.items() if key != "first"
        }
        preview = replace(
            self._seek_chart(chart, seek),
            index=VIEWER_DIFFICULTY_INDEX,
            metadata=metadata,
        )
        return compile_maidata(
            Maidata(
                metadata_items=metadata_items,
                charts=[preview],
                source_path=maidata.source_path,
            )
        )

    def _seek_chart(self, chart: Chart, seek: float) -> Chart:
        if seek <= 0:
            return chart
        events = [
            event
            for event in chart.events
            if chart.event_time_seconds(event) >= seek - 1e-6
        ]
        return replace(chart, first=chart.first - seek, events=events)

    def _viewer_chart_options(self, charts: list[Chart]) -> list[dict[str, Any]]:
        return [
            {
                "index": chart.index,
                "difficulty": DIFFICULTY_BY_INDEX.get(chart.index, "unknown"),
                "level": chart.level,
            }
            for chart in charts
        ]

    def _audio_duration(self, meta: dict[str, Any]) -> float | None:
        timing = meta.get("timing")
        if not isinstance(timing, dict):
            return None
        value = timing.get("duration")
        if isinstance(value, (int, float)) and value > 0:
            return round(float(value), 6)
        return None

    def song_audio(self, root_value: str | None, song_id: str) -> Path | None:
        """Resolve a song's audio without failing when the chart has none."""
        root = self.settings.data_root_by_value(root_value)
        song_dir = self._song_dir(root, song_id)
        meta = _read_json(song_dir / "meta.json", {}) or {}
        return self._audio_path(song_dir, meta)

    def background_file(
        self, root_value: str | None, song_id: str
    ) -> Path | None:
        root = self.settings.data_root_by_value(root_value)
        song_dir = self._song_dir(root, song_id)
        meta = _read_json(song_dir / "meta.json", {}) or {}
        return self._background_file_in(song_dir, meta)

    def _background_file_in(
        self, song_dir: Path, meta: dict[str, Any]
    ) -> Path | None:
        directories = [song_dir]
        raw_dir = meta.get("raw_dir")
        if raw_dir:
            directories.append(Path(str(raw_dir)).expanduser())
        for directory in directories:
            if not directory.is_dir():
                continue
            for name in BACKGROUND_NAMES:
                candidate = directory / name
                if candidate.is_file():
                    return candidate
        return None

    def _cover_url(
        self, root: Path, song_id: str, meta: dict[str, Any]
    ) -> str | None:
        from urllib.parse import urlencode

        song_dir = root / song_id
        if self._background_file_in(song_dir, meta) is None:
            return None
        query = urlencode({"root": self.settings.rel(root), "song": song_id})
        return f"/api/majdataview/bg?{query}"

    # -- helpers ---------------------------------------------------------
    def _song_dir(self, root: Path, song_id: str) -> Path:
        safe = Path(song_id).name
        if not safe or safe != song_id:
            raise ValueError("非法歌曲 ID")
        path = (root / safe).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("歌曲不在数据目录内") from exc
        if not path.is_dir():
            raise ValueError(f"歌曲不存在: {song_id}")
        return path

    def _audio_path(self, song_dir: Path, meta: dict[str, Any]) -> Path | None:
        candidates: list[Path] = []
        raw = meta.get("audio")
        if raw:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = song_dir / path
            candidates.append(path)
        raw_dir = meta.get("raw_dir")
        if raw_dir:
            candidates.append(Path(str(raw_dir)).expanduser())
        candidates.append(song_dir)
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
                return candidate
            if candidate.is_dir():
                found = _audio_in_dir(candidate)
                if found is not None:
                    return found
        return None

    def _audio_url(self, root: Path, song_id: str) -> str:
        from urllib.parse import urlencode

        query = urlencode({"root": self.settings.rel(root), "song": song_id})
        return f"/api/audio?{query}"

    def audio_file(self, *, path_value: str | None, root_value: str | None, song_id: str | None) -> Path:
        if root_value is not None and song_id:
            root = self.settings.data_root_by_value(root_value)
            song_dir = self._song_dir(root, song_id)
            meta = _read_json(song_dir / "meta.json", {}) or {}
            audio = self._audio_path(song_dir, meta)
            if audio is None:
                raise ValueError("该歌曲没有可用音频")
            return audio
        if not path_value:
            raise ValueError("缺少音频路径")
        path = self.settings.resolve_under(
            path_value, [self.settings.repo_root], label="音频", must_exist=True
        )
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError("不是支持的音频文件")
        return path

    def _chart_brief(self, chart: dict[str, Any]) -> dict[str, Any]:
        index = int(chart.get("index", 0) or 0)
        return {
            "index": index,
            "difficulty": DIFFICULTY_BY_INDEX.get(index, "unknown"),
            "level": str(chart.get("level") or ""),
            "designer": str(chart.get("designer") or ""),
            "notes": int(chart.get("notes_total") or 0),
            "density": chart.get("notes_per_second"),
            "duration": chart.get("duration_seconds"),
            "alignment": chart.get("alignment"),
        }

    def _chart_row(self, chart: Chart) -> dict[str, Any]:
        stats = chart_stats(chart)
        return {
            "index": chart.index,
            "difficulty": DIFFICULTY_BY_INDEX.get(chart.index, "unknown"),
            "level": chart.level,
            "designer": chart.designer,
            "notes": stats["notes_total"],
            "density": stats["notes_per_second"],
            "duration": stats["duration_seconds"],
            "counts": stats["counts"],
            "bpm_min": stats["bpm_min"],
            "bpm_max": stats["bpm_max"],
        }

    def _alignment(self, meta: dict[str, Any]) -> Any:
        charts = meta.get("charts") or []
        values = [
            (chart.get("alignment") or {}).get("confidence")
            for chart in charts
            if isinstance(chart, dict)
        ]
        values = [value for value in values if isinstance(value, (int, float))]
        return {
            "min_confidence": min(values) if values else None,
            "max_confidence": max(values) if values else None,
        }

    def _single_chart_maidata(self, maidata: Maidata, chart: Chart) -> Maidata:
        return Maidata(
            metadata_items=list(maidata.metadata_items),
            charts=[chart],
            source_path=maidata.source_path,
        )

    def _chart_payload(
        self,
        chart: Chart,
        *,
        title: str,
        artist: str,
        audio_url: str | None,
        cover_url: str | None,
        source: dict[str, Any],
        maidata_text: str,
        charts: list[dict[str, Any]],
        audio_duration_seconds: float | None,
    ) -> dict[str, Any]:
        validation = validate_chart(chart)
        stats = chart_stats(chart)
        return {
            "source": source,
            "title": title,
            "artist": artist,
            "index": chart.index,
            "difficulty": DIFFICULTY_BY_INDEX.get(chart.index, "unknown"),
            "level": chart.level,
            "designer": chart.designer,
            "initial_bpm": chart.initial_bpm,
            "first": chart.first,
            "end_tick": chart.end_tick,
            "duration_seconds": round(chart.duration_seconds, 6),
            "audio_duration_seconds": audio_duration_seconds,
            "charts": charts,
            "bpm_changes": [
                {
                    "tick": change.tick,
                    "bpm": change.bpm,
                    "seconds": round(chart.seconds_at_tick(change.tick), 6),
                }
                for change in chart.bpm_changes
            ],
            "events": [self._event(event, chart) for event in chart.events],
            "stats": stats,
            "validation": {
                "ok": validation.ok,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            },
            "audio_url": audio_url,
            "cover_url": cover_url,
            "maidata": maidata_text,
        }

    def _event(self, event: Any, chart: Chart) -> dict[str, Any]:
        start = chart.event_time_seconds(event)
        end = chart.event_end_seconds(event)
        segments = []
        if event.kind == "slide":
            bpm = chart.bpm_at(event.tick)
            wait = 60.0 / bpm if bpm > 0 else 0.0
            cursor = start + wait
            previous = event.position
            branch: int | None = None
            last_index = len(event.slide) - 1
            for index, segment in enumerate(event.slide):
                if event.chain_type == "mixed" and segment.branch != branch:
                    # A new '*' branch restarts from the shared star tap.
                    branch = segment.branch
                    cursor = start + wait
                    previous = event.position
                if event.chain_type == "path" and index == last_index:
                    duration = (
                        event.duration.approx_seconds(bpm)
                        if event.duration is not None
                        else None
                    )
                else:
                    duration = (
                        segment.duration.approx_seconds(bpm)
                        if segment.duration is not None
                        else None
                    )
                duration = float(duration or 0.0)
                segment_start = cursor
                segment_end = segment_start + duration
                segments.append(
                    {
                        "shape": segment.shape,
                        "start": previous,
                        "end": segment.end,
                        "via": segment.via,
                        "start_seconds": round(segment_start, 6),
                        "end_seconds": round(segment_end, 6),
                        "duration_seconds": round(duration, 6),
                    }
                )
                if event.chain_type in ("path", "mixed"):
                    cursor = segment_end
                    previous = segment.end
        return {
            "tick": event.tick,
            "seconds": round(start, 6),
            "end_seconds": round(end, 6),
            "kind": event.kind,
            "position": event.position,
            "flags": list(event.flags),
            "chain_type": event.chain_type,
            "segments": segments,
            "bar": round(event.tick / TICKS_PER_BAR, 3),
        }
