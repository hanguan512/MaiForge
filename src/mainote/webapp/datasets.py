"""Dataset management: import one song per zip archive.

Each uploaded ``.zip`` is treated as one song and must contain an audio file
(preferably ``track.*``) plus a ``maidata`` file, optionally nested in a single
top-level folder.  Files are copied into ``data/raw/<song_id>/`` and then run
through the regular :func:`mainote.dataset.ingest.ingest_dataset` pipeline into
the first configured browse root (``data/processed`` by default).

The archive handling itself lives in :mod:`mainote.dataset.zipimport` so the
online catalogue downloader can reuse the same safety checks.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..dataset.zipimport import (  # noqa: F401 - MAX_ARCHIVE_BYTES is re-exported
    MAX_ARCHIVE_BYTES,
    import_zip_archive,
)
from ..maidata.stats import DIFFICULTY_BY_INDEX
from .catalog import BACKGROUND_NAMES
from .settings import WebSettings
from .training import training_catalog


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class DatasetImporter:
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings
        self.raw_root = settings.repo_root / "data" / "raw"
        # Import into the first browse root so the result is immediately visible.
        self.processed_root = settings.data_roots[0]
        self._lock = threading.Lock()

    # -- listing ---------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        raw_songs = self._song_dirs(self.raw_root)
        readiness = {
            song["id"]: song
            for song in training_catalog(
                self.settings, self.settings.rel(self.processed_root)
            )["songs"]
        }
        songs = []
        for song_dir in self._song_dirs(self.processed_root):
            meta = _read_json(song_dir / "meta.json", {}) or {}
            song_id = str(meta.get("song_id") or song_dir.name)
            ready = readiness.get(song_dir.name) or readiness.get(song_id, {})
            charts = []
            for chart in meta.get("charts") or []:
                if not isinstance(chart, dict):
                    continue
                index = int(chart.get("index", 0) or 0)
                charts.append(
                    {
                        "index": index,
                        "difficulty": DIFFICULTY_BY_INDEX.get(index, "unknown"),
                        "level": str(chart.get("level") or ""),
                        "notes": int(chart.get("notes_total") or 0),
                    }
                )
            try:
                updated = (song_dir / "meta.json").stat().st_mtime
            except OSError:
                updated = None
            songs.append(
                {
                    "id": song_dir.name,
                    "title": str(meta.get("title") or song_dir.name),
                    "artist": str(meta.get("artist") or ""),
                    "has_audio": bool(meta.get("audio")),
                    "cover_url": self._cover_url(song_dir, meta),
                    "chart_count": len(charts),
                    "charts": charts,
                    "updated_at": updated,
                    "sample_ready": bool(ready.get("sample_ready")),
                    "features_ready": bool(ready.get("features_ready")),
                }
            )
        return {
            "raw_root": self.settings.rel(self.raw_root),
            "processed_root": self.settings.rel(self.processed_root),
            "raw_count": len(raw_songs),
            "processed_count": len(songs),
            "songs": songs,
        }

    def _cover_url(self, song_dir: Path, meta: dict[str, Any]) -> str | None:
        from urllib.parse import urlencode

        directories = [song_dir]
        raw_dir = meta.get("raw_dir")
        if raw_dir:
            directories.append(Path(str(raw_dir)).expanduser())
        has_cover = any(
            (directory / name).is_file()
            for directory in directories
            if directory.is_dir()
            for name in BACKGROUND_NAMES
        )
        if not has_cover:
            return None
        query = urlencode(
            {
                "root": self.settings.rel(self.processed_root),
                "song": song_dir.name,
            }
        )
        return f"/api/majdataview/bg?{query}"

    def _song_dirs(self, root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        return [
            child
            for child in sorted(root.iterdir())
            if child.is_dir()
            and (
                (child / "meta.json").is_file()
                or (child / "maidata.txt").is_file()
                or (child / "maidata").is_file()
            )
        ]

    # -- import ----------------------------------------------------------
    def import_zip(
        self,
        source: Path,
        *,
        filename: str | None = None,
        align: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            result = import_zip_archive(
                source,
                raw_root=self.raw_root,
                processed_root=self.processed_root,
                filename=filename,
                align=align,
                staging_root=self._staging_root(),
            )
        if result.get("raw"):
            result["raw"] = self.settings.rel(result["raw"])
        if result.get("processed"):
            result["processed"] = self.settings.rel(result["processed"])
        return result

    def _staging_root(self) -> Path:
        path = self.settings.upload_dir / "datasets"
        path.mkdir(parents=True, exist_ok=True)
        return path
