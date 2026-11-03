"""Option scanners and read-only summaries used by the web UI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..audio.io import AUDIO_EXTENSIONS
from .catalog import Catalog
from .jobs import JobManager
from .settings import WebSettings

_GPU_CACHE: dict[str, Any] = {"at": 0.0, "value": None}


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _read_metrics(path: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def gpu_status() -> dict[str, Any] | None:
    now = time.time()
    if now - float(_GPU_CACHE.get("at") or 0) < 2.0:
        return _GPU_CACHE.get("value")
    value = None
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            first = (result.stdout or "").strip().splitlines()
            if result.returncode == 0 and first:
                parts = [part.strip() for part in first[0].split(",")]
                value = {
                    "name": parts[0],
                    "used_mb": int(float(parts[1])),
                    "total_mb": int(float(parts[2])),
                    "util": int(float(parts[3])),
                    "temp": int(float(parts[4])),
                }
        except Exception:  # noqa: BLE001 - status must never break the UI
            value = None
    _GPU_CACHE.update({"at": now, "value": value})
    return value


def list_configs(settings: WebSettings) -> list[dict[str, str]]:
    items = []
    if settings.config_root.is_dir():
        for path in sorted(settings.config_root.glob("*.yaml")) + sorted(
            settings.config_root.glob("*.yml")
        ):
            items.append({"value": settings.rel(path), "label": path.name})
    return items


def _iter_files(root: Path, pattern: str, *, limit: int = 500):
    if not root.is_dir():
        return
    count = 0
    for path in root.rglob(pattern):
        if any(part in {".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        if path.is_file():
            yield path
            count += 1
            if count >= limit:
                return


def list_checkpoints(settings: WebSettings) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in (settings.runs_root, settings.out_root):
        for path in _iter_files(root, "*.pt"):
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            seen.add(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            name = path.name
            rank = 0 if name == "best.pt" else 1 if name == "last.pt" else 2
            items.append(
                {
                    "value": settings.rel(path),
                    "label": f"{settings.rel(path)} · {stat.st_size / 1024 / 1024:.1f} MB",
                    "name": name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "rank": rank,
                    "recommended": name == "best.pt",
                }
            )
    items.sort(key=lambda item: (item["rank"], -item["mtime"]))
    return items[:200]


def list_audio(settings: WebSettings, catalog: Catalog) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(path: Path, group: str) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(settings.repo_root)
        except (OSError, ValueError):
            return
        if not resolved.is_file() or resolved.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        key = os.path.normcase(str(resolved))
        if key in seen:
            return
        seen.add(key)
        items.append(
            {
                "value": settings.rel(resolved),
                "label": settings.rel(resolved),
                "group": group,
                "name": resolved.name,
            }
        )

    for root in settings.data_roots:
        for song_dir in sorted(root.iterdir()) if root.is_dir() else []:
            if not song_dir.is_dir():
                continue
            meta = _read_json(song_dir / "meta.json", {}) or {}
            raw = meta.get("audio")
            if raw:
                path = Path(str(raw)).expanduser()
                if not path.is_absolute():
                    path = song_dir / path
                add(path, "processed")
            raw_dir = meta.get("raw_dir")
            if raw_dir:
                base = Path(str(raw_dir)).expanduser()
                if not base.is_absolute():
                    base = song_dir / base
                if base.is_dir():
                    for extension in AUDIO_EXTENSIONS:
                        for path in sorted(base.glob(f"*{extension}")):
                            add(path, "raw")
    for path in _iter_files(settings.out_root, "*.mp3", limit=100):
        add(path, "out")
    for extension in AUDIO_EXTENSIONS:
        for path in _iter_files(settings.out_root, f"*{extension}", limit=200):
            add(path, "out")
    for path in _iter_files(settings.repo_root / "dataset", "track.*", limit=50):
        add(path, "dataset")
    return items[:300]


def list_timing(settings: WebSettings) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in _iter_files(settings.out_root, "*timing*.json", limit=150):
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append({"value": settings.rel(path), "label": settings.rel(path), "mtime": stat.st_mtime})
    items.sort(key=lambda item: -item["mtime"])
    return items[:100]


def list_features(settings: WebSettings) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in _iter_files(settings.repo_root / "data" / "features", "*.npz", limit=200):
        items.append({"value": settings.rel(path), "label": settings.rel(path)})
    for path in _iter_files(settings.out_root, "*.npz", limit=100):
        items.append({"value": settings.rel(path), "label": settings.rel(path)})
    return items[:250]


def _run_summary(settings: WebSettings, path: Path) -> dict[str, Any] | None:
    metrics_path = path / "metrics.jsonl"
    checkpoint_files = sorted(path.glob("*.pt"))
    if not metrics_path.exists() and not checkpoint_files:
        return None
    metrics = _read_metrics(metrics_path)
    val_losses = [
        float(row["val_loss"])
        for row in metrics
        if isinstance(row.get("val_loss"), (int, float))
    ]
    note_scores = [
        float((row.get("note_eval") or {}).get("note_score"))
        for row in metrics
        if isinstance(row.get("note_eval"), dict)
        and isinstance(row["note_eval"].get("note_score"), (int, float))
    ]
    last_epoch = max(
        (int(row["epoch"]) for row in metrics if isinstance(row.get("epoch"), int)),
        default=None,
    )
    last_step = max(
        (int(row["step"]) for row in metrics if isinstance(row.get("step"), int)),
        default=None,
    )
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "name": settings.rel(path),
        "path": settings.rel(path),
        "best_val_loss": min(val_losses) if val_losses else None,
        "best_note_score": max(note_scores) if note_scores else None,
        "last_epoch": last_epoch,
        "last_step": last_step,
        "checkpoint_count": len(checkpoint_files),
        "checkpoints": sorted(settings.rel(item) for item in checkpoint_files),
        "mtime": mtime,
        "has_metrics": metrics_path.exists(),
    }


def list_runs(settings: WebSettings) -> list[dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    candidates: list[Path] = []
    if settings.runs_root.is_dir():
        candidates.extend(
            path for path in sorted(settings.runs_root.iterdir()) if path.is_dir()
        )
    if settings.out_root.is_dir():
        for current, dirs, files in os.walk(settings.out_root):
            depth = len(Path(current).relative_to(settings.out_root).parts)
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [
                name
                for name in dirs
                if name not in {".venv", "node_modules", "__pycache__", "uploads"}
            ]
            if "metrics.jsonl" in files or any(name.endswith(".pt") for name in files):
                candidates.append(Path(current))
    for path in candidates:
        try:
            summary = _run_summary(settings, path)
        except OSError:
            summary = None
        if summary:
            key = summary["path"]
            items[key] = summary
    ordered = sorted(items.values(), key=lambda item: -item["mtime"])
    return ordered[:60]


def run_detail(settings: WebSettings, path_value: str) -> dict[str, Any]:
    path = settings.resolve_under(
        path_value, [settings.runs_root, settings.out_root], label="run", must_exist=True
    )
    if not path.is_dir():
        raise ValueError(f"run 不存在: {path_value}")
    summary = _run_summary(settings, path) or {}
    files = []
    for item in sorted(path.iterdir()):
        if not item.is_file():
            continue
        try:
            stat = item.stat()
        except OSError:
            continue
        files.append(
            {
                "name": item.name,
                "path": settings.rel(item),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return {
        **summary,
        "metrics": _read_metrics(path / "metrics.jsonl"),
        "files": files,
        "config": _read_json(path / "config.json", None),
    }


def overview(settings: WebSettings, catalog: Catalog, manager: JobManager) -> dict[str, Any]:
    song_count = 0
    chart_count = 0
    roots = catalog.roots()
    for root in roots:
        if not root["exists"]:
            continue
        for song in catalog.songs(root["value"]):
            song_count += 1
            chart_count += int(song.get("chart_count") or 0)
    return {
        "songs": song_count,
        "charts": chart_count,
        "roots": roots,
        "runs": len(list_runs(settings)),
        "checkpoints": len(list_checkpoints(settings)),
        "configs": len(list_configs(settings)),
        "gpu": gpu_status(),
        "active_job": manager.snapshot(include_log=False),
    }
