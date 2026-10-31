"""Local web control room for P3 training and inference.

The server only binds to localhost by default.  It never accepts arbitrary
commands: the browser can start the same CLI entry points that are documented
in ``docs/TRAINING.md``, with validated paths and fixed argument lists.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
)

from ..audio.io import AUDIO_EXTENSIONS
from ..maidata.parser import parse_file
from ..maidata.stats import DIFFICULTY_BY_INDEX, chart_stats
from ..maidata.validate import validate_maidata

WEB_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEB_DIR.parents[2]
TEMPLATE_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

RUNS_ROOT = REPO_ROOT / "runs"
OUT_ROOT = REPO_ROOT / "out"
DATA_ROOT = REPO_ROOT / "data"
CONFIG_ROOT = REPO_ROOT / "configs"
DESKTOP = Path.home() / "Desktop"

AUDIO_EXTENSIONS = set(AUDIO_EXTENSIONS)
TEXT_EXTENSIONS = {".txt", ".json", ".jsonl", ".yaml", ".yml", ".log", ".md", ".csv"}
DIFFICULTIES = {"basic", "advanced", "expert", "master", "remaster"}

_GPU_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_TITLE_CACHE: dict[str, str] = {}
_MANAGER: "JobManager | None" = None


# ---------------------------------------------------------------------------
# small helpers


def _rel(path: Path | str) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _resolve_under(
    value: str | Path,
    roots: list[Path],
    *,
    label: str = "路径",
    must_exist: bool = False,
) -> Path:
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    try:
        path = raw.resolve()
    except OSError as exc:
        raise ValueError(f"{label} 无法解析: {value}") from exc
    if must_exist and not path.exists():
        raise ValueError(f"{label} 不存在: {path}")
    for root in roots:
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        return path
    allowed = ", ".join(_rel(root) for root in roots)
    raise ValueError(f"{label} 不在允许目录内（{allowed}）: {path}")


def _optional_int(value: Any, label: str, *, minimum: int = 0) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是整数") from exc
    if number < minimum:
        raise ValueError(f"{label} 不能小于 {minimum}")
    return number


def _optional_float(value: Any, label: str, *, minimum: float = 0.0) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if number < minimum:
        raise ValueError(f"{label} 不能小于 {minimum}")
    return number


def _unique_dir(path: Path) -> Path:
    candidate = path
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{index}")
        index += 1
    return candidate


def _looks_used(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / "last.pt").exists() or (path / "metrics.jsonl").exists():
        return True
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _tail_lines(path: Path, *, max_lines: int = 240, max_bytes: int = 262144) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return []
    text = data.decode("utf-8", "replace")
    lines = text.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]
    return lines[-max_lines:]


def _read_metrics(path: Path, *, limit: int = 600) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _tail_lines(path, max_lines=limit, max_bytes=2_000_000):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _song_title(song_id: str) -> str:
    if song_id in _TITLE_CACHE:
        return _TITLE_CACHE[song_id]
    meta = _read_json(DATA_ROOT / "processed" / song_id / "meta.json", {}) or {}
    title = str(meta.get("title") or song_id)
    _TITLE_CACHE[song_id] = title
    return title


def _gpu_status() -> dict[str, Any] | None:
    now = time.time()
    if now - float(_GPU_CACHE.get("at") or 0) < 2.0:
        return _GPU_CACHE.get("value")
    if not shutil.which("nvidia-smi"):
        _GPU_CACHE.update({"at": now, "value": None})
        return None
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
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        first = (result.stdout or "").strip().splitlines()
        if result.returncode != 0 or not first:
            raise RuntimeError(result.stderr[:200])
        parts = [part.strip() for part in first[0].split(",")]
        value = {
            "name": parts[0],
            "used_mb": int(float(parts[1])),
            "total_mb": int(float(parts[2])),
            "util": int(float(parts[3])),
            "temp": int(float(parts[4])),
        }
    except Exception:  # noqa: BLE001 - status must not break the UI
        value = None
    _GPU_CACHE.update({"at": now, "value": value})
    return value


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    src = REPO_ROOT / "src"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    paths: list[str] = []
    venv = REPO_ROOT / ".venv"
    scripts = venv / ("Scripts" if os.name == "nt" else "bin")
    if scripts.exists():
        paths.append(str(scripts))
    ffmpeg_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "ffmpeg" / "bin"
    if ffmpeg_dir.exists():
        paths.append(str(ffmpeg_dir))
    if paths:
        env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    return env


# ---------------------------------------------------------------------------
# jobs


@dataclass
class Job:
    id: str
    kind: str
    status: str
    started_at: float
    log_path: Path
    output_dir: Path
    command: list[str]
    extra: dict[str, Any] = field(default_factory=dict)
    process: subprocess.Popen | None = None
    finished_at: float | None = None
    returncode: int | None = None
    stop_requested: bool = False


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job: Job | None = None

    # -- lifecycle -------------------------------------------------------
    def _active(self) -> bool:
        return self._job is not None and self._job.status in ("running", "stopping")

    def _ensure_idle(self) -> None:
        if self._active():
            raise ValueError("已有任务在运行，请先停止或等待完成")

    def start_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = _resolve_under(
            payload.get("config") or "configs/p3_4070.yaml",
            [CONFIG_ROOT],
            label="配置文件",
            must_exist=True,
        )
        if config.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("训练配置必须是 .yaml/.yml")
        device = str(payload.get("device") or "auto")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("device 必须是 auto/cuda/cpu")

        resume = payload.get("resume")
        resume_path: Path | None = None
        if resume:
            resume_path = _resolve_under(
                resume, [RUNS_ROOT], label="resume checkpoint", must_exist=True
            )

        raw_output = str(payload.get("output_dir") or "").strip()
        if resume_path is not None and not raw_output:
            output = resume_path.parent
        else:
            output = self._train_output(raw_output)
        if resume_path is None and _looks_used(output):
            output = _unique_dir(output)
        output.mkdir(parents=True, exist_ok=True)

        max_steps = _optional_int(payload.get("max_steps"), "max_steps")
        if max_steps == 0:
            max_steps = None
        num_workers = _optional_int(payload.get("num_workers"), "num_workers")
        dry_run = bool(payload.get("dry_run"))

        config_data = _read_json(config, {}) or {}
        epochs = 0
        try:
            import yaml

            parsed = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            epochs = int(parsed.get("train", {}).get("epochs", 0) or 0)
        except Exception:  # noqa: BLE001 - display only
            epochs = 0
        del config_data

        command = [
            sys.executable,
            "-m",
            "mainote",
            "ml",
            "train",
            "--config",
            str(config),
            "--device",
            device,
            "--output-dir",
            str(output),
        ]
        if resume_path is not None:
            command += ["--resume", str(resume_path)]
        if max_steps:
            command += ["--max-steps", str(max_steps)]
        if num_workers is not None:
            command += ["--num-workers", str(num_workers)]
        if dry_run:
            command += ["--dry-run"]

        return self._start_job(
            "train",
            command,
            output,
            {
                "config": _rel(config),
                "resume": _rel(resume_path) if resume_path else None,
                "epochs": epochs,
                "max_steps": max_steps,
                "dry_run": dry_run,
            },
        )

    def start_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        audio = _resolve_under(
            payload.get("audio") or "",
            [REPO_ROOT, DESKTOP],
            label="音频",
            must_exist=True,
        )
        if audio.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"不支持的音频格式: {audio.suffix}")
        checkpoint = _resolve_under(
            payload.get("checkpoint") or "",
            [RUNS_ROOT],
            label="checkpoint",
            must_exist=True,
        )
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("checkpoint 必须是 .pt 文件")

        timing: Path | None = None
        if payload.get("timing"):
            timing = _resolve_under(
                payload["timing"], [REPO_ROOT], label="timing", must_exist=True
            )
        features: Path | None = None
        if payload.get("features"):
            features = _resolve_under(
                payload["features"], [REPO_ROOT], label="features", must_exist=True
            )

        difficulty = str(payload.get("difficulty") or "master").lower()
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"未知难度: {difficulty}")
        level = _optional_float(payload.get("level"), "level", minimum=0.0)
        temperature = _optional_float(
            payload.get("temperature"), "temperature", minimum=0.0
        )
        top_k = _optional_int(payload.get("top_k"), "top_k", minimum=1)
        max_new_tokens = _optional_int(
            payload.get("max_new_tokens"), "max_new_tokens", minimum=1
        )
        seed = _optional_int(payload.get("seed"), "seed", minimum=0)
        device = str(payload.get("device") or "auto")
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("device 必须是 auto/cuda/cpu")

        output = self._infer_output(payload.get("output_dir"))
        if _looks_used(output):
            output = _unique_dir(output)
        output.mkdir(parents=True, exist_ok=True)
        maidata_path = output / "maidata.txt"

        command = [
            sys.executable,
            "-m",
            "mainote",
            "ml",
            "infer",
            "--checkpoint",
            str(checkpoint),
            "--audio",
            str(audio),
            "-o",
            str(maidata_path),
            "--difficulty",
            difficulty,
            "--device",
            device,
        ]
        if level is not None:
            command += ["--level", str(level)]
        if timing is not None:
            command += ["--timing", str(timing)]
        if features is not None:
            command += ["--features", str(features)]
        if temperature is not None and temperature > 0:
            command += ["--temperature", str(temperature)]
        if top_k is not None:
            command += ["--top-k", str(top_k)]
        if max_new_tokens is not None:
            command += ["--max-new-tokens", str(max_new_tokens)]
        if seed is not None:
            command += ["--seed", str(seed)]

        return self._start_job(
            "infer",
            command,
            output,
            {
                "audio": str(audio),
                "audio_label": audio.name,
                "checkpoint": _rel(checkpoint),
                "difficulty": difficulty,
                "level": level,
                "timing": _rel(timing) if timing else None,
                "features": _rel(features) if features else None,
                "result_maidata": str(maidata_path),
            },
        )

    def _train_output(self, raw: str) -> Path:
        text = raw.strip() or "runs/web_train"
        path = Path(text)
        if not path.is_absolute():
            if text.replace("\\", "/").startswith("runs/"):
                path = REPO_ROOT / path
            else:
                path = RUNS_ROOT / path
        path = path.resolve()
        try:
            path.relative_to(RUNS_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"训练输出目录必须在 runs/ 下: {path}") from exc
        return path

    def _infer_output(self, raw: Any) -> Path:
        text = str(raw or "").strip() or "out/web_infer"
        path = Path(text)
        if not path.is_absolute():
            if text.replace("\\", "/").startswith("out/"):
                path = REPO_ROOT / path
            else:
                path = OUT_ROOT / path
        path = path.resolve()
        try:
            path.relative_to(OUT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"推理输出目录必须在 out/ 下: {path}") from exc
        return path

    def _start_job(
        self, kind: str, command: list[str], output: Path, extra: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle()
            job_id = uuid.uuid4().hex[:8]
            log_path = (
                OUT_ROOT
                / "webui"
                / f"{time.strftime('%Y%m%d_%H%M%S')}_{kind}_{job_id}.log"
            )
            job = Job(
                id=job_id,
                kind=kind,
                status="running",
                started_at=time.time(),
                log_path=log_path,
                output_dir=output,
                command=command,
                extra=extra,
            )
            self._job = job
        try:
            self._spawn(job)
        except Exception as exc:  # noqa: BLE001 - surface spawn failures
            with self._lock:
                job.status = "failed"
                job.finished_at = time.time()
                job.returncode = -1
                job.extra = {**job.extra, "spawn_error": str(exc)}
            raise ValueError(f"无法启动进程: {exc}") from exc
        return self.snapshot(include_log=False)

    def _spawn(self, job: Job) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        else:
            kwargs["start_new_session"] = True
        with job.log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            job.process = subprocess.Popen(
                job.command,
                cwd=str(REPO_ROOT),
                env=_child_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                **kwargs,
            )
        threading.Thread(target=self._monitor, args=(job,), daemon=True).start()

    def _monitor(self, job: Job) -> None:
        assert job.process is not None
        returncode = job.process.wait()
        with self._lock:
            if self._job is not job:
                return
            job.returncode = returncode
            job.finished_at = time.time()
            if job.stop_requested:
                job.status = "stopped"
            elif returncode == 0:
                job.status = "completed"
            else:
                job.status = "failed"

    def stop(self) -> bool:
        with self._lock:
            job = self._job
            if job is None or job.process is None or not self._active():
                return False
            job.stop_requested = True
            job.status = "stopping"
            process = job.process
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                creationflags=flags,
            )
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                process.terminate()

        def force_kill() -> None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        threading.Thread(target=force_kill, daemon=True).start()
        return True

    # -- snapshots -------------------------------------------------------
    def snapshot(self, *, include_log: bool = True) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {"active": False, "job": None}
            now = time.time()
            data: dict[str, Any] = {
                "active": job.status in ("running", "stopping"),
                "job": {
                    "id": job.id,
                    "kind": job.kind,
                    "status": job.status,
                    "started_at": job.started_at,
                    "finished_at": job.finished_at,
                    "elapsed": round((job.finished_at or now) - job.started_at, 1),
                    "returncode": job.returncode,
                    "output_dir": _rel(job.output_dir),
                    "log_path": _rel(job.log_path),
                    "command": list(job.command),
                    "extra": _jsonable(job.extra),
                },
            }
            if job.kind == "infer":
                data["result_maidata"] = _rel(job.output_dir / "maidata.txt")
        if include_log:
            data["log"] = _tail_lines(job.log_path, max_lines=260)
        data["metrics"] = (
            _read_metrics(job.output_dir / "metrics.jsonl") if job.kind == "train" else []
        )
        return data


def get_manager() -> JobManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = JobManager()
    return _MANAGER


# ---------------------------------------------------------------------------
# option scanners


def _list_configs() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted(CONFIG_ROOT.glob("*.yaml")) + sorted(CONFIG_ROOT.glob("*.yml")):
        items.append({"value": _rel(path), "label": path.name})
    return items


def _list_checkpoints() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in RUNS_ROOT.glob("**/*.pt"):
        try:
            stat = path.stat()
        except OSError:
            continue
        name = path.name
        rank = 0 if name == "best.pt" else 1 if name == "last.pt" else 2
        items.append(
            {
                "value": _rel(path),
                "label": f"{_rel(path)} · {stat.st_size / 1024 / 1024:.1f} MB",
                "name": name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "rank": rank,
                "recommended": name == "best.pt",
            }
        )
    items.sort(key=lambda item: (item["rank"], -item["mtime"]))
    return items[:120]


def _list_audio() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(path: Path, group: str, label: str) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if not resolved.is_file() or resolved.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        key = os.path.normcase(str(resolved))
        if key in seen:
            return
        seen.add(key)
        items.append(
            {
                "value": str(resolved),
                "label": label,
                "group": group,
                "name": resolved.name,
            }
        )

    if DESKTOP.exists():
        for path in sorted(DESKTOP.iterdir()):
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                add(path, "desktop", f"桌面 · {path.name}")

    raw_ids: set[str] = set()
    raw_root = DATA_ROOT / "raw"
    if raw_root.exists():
        for song_dir in sorted(raw_root.iterdir()):
            if not song_dir.is_dir():
                continue
            for path in sorted(song_dir.glob("audio.*")):
                if path.suffix.lower() in AUDIO_EXTENSIONS:
                    raw_ids.add(song_dir.name)
                    add(
                        path,
                        "raw",
                        f"{song_dir.name} · {_song_title(song_dir.name)}",
                    )

    dataset_root = REPO_ROOT / "dataset"
    if dataset_root.exists():
        for song_dir in sorted(dataset_root.iterdir()):
            if not song_dir.is_dir() or song_dir.name in raw_ids:
                continue
            for path in sorted(song_dir.glob("track.*")):
                if path.suffix.lower() in AUDIO_EXTENSIONS:
                    add(path, "dataset", f"dataset · {song_dir.name}")
    return items[:400]


def _list_timing() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in OUT_ROOT.glob("**/*.json"):
        if "timing" not in path.name.lower():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        key = os.path.normcase(str(path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "value": _rel(path),
                "label": _rel(path),
                "mtime": stat.st_mtime,
            }
        )
    items.sort(key=lambda item: -item["mtime"])
    return items[:100]


def _list_features() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted((DATA_ROOT / "features").glob("*.npz")):
        items.append({"value": _rel(path), "label": path.name})
    return items[:200]


def _run_summary(path: Path) -> dict[str, Any] | None:
    metrics_path = path / "metrics.jsonl"
    checkpoints = list(path.glob("*.pt"))
    if not metrics_path.exists() and not checkpoints:
        return None
    metrics = _read_metrics(metrics_path)
    val_losses = [
        float(row["val_loss"])
        for row in metrics
        if isinstance(row.get("val_loss"), (int, float))
    ]
    best_val = min(val_losses) if val_losses else None
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
        "name": path.name,
        "path": _rel(path),
        "best_val_loss": best_val,
        "last_epoch": last_epoch,
        "last_step": last_step,
        "checkpoint_count": len(checkpoints),
        "checkpoints": sorted(p.name for p in checkpoints),
        "mtime": mtime,
        "has_metrics": metrics_path.exists(),
    }


def _list_runs() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not RUNS_ROOT.exists():
        return items
    for path in sorted(RUNS_ROOT.iterdir()):
        if not path.is_dir():
            continue
        summary = _run_summary(path)
        if summary:
            items.append(summary)
    items.sort(key=lambda item: -item["mtime"])
    return items[:60]


def _run_detail(name: str) -> dict[str, Any]:
    safe_name = Path(name).name
    if not safe_name or safe_name in (".", "..") or safe_name != name:
        raise ValueError("非法 run 名称")
    path = (RUNS_ROOT / safe_name).resolve()
    try:
        path.relative_to(RUNS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("run 不在 runs/ 下") from exc
    if not path.is_dir():
        raise ValueError(f"run 不存在: {safe_name}")
    summary = _run_summary(path) or {}
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
                "path": _rel(item),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    config = _read_json(path / "config.json", None)
    return {
        **summary,
        "metrics": _read_metrics(path / "metrics.jsonl", limit=1000),
        "files": files,
        "config": config,
    }


def _overview() -> dict[str, Any]:
    sample_summary = _read_json(DATA_ROOT / "samples" / "summary.json", {}) or {}
    feature_index = _read_json(
        DATA_ROOT / "features" / "features_index.json", {}
    ) or {}
    return {
        "songs": sample_summary.get("song_count"),
        "charts": sample_summary.get("chart_count"),
        "split_song_counts": sample_summary.get("split_song_counts"),
        "split_chart_counts": sample_summary.get("split_chart_counts"),
        "difficulty_chart_counts": sample_summary.get("difficulty_chart_counts"),
        "features": feature_index.get("written"),
        "feature_errors": len(feature_index.get("errors") or []),
        "runs": len(_list_runs()),
    }


# ---------------------------------------------------------------------------
# chart / file inspection


def _chart_info(path: Path, index: int | None = None) -> dict[str, Any]:
    maidata = parse_file(path)
    if not maidata.charts:
        raise ValueError("谱面没有难度块")
    chart = maidata.charts[0]
    if index is not None:
        chart = next(
            (item for item in maidata.charts if item.index == index), maidata.charts[0]
        )
    stats = chart_stats(chart)
    validation = validate_maidata(maidata)
    summary = _read_json(path.parent / "p3_summary.json", None)
    audio = summary.get("audio") if isinstance(summary, dict) else None
    if audio and not Path(audio).exists():
        audio = None
    text = path.read_text(encoding="utf-8")
    files = []
    for name in (
        "maidata.txt",
        "p3_ir.json",
        "p3_summary.json",
        "timing.json",
        "p3_validation.json",
    ):
        item = path.parent / name
        if item.exists():
            files.append({"name": name, "path": _rel(item)})
    return {
        "path": _rel(path),
        "title": maidata.metadata.get("title", path.parent.name),
        "artist": maidata.metadata.get("artist", ""),
        "bpm": maidata.metadata.get("wholebpm", ""),
        "difficulty": DIFFICULTY_BY_INDEX.get(chart.index, "unknown"),
        "index": chart.index,
        "level": chart.level,
        "stats": stats,
        "validation": {
            "ok": validation.ok,
            "errors": list(validation.errors),
            "warnings": list(validation.warnings),
        },
        "charts": [
            {
                "index": item.index,
                "difficulty": DIFFICULTY_BY_INDEX.get(item.index, "unknown"),
                "level": item.level,
                "notes": chart_stats(item)["notes_total"],
            }
            for item in maidata.charts
        ],
        "summary": summary,
        "audio": audio,
        "files": files,
        "text": text[:200000],
        "size": path.stat().st_size,
    }


# ---------------------------------------------------------------------------
# Flask app


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
    )
    app.json.ensure_ascii = False
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    manager = get_manager()

    @app.after_request
    def _no_cache(response: Response) -> Response:
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/options")
    def options() -> Response:
        return jsonify(
            {
                "configs": _list_configs(),
                "checkpoints": _list_checkpoints(),
                "audio": _list_audio(),
                "timing": _list_timing(),
                "features": _list_features(),
                "runs": _list_runs(),
                "gpu": _gpu_status(),
                "overview": _overview(),
                "defaults": {
                    "config": "configs/p3_4070.yaml",
                    "checkpoint": "runs/p3_4070/best.pt",
                    "audio": str((DESKTOP / "233.mp3").resolve()),
                    "timing": "out/timing_233_130.json",
                    "train_output": "runs/web_train",
                    "infer_output": "out/web_infer",
                },
                "active_job": manager.snapshot(include_log=False),
            }
        )

    @app.get("/api/job")
    def job() -> Response:
        return jsonify(manager.snapshot())

    @app.post("/api/job/stop")
    def stop() -> Response:
        stopped = manager.stop()
        return jsonify({"stopped": stopped, **manager.snapshot(include_log=False)})

    @app.post("/api/train/start")
    def train_start() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            result = manager.start_train(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.post("/api/infer/start")
    def infer_start() -> Response:
        payload = request.get_json(silent=True) or {}
        try:
            result = manager.start_infer(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.get("/api/runs")
    def runs() -> Response:
        return jsonify({"runs": _list_runs()})

    @app.get("/api/run/<path:name>")
    def run(name: str) -> Response:
        try:
            return jsonify(_run_detail(name))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.get("/api/chart")
    def chart() -> Response:
        raw = request.args.get("path") or ""
        if not raw:
            return jsonify({"error": "缺少 path"}), 400
        try:
            path = _resolve_under(
                raw, [REPO_ROOT, DESKTOP], label="谱面", must_exist=True
            )
            index = request.args.get("index", type=int)
            return jsonify(_chart_info(path, index))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001 - parse boundary
            return jsonify({"error": f"解析失败: {exc}"}), 400

    @app.get("/api/audio")
    def audio() -> Response:
        raw = request.args.get("path") or ""
        try:
            path = _resolve_under(
                raw, [REPO_ROOT, DESKTOP], label="音频", must_exist=True
            )
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                raise ValueError("不是支持的音频文件")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        mimetype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return send_file(
            path,
            mimetype=mimetype,
            conditional=True,
            download_name=path.name,
            as_attachment=request.args.get("download") == "1",
        )

    @app.get("/api/file")
    def file() -> Response:
        raw = request.args.get("path") or ""
        try:
            path = _resolve_under(
                raw, [REPO_ROOT, DESKTOP], label="文件", must_exist=True
            )
            if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
                raise ValueError("不支持预览该文件")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if request.args.get("download") == "1":
            return send_file(path, as_attachment=True, download_name=path.name)
        text = path.read_text(encoding="utf-8", errors="replace")
        return Response(text, content_type="text/plain; charset=utf-8")

    return app


def main(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    app = create_app()
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"MaiForge Studio: {url}", flush=True)
    try:
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
