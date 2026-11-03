"""Subprocess job manager for training and inference.

Only the documented ``mainote`` CLI entry points are spawned; the browser
cannot send arbitrary commands.  One job runs at a time, mirroring the GPU
constraint of real training.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..audio.io import AUDIO_EXTENSIONS
from .settings import WebSettings
from .training import (
    FEATURES_REL,
    SAMPLES_REL,
    apply_hyperparams,
    build_selection_manifest,
    features_missing,
    normalize_song_ids,
    read_config,
    validate_song_ids,
    write_config,
)

DIFFICULTIES = {"basic", "advanced", "expert", "master", "remaster"}
DEVICES = {"auto", "cuda", "cpu"}


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


def _looks_used(path: Path) -> bool:
    if not path.exists():
        return False
    if (path / "last.pt").exists() or (path / "metrics.jsonl").exists():
        return True
    try:
        return any(path.iterdir())
    except OSError:
        return False


def _unique_dir(path: Path) -> Path:
    candidate = path
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{index}")
        index += 1
    return candidate


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
    def __init__(self, settings: WebSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._job: Job | None = None
        self._history: list[Job] = []

    # -- lifecycle -------------------------------------------------------
    def _active(self) -> bool:
        return self._job is not None and self._job.status in ("running", "stopping")

    def _ensure_idle(self) -> None:
        if self._active():
            raise ValueError("已有任务在运行,请先停止或等待完成")

    def start_train(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self.settings
        config = settings.resolve_under(
            payload.get("config") or "configs/p3_base.yaml",
            [settings.config_root],
            label="配置文件",
            must_exist=True,
        )
        if config.suffix.lower() not in (".yaml", ".yml"):
            raise ValueError("训练配置必须是 .yaml/.yml")
        device = str(payload.get("device") or "auto")
        if device not in DEVICES:
            raise ValueError("device 必须是 auto/cuda/cpu")

        resume = payload.get("resume")
        resume_path: Path | None = None
        if resume:
            resume_path = settings.resolve_under(
                resume,
                [settings.runs_root, settings.out_root],
                label="resume checkpoint",
                must_exist=True,
            )

        raw_output = str(payload.get("output_dir") or "").strip()
        if resume_path is not None and not raw_output:
            output = resume_path.parent
        else:
            output = self._under_root(
                raw_output or "web_train", settings.runs_root, "训练输出目录"
            )
        if resume_path is None and _looks_used(output):
            output = _unique_dir(output)
        output.mkdir(parents=True, exist_ok=True)

        max_steps = _optional_int(payload.get("max_steps"), "max_steps")
        if max_steps == 0:
            max_steps = None
        num_workers = _optional_int(payload.get("num_workers"), "num_workers")
        dry_run = bool(payload.get("dry_run"))

        song_ids = normalize_song_ids(payload.get("song_ids"))
        hyperparams = payload.get("hyperparams")
        seed = _optional_int(payload.get("seed"), "seed", minimum=0)
        val_ratio = _optional_float(
            payload.get("val_ratio"), "val_ratio", minimum=0.0
        )
        if val_ratio is not None and val_ratio > 0.95:
            raise ValueError("val_ratio 不能大于 0.95")
        if val_ratio is None:
            val_ratio = 0.1
        features_value = str(payload.get("features_dir") or "").strip()
        generated = bool(
            song_ids or hyperparams is not None or seed is not None or features_value
        )

        cfg: dict[str, Any] | None = None
        manifest_info: dict[str, Any] | None = None
        manifest_path: Path | None = None
        if generated:
            cfg = read_config(config)
            if song_ids:
                selected_root = settings.data_root_by_value(
                    str(payload.get("root") or "") or None
                )
                specs = validate_song_ids(settings, selected_root, song_ids)
                no_audio = [item["id"] for item in specs if not item["has_audio"]]
                if no_audio:
                    shown = "、".join(no_audio[:6])
                    raise ValueError(
                        f"以下歌曲没有音频，无法用于训练: {shown}"
                    )
                feature_dir = settings.resolve_under(
                    features_value or str(FEATURES_REL),
                    [settings.repo_root],
                    label="特征目录",
                )
                missing = features_missing(settings, song_ids, feature_dir)
                if missing:
                    shown = "、".join(missing[:6])
                    suffix = " 等" if len(missing) > 6 else ""
                    raise ValueError(
                        f"以下歌曲缺少 log-mel 特征: {shown}{suffix}。"
                        "请先点击“准备所选数据”。"
                    )
            if hyperparams is not None:
                apply_hyperparams(cfg, hyperparams)
            if seed is not None:
                cfg["seed"] = seed

            if song_ids:
                assert feature_dir is not None
                manifest_path = output / "training_manifest.jsonl"
                manifest_info = build_selection_manifest(
                    settings,
                    song_ids,
                    manifest_path,
                    seed=seed if seed is not None else int(cfg.get("seed", 0) or 0),
                    val_ratio=val_ratio,
                )
                data_cfg = cfg.setdefault("data", {})
                if not isinstance(data_cfg, dict):
                    raise ValueError("配置中的 data 不是对象")
                data_cfg["manifest"] = str(manifest_path.resolve())
                data_cfg["features_dir"] = str(feature_dir.resolve())
                train_cfg = cfg.setdefault("train", {})
                if not isinstance(train_cfg, dict):
                    raise ValueError("配置中的 train 不是对象")
                if not manifest_info["val_songs"]:
                    # Only one song selected: note-level evaluation needs a val
                    # split, so fall back to validation by training loss.
                    note_eval = train_cfg.setdefault("note_eval", {})
                    if isinstance(note_eval, dict):
                        note_eval["enabled"] = False
                    train_cfg["checkpoint_metric"] = "val_loss"

            generated_config = output / "train_config.yaml"
            write_config(cfg, generated_config)
            config = generated_config

        epochs = 0
        try:
            source = cfg if cfg is not None else read_config(config)
            epochs = int(source.get("train", {}).get("epochs", 0) or 0)
        except Exception:  # noqa: BLE001 - display only
            epochs = 0

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

        extra: dict[str, Any] = {
            "config": settings.rel(config),
            "resume": settings.rel(resume_path) if resume_path else None,
            "epochs": epochs,
            "max_steps": max_steps,
            "dry_run": dry_run,
            "device": device,
        }
        if manifest_info is not None and manifest_path is not None:
            extra.update(
                {
                    "songs": len(song_ids),
                    "train_songs": len(manifest_info["train_songs"]),
                    "val_songs": len(manifest_info["val_songs"]),
                    "manifest": settings.rel(manifest_path),
                }
            )
        return self._start_job("train", command, output, extra)

    def start_prepare(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Export samples + log-mel features for selected songs."""
        settings = self.settings
        root = settings.data_root_by_value(str(payload.get("root") or "") or None)
        song_ids = normalize_song_ids(payload.get("song_ids"))
        if not song_ids:
            raise ValueError("请选择至少一首歌再准备数据")
        specs = validate_song_ids(settings, root, song_ids)
        no_audio = [item["id"] for item in specs if not item["has_audio"]]
        if no_audio:
            shown = "、".join(no_audio[:6])
            raise ValueError(f"以下歌曲没有音频，无法提取特征: {shown}")

        samples_out = settings.resolve_under(
            str(payload.get("samples_out") or SAMPLES_REL),
            [settings.repo_root],
            label="样本输出目录",
        )
        features_out = settings.resolve_under(
            str(payload.get("features_out") or FEATURES_REL),
            [settings.repo_root],
            label="特征输出目录",
        )
        overwrite = bool(payload.get("overwrite_features"))
        output = self._under_root("webui/prepare", settings.out_root, "数据准备输出目录")
        output.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "mainote",
            "ml",
            "prepare-data",
            str(root),
            "--samples-out",
            str(samples_out),
            "--features-out",
            str(features_out),
            "--songs",
            ",".join(song_ids),
            "--require-audio",
        ]
        if overwrite:
            command.append("--overwrite-features")
        return self._start_job(
            "prepare",
            command,
            output,
            {
                "root": settings.rel(root),
                "songs": len(song_ids),
                "song_ids": song_ids,
                "samples_out": settings.rel(samples_out),
                "features_out": settings.rel(features_out),
                "overwrite_features": overwrite,
            },
        )

    def start_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = self.settings
        audio = settings.resolve_under(
            payload.get("audio") or "",
            [settings.repo_root],
            label="音频",
            must_exist=True,
        )
        if audio.suffix.lower() not in AUDIO_EXTENSIONS:
            raise ValueError(f"不支持的音频格式: {audio.suffix}")
        checkpoint = settings.resolve_under(
            payload.get("checkpoint") or "",
            [settings.runs_root, settings.out_root],
            label="checkpoint",
            must_exist=True,
        )
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("checkpoint 必须是 .pt 文件")

        timing: Path | None = None
        if payload.get("timing"):
            timing = settings.resolve_under(
                payload["timing"], [settings.repo_root], label="timing", must_exist=True
            )
        features: Path | None = None
        if payload.get("features"):
            features = settings.resolve_under(
                payload["features"], [settings.repo_root], label="features", must_exist=True
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
        bpm = _optional_float(payload.get("bpm"), "bpm", minimum=0.0)
        first = _optional_float(payload.get("first"), "first", minimum=0.0)
        device = str(payload.get("device") or "auto")
        if device not in DEVICES:
            raise ValueError("device 必须是 auto/cuda/cpu")

        output = self._under_root(
            str(payload.get("output_dir") or "web_infer").strip(),
            settings.out_root,
            "推理输出目录",
        )
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
        # Always pass temperature: the CLI default is sampling, so omitting a
        # requested greedy decode would silently sample instead.
        if temperature is not None:
            command += ["--temperature", str(temperature)]
        if top_k is not None:
            command += ["--top-k", str(top_k)]
        if max_new_tokens is not None:
            command += ["--max-new-tokens", str(max_new_tokens)]
        if seed is not None:
            command += ["--seed", str(seed)]
        if bpm is not None:
            command += ["--bpm", str(bpm)]
        if first is not None:
            command += ["--first", str(first)]

        return self._start_job(
            "infer",
            command,
            output,
            {
                "audio": settings.rel(audio),
                "audio_label": audio.name,
                "checkpoint": settings.rel(checkpoint),
                "difficulty": difficulty,
                "level": level,
                "timing": settings.rel(timing) if timing else None,
                "features": settings.rel(features) if features else None,
                "result_maidata": settings.rel(maidata_path),
            },
        )

    def _under_root(self, raw: str, root: Path, label: str) -> Path:
        text = raw.strip()
        if not text:
            text = root.name
        path = Path(text)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"{label}必须在 {self.settings.rel(root)} 下: {path}") from exc
        return path

    def _start_job(
        self, kind: str, command: list[str], output: Path, extra: dict[str, Any]
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle()
            job_id = uuid.uuid4().hex[:8]
            log_path = (
                self.settings.job_dir
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

    def _child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        source = self.settings.repo_root / "src"
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{source}{os.pathsep}{existing}" if existing else str(source)
        scripts = self.settings.repo_root / ".venv" / "bin"
        if scripts.exists():
            env["PATH"] = f"{scripts}{os.pathsep}{env.get('PATH', '')}"
        return env

    def _spawn(self, job: Job) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        with job.log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            job.process = subprocess.Popen(
                job.command,
                cwd=str(self.settings.repo_root),
                env=self._child_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
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
            self._history.append(job)
            del self._history[:-20]

    def stop(self) -> bool:
        with self._lock:
            job = self._job
            if job is None or job.process is None or not self._active():
                return False
            job.stop_requested = True
            job.status = "stopping"
            process = job.process
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
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
    def _job_dict(self, job: Job) -> dict[str, Any]:
        now = time.time()
        return {
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "elapsed": round((job.finished_at or now) - job.started_at, 1),
            "returncode": job.returncode,
            "output_dir": self.settings.rel(job.output_dir),
            "log_path": self.settings.rel(job.log_path),
            "command": list(job.command),
            "extra": self._jsonable(job.extra),
        }

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, Path):
            return self.settings.rel(value)
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        return value

    def snapshot(self, *, include_log: bool = True) -> dict[str, Any]:
        with self._lock:
            job = self._job
            if job is None:
                return {"active": False, "job": None, "history": []}
            data: dict[str, Any] = {
                "active": job.status in ("running", "stopping"),
                "job": self._job_dict(job),
                "history": [self._job_dict(item) for item in reversed(self._history[-8:])],
            }
            if job.kind == "infer":
                data["result_maidata"] = self.settings.rel(job.output_dir / "maidata.txt")
        if include_log:
            data["log"] = _tail_lines(job.log_path, max_lines=260)
        data["metrics"] = (
            _read_metrics(job.output_dir / "metrics.jsonl") if job.kind == "train" else []
        )
        return data
