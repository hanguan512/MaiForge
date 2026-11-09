"""Background jobs for local dataset archive imports.

A ``.maidb`` bundle can contain hundreds of songs, and ``align`` makes each
song's ingest take noticeably longer.  Running the batch in a background
thread lets the browser poll a real progress value (per archive and, inside a
bundle, per song) instead of staring at one synchronous request.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .datasets import DatasetImporter, failed_import
from .settings import WebSettings

HISTORY_LIMIT = 10


class DatasetImportJobs:
    def __init__(self, settings: WebSettings, importer: DatasetImporter) -> None:
        self.settings = settings
        self.importer = importer
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._latest: dict[str, Any] | None = None

    def new_workspace(self) -> tuple[str, Path]:
        """Reserve a directory for the uploaded archives of a new job."""
        self.ensure_idle()
        job_id = uuid.uuid4().hex[:12]
        directory = self.settings.upload_dir / "datasets" / job_id
        directory.mkdir(parents=True, exist_ok=False)
        return job_id, directory

    def ensure_idle(self) -> None:
        with self._lock:
            if self._latest and self._latest["status"] == "running":
                raise ValueError("已有数据集导入任务在运行，请等待完成")

    def start(
        self,
        job_id: str,
        files: list[tuple[str, Path]],
        *,
        align: bool,
        workspace: Path,
    ) -> dict[str, Any]:
        if not files:
            shutil.rmtree(workspace, ignore_errors=True)
            raise ValueError("请选择至少一个 .zip 或 .maidb")
        with self._lock:
            if self._latest and self._latest["status"] == "running":
                raise ValueError("已有数据集导入任务在运行，请等待完成")
            job: dict[str, Any] = {
                "id": job_id,
                "status": "running",
                "align": bool(align),
                "total_files": len(files),
                "completed_files": 0,
                "current": None,
                "phase": None,
                "current_song": None,
                "songs_done": 0,
                "songs_total": 0,
                "progress": 0.0,
                "results": [],
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
            }
            self._jobs[job_id] = job
            self._latest = job
            self._trim_history()
        thread = threading.Thread(
            target=self._run,
            args=(job, files, align, workspace),
            name=f"mainote-import-{job_id}",
            daemon=True,
        )
        thread.start()
        with self._lock:
            return self._public(job)

    def _run(
        self,
        job: dict[str, Any],
        files: list[tuple[str, Path]],
        align: bool,
        workspace: Path,
    ) -> None:
        total_files = len(files)
        try:
            for position, (name, source) in enumerate(files):
                with self._lock:
                    job["current"] = name
                    job["phase"] = "extract"
                    job["current_song"] = None
                    job["songs_done"] = 0
                    job["songs_total"] = 0
                    job["progress"] = position / total_files

                def progress(
                    update: dict[str, Any], position: int = position
                ) -> None:
                    with self._lock:
                        job["phase"] = str(update.get("phase") or job["phase"])
                        job["current_song"] = update.get("song_id")
                        done = max(0, int(update.get("done") or 0))
                        total = max(0, int(update.get("total") or 0))
                        job["songs_done"] = done
                        job["songs_total"] = total
                        fraction = done / total if total else 0.0
                        job["progress"] = min(
                            1.0, (position + fraction) / total_files
                        )

                try:
                    result = self.importer.import_file(
                        source,
                        filename=name,
                        align=align,
                        progress=progress,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    result = failed_import(name, str(exc))
                with self._lock:
                    job["results"].append(result)
                    job["completed_files"] = position + 1
                    job["progress"] = (position + 1) / total_files
                    job["current_song"] = None
                    job["songs_done"] = 0
                    job["songs_total"] = 0
        except Exception as exc:  # noqa: BLE001 - surface unexpected job failures
            with self._lock:
                job["error"] = str(exc)
        finally:
            with self._lock:
                job["status"] = "done"
                job["current"] = None
                job["phase"] = None
                job["current_song"] = None
                job["finished_at"] = time.time()
            shutil.rmtree(workspace, ignore_errors=True)

    # -- status ----------------------------------------------------------
    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return self._public(job) if job else None

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public(job) for job in self._jobs.values()]

    def _trim_history(self) -> None:
        if len(self._jobs) <= HISTORY_LIMIT:
            return
        finished = sorted(
            (
                job
                for job in self._jobs.values()
                if job["status"] != "running"
            ),
            key=lambda job: job["started_at"],
        )
        for job in finished[: len(self._jobs) - HISTORY_LIMIT]:
            self._jobs.pop(job["id"], None)

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        public = dict(job)
        # The import thread appends while the UI polls; snapshot the list so
        # JSON serialization never observes a half-updated batch.
        public["results"] = [dict(item) for item in job.get("results", [])]
        return public
