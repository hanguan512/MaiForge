"""Online catalogue integration: AWMC-MaiArchive (download.wmc.pub).

Searching is a direct read of the site's public JSON API.  Importing runs in a
background thread because one chart zip is several megabytes; the browser polls
the job snapshot.  Only fixed, server-side downloads to AWMC are allowed: the
browser cannot supply a URL, and song ids must be numeric.
"""

from __future__ import annotations

import threading
import time
import urllib.parse
import uuid
import zipfile
from pathlib import Path
from typing import Any

from ..dataset.wmc import IMAGE_NAME_RE, SHORTID_RE, WmcClient
from .datasets import DatasetImporter
from .settings import WebSettings

MAX_IMPORT_SONGS = 100
DOWNLOAD_DELAY = 1.0  # seconds between chart downloads
HISTORY_LIMIT = 20


def _clean_ids(values: Any) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError("请选择至少一首歌曲")
    ids: list[str] = []
    for value in values:
        text = str(value).strip()
        if not SHORTID_RE.match(text):
            raise ValueError(f"非法谱面 ID: {value}")
        if text not in ids:
            ids.append(text)
    if len(ids) > MAX_IMPORT_SONGS:
        raise ValueError(f"单次最多导入 {MAX_IMPORT_SONGS} 首歌曲")
    return ids


class WmcService:
    def __init__(
        self,
        settings: WebSettings,
        importer: DatasetImporter,
        *,
        client: WmcClient | None = None,
    ) -> None:
        self.settings = settings
        self.importer = importer
        self.client = client or WmcClient()
        self.raw_root = settings.repo_root / "data" / "raw"
        self.cache_root = settings.repo_root / "out" / "webui" / "cache" / "wmc"
        self._lock = threading.RLock()
        self._job: dict[str, Any] | None = None
        self._jobs: dict[str, dict[str, Any]] = {}

    # -- catalogue -------------------------------------------------------
    def search(
        self,
        query: str = "",
        *,
        version: str | int | None = None,
        chart_type: str | None = None,
        genre: str | None = None,
        sort: str = "id",
        page: int = 1,
        limit: int = 30,
    ) -> dict[str, Any]:
        data = self.client.search(
            query,
            version=version,
            chart_type=chart_type,
            genre=genre,
            sort=sort,
            page=page,
            limit=limit,
        )
        data["source"] = "AWMC-MaiArchive · download.wmc.pub"
        for chart in data["charts"]:
            cover = self._cover_url(chart)
            if cover:
                chart["cover_url"] = cover
        return data

    @staticmethod
    def _cover_url(chart: dict[str, Any]) -> str | None:
        shortid = str(chart.get("shortid") or "").strip()
        image = str(chart.get("image") or "").strip()
        if not SHORTID_RE.match(shortid) or not IMAGE_NAME_RE.match(image):
            return None
        query = urllib.parse.urlencode({"image": image})
        return f"/api/wmc/cover/{shortid}?{query}"

    def cover(self, shortid: str, image: str | None = None) -> Path:
        """Return a cached local copy of a chart's online cover image."""
        shortid = str(shortid).strip()
        if not SHORTID_RE.match(shortid):
            raise ValueError("谱面 ID 必须是 5-7 位数字")
        name = str(image or "").strip()
        if image is not None and not name:
            raise ValueError("缺少封面文件名")
        if name and not IMAGE_NAME_RE.match(name):
            raise ValueError("非法的封面文件名")
        if not name:
            chart = self.client.chart(shortid)
            name = str(chart.get("image") or "").strip()
            if not IMAGE_NAME_RE.match(name):
                raise ValueError("该谱面没有可用封面")

        target = self.cache_root / "covers" / f"{shortid}_{name}"
        if not (target.is_file() and target.stat().st_size > 0):
            self.client.download_image(shortid, name, target)
        return target

    # -- import jobs -----------------------------------------------------
    def start(
        self,
        ids: Any,
        *,
        align: bool = False,
        include_video: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        selected = _clean_ids(ids)
        with self._lock:
            if self._job and self._job["status"] in {"running", "stopping"}:
                raise ValueError("已有在线导入任务在运行,请等待完成")
            job: dict[str, Any] = {
                "id": uuid.uuid4().hex[:12],
                "status": "running",
                "total": len(selected),
                "completed": 0,
                "current": None,
                "include_video": bool(include_video),
                "align": bool(align),
                "force": bool(force),
                "results": [],
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
                "cancel_requested": False,
            }
            self._job = job
            self._jobs[job["id"]] = job
            self._trim_history()
        thread = threading.Thread(
            target=self._run, args=(job, selected), name="mainote-wmc", daemon=True
        )
        thread.start()
        return self._public(job)

    def _run(self, job: dict[str, Any], ids: list[str]) -> None:
        try:
            for position, shortid in enumerate(ids):
                if job["cancel_requested"]:
                    break
                job["current"] = shortid
                try:
                    result = self._import_one(
                        shortid,
                        align=job["align"],
                        include_video=job["include_video"],
                        force=job["force"],
                    )
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    result = {
                        "shortid": shortid,
                        "ok": False,
                        "skipped": False,
                        "title": None,
                        "errors": [str(exc)],
                    }
                job["results"].append(_normalize_result(result))
                job["completed"] = position + 1
                if position + 1 < len(ids) and not job["cancel_requested"]:
                    time.sleep(DOWNLOAD_DELAY)
        finally:
            job["current"] = None
            job["finished_at"] = time.time()
            if job["cancel_requested"]:
                job["status"] = "cancelled"
            else:
                job["status"] = "done"

    def _import_one(
        self,
        shortid: str,
        *,
        align: bool,
        include_video: bool,
        force: bool,
    ) -> dict[str, Any]:
        song_id = f"wmc_{shortid}"
        if not force and (
            (self.raw_root / song_id).is_dir()
            or (self.importer.processed_root / song_id).is_dir()
        ):
            return {
                "shortid": shortid,
                "ok": True,
                "skipped": True,
                "title": None,
                "song_id": song_id,
                "errors": [],
                "warnings": ["已存在,跳过(勾选强制覆盖可重新下载)"],
            }

        chart = self.client.chart(shortid)
        suffix = "_video" if include_video else ""
        cached = self.cache_root / f"{shortid}{suffix}.zip"
        if not (cached.is_file() and _zip_ok(cached)):
            self.cache_root.mkdir(parents=True, exist_ok=True)
            self.client.download_zip(
                chart, cached, include_video=include_video
            )
        imported = self.importer.import_zip(
            cached,
            filename=f"{song_id}.zip",
            align=align,
        )
        imported.update(
            {
                "shortid": shortid,
                "skipped": False,
                "page": chart.get("page"),
                "version": chart.get("version"),
                "bpm": chart.get("bpm"),
                "levels": chart.get("levels"),
            }
        )
        return imported

    # -- status ----------------------------------------------------------
    def snapshot(self, job_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            if job_id:
                job = self._jobs.get(job_id)
            else:
                job = self._job
            return self._public(job) if job else None

    def cancel(self, job_id: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id) if job_id else self._job
            if job and job["status"] == "running":
                job["cancel_requested"] = True
                job["status"] = "stopping"
            return self._public(job) if job else None

    def catalogue_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public(job) for job in self._jobs.values()]

    def _trim_history(self) -> None:
        if len(self._jobs) <= HISTORY_LIMIT:
            return
        finished = sorted(
            (job for job in self._jobs.values() if job["status"] not in {"running", "stopping"}),
            key=lambda job: job["started_at"],
        )
        for job in finished[: len(self._jobs) - HISTORY_LIMIT]:
            self._jobs.pop(job["id"], None)

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value
            for key, value in job.items()
            if key != "cancel_requested"
        }
        # The import thread appends to ``results`` while the UI polls; snapshot
        # the list so serialization never observes a half-updated job.
        public["results"] = [dict(item) for item in job.get("results", [])]
        return public


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Guarantee the result shape the browser expects.

    Failed downloads and cancelled songs do not run through
    :func:`mainote.dataset.zipimport.import_zip_archive`, so they used to omit
    ``warnings``.  The web UI spreads both arrays when rendering and crashed
    with a blank page as soon as a single result lacked one.
    """
    normalized = dict(result)
    for key in ("errors", "warnings"):
        value = normalized.get(key)
        if value is None:
            normalized[key] = []
        elif isinstance(value, (list, tuple)):
            normalized[key] = [str(item) for item in value]
        else:
            normalized[key] = [str(value)]
    normalized.setdefault("song_id", None)
    normalized.setdefault("title", None)
    normalized.setdefault("skipped", False)
    return normalized


def _zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False
