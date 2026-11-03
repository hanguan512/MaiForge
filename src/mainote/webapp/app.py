"""FastAPI application for the local MaiForge studio."""

from __future__ import annotations

import mimetypes
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ..audio.io import AUDIO_EXTENSIONS
from ..dataset.wmc import WmcError
from .architecture import describe_architecture
from .catalog import Catalog
from .datasets import MAX_ARCHIVE_BYTES, DatasetImporter
from .jobs import JobManager
from .options import (
    list_audio,
    list_checkpoints,
    list_configs,
    list_features,
    list_runs,
    list_timing,
    overview,
    run_detail,
)
from .settings import WebSettings
from .training import read_config, training_catalog
from .viewer import ensure_mp3, ensure_silence
from .wmc import WmcService

TEXT_EXTENSIONS = {".txt", ".json", ".jsonl", ".yaml", ".yml", ".log", ".md", ".csv"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _failed_import(name: str, message: str) -> dict[str, Any]:
    return {
        "file": name,
        "ok": False,
        "song_id": None,
        "title": None,
        "artist": None,
        "chart_count": 0,
        "has_audio": False,
        "raw": None,
        "processed": None,
        "warnings": [],
        "errors": [message],
    }


def create_app(settings: WebSettings | None = None) -> FastAPI:
    settings = settings or WebSettings()
    catalog = Catalog(settings)
    manager = JobManager(settings)
    importer = DatasetImporter(settings)
    wmc = WmcService(settings, importer)

    app = FastAPI(
        title="MaiForge Studio",
        version="0.1.0",
        description="Local chart browser, P3 inference and training control room.",
    )
    app.state.settings = settings
    app.state.catalog = catalog
    app.state.manager = manager
    app.state.importer = importer
    app.state.wmc = wmc

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # -- meta ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "mainote-webapp"}

    @app.get("/api/overview")
    def api_overview() -> dict[str, Any]:
        return overview(settings, catalog, manager)

    @app.get("/api/options")
    def api_options() -> dict[str, Any]:
        return {
            "configs": list_configs(settings),
            "checkpoints": list_checkpoints(settings),
            "audio": list_audio(settings, catalog),
            "timing": list_timing(settings),
            "features": list_features(settings),
            "runs": list_runs(settings),
            "gpu": overview(settings, catalog, manager)["gpu"],
            "defaults": {
                "config": "configs/p3_base.yaml",
                "checkpoint": "",
                "train_output": "web_train",
                "infer_output": "web_infer",
            },
            "active_job": manager.snapshot(include_log=False),
        }

    @app.get("/api/roots")
    def api_roots() -> dict[str, Any]:
        return {"roots": catalog.roots()}

    @app.get("/api/train/data")
    def api_train_data(root: str | None = None) -> dict[str, Any]:
        return training_catalog(settings, root)

    @app.get("/api/train/config")
    def api_train_config(
        path: str = Query("configs/p3_base.yaml"),
    ) -> dict[str, Any]:
        resolved = settings.resolve_under(
            path,
            [settings.config_root],
            label="训练配置",
            must_exist=True,
        )
        return {"path": settings.rel(resolved), "config": read_config(resolved)}

    @app.get("/api/model/architecture")
    def api_model_architecture(
        hidden: int = Query(256, ge=8, le=8192),
        audio_layers: int = Query(4, ge=1, le=128),
        decoder_layers: int = Query(6, ge=1, le=128),
        heads: int = Query(8, ge=1, le=256),
        ffn: int = Query(1024, ge=1, le=65536),
        dropout: float = Query(0.1, ge=0.0, lt=1.0),
        max_audio_frames: int = Query(4096, ge=1, le=65536),
        max_tokens: int = Query(2048, ge=1, le=32768),
        n_mels: int = Query(80, ge=1, le=1024),
        audio_stride: int = Query(2, ge=1, le=1024),
        embed_scale: bool = Query(True),
        final_norm: bool = Query(True),
    ) -> dict[str, Any]:
        return describe_architecture(
            hidden=hidden,
            audio_layers=audio_layers,
            decoder_layers=decoder_layers,
            heads=heads,
            ffn=ffn,
            dropout=dropout,
            max_audio_frames=max_audio_frames,
            max_tokens=max_tokens,
            n_mels=n_mels,
            audio_stride=audio_stride,
            embed_scale=embed_scale,
            final_norm=final_norm,
        )

    # -- dataset management ----------------------------------------------
    @app.get("/api/datasets")
    def api_datasets() -> dict[str, Any]:
        return importer.summary()

    @app.post("/api/datasets/import")
    async def api_dataset_import(
        files: list[UploadFile] = File(...),
        align: bool = Form(False),
    ) -> dict[str, Any]:
        if not files:
            raise ValueError("请选择至少一个 zip 压缩包")
        upload_root = settings.upload_dir / "datasets"
        upload_root.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(
            prefix="mainote_uploads_", dir=str(upload_root)
        ) as tmp:
            tmp_dir = Path(tmp)
            for position, upload in enumerate(files):
                name = Path(upload.filename or f"dataset_{position}.zip").name
                if Path(name).suffix.lower() != ".zip":
                    results.append(_failed_import(name, "只支持 .zip 压缩包"))
                    await upload.close()
                    continue
                target = tmp_dir / f"{position:03d}_{name}"
                try:
                    total = 0
                    with target.open("wb") as handle:
                        while chunk := await upload.read(1024 * 1024):
                            total += len(chunk)
                            if total > MAX_ARCHIVE_BYTES:
                                raise ValueError("压缩包过大(上限 1 GB)")
                            handle.write(chunk)
                    results.append(
                        await run_in_threadpool(
                            importer.import_zip,
                            target,
                            filename=name,
                            align=align,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep the batch going
                    results.append(_failed_import(name, str(exc)))
                finally:
                    await upload.close()
        return {
            "ok": all(item["ok"] for item in results),
            "align": align,
            "results": results,
        }

    # -- online catalogue (AWMC-MaiArchive) ------------------------------
    @app.get("/api/wmc/search")
    def api_wmc_search(
        q: str = "",
        version: str | None = None,
        chart_type: str | None = Query(None, alias="type"),
        genre: str | None = None,
        sort: str = "id",
        page: int = 1,
        limit: int = 30,
    ) -> dict[str, Any]:
        return wmc.search(
            q,
            version=version,
            chart_type=chart_type,
            genre=genre,
            sort=sort,
            page=page,
            limit=limit,
        )

    @app.get("/api/wmc/cover/{shortid}")
    def api_wmc_cover(shortid: str, image: str | None = None) -> FileResponse:
        try:
            target = wmc.cover(shortid, image)
        except WmcError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        mimetype = mimetypes.guess_type(str(target))[0] or "image/png"
        return FileResponse(
            target,
            media_type=mimetype,
            filename=target.name,
            headers={"Cache-Control": "public, max-age=604800"},
        )

    @app.post("/api/wmc/import")
    def api_wmc_import(payload: dict[str, Any]) -> dict[str, Any]:
        return wmc.start(
            payload.get("ids"),
            align=bool(payload.get("align")),
            include_video=bool(payload.get("include_video")),
            force=bool(payload.get("force")),
        )

    @app.get("/api/wmc/imports")
    def api_wmc_imports() -> dict[str, Any]:
        return {"jobs": wmc.catalogue_jobs()}

    @app.get("/api/wmc/imports/{job_id}")
    def api_wmc_import(job_id: str) -> dict[str, Any]:
        job = wmc.snapshot(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="导入任务不存在")
        return job

    @app.post("/api/wmc/imports/{job_id}/cancel")
    def api_wmc_cancel(job_id: str) -> dict[str, Any]:
        job = wmc.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="导入任务不存在")
        return job

    # -- browse ----------------------------------------------------------
    @app.get("/api/songs")
    def api_songs(root: str | None = None) -> dict[str, Any]:
        return {"root": root, "songs": catalog.songs(root)}

    @app.get("/api/song")
    def api_song(root: str | None = None, id: str = Query(...)) -> dict[str, Any]:
        return catalog.song(root, id)

    @app.get("/api/chart")
    def api_chart(
        root: str | None = None,
        id: str = Query(...),
        index: int = Query(5),
    ) -> dict[str, Any]:
        return catalog.chart(root, id, index)

    @app.get("/api/chart/file")
    def api_chart_file(
        path: str = Query(...), index: int | None = Query(None)
    ) -> dict[str, Any]:
        return catalog.chart_file(path, index)

    @app.get("/api/audio")
    def api_audio(
        path: str | None = None,
        root: str | None = None,
        song: str | None = None,
    ) -> FileResponse:
        audio = catalog.audio_file(
            path_value=path, root_value=root, song_id=song
        )
        mimetype = mimetypes.guess_type(str(audio))[0] or "application/octet-stream"
        return FileResponse(audio, media_type=mimetype, filename=audio.name)

    @app.get("/api/files")
    def api_file(path: str = Query(...), download: bool = False) -> Any:
        target = settings.resolve_under(
            path, [settings.repo_root], label="文件", must_exist=True
        )
        if not target.is_file():
            raise ValueError(f"不是文件: {path}")
        if download:
            return FileResponse(target, filename=target.name)
        if target.suffix.lower() not in TEXT_EXTENSIONS:
            raise ValueError("不支持预览该文件类型")
        return PlainTextResponse(target.read_text(encoding="utf-8", errors="replace"))

    # -- MajdataView preview --------------------------------------------
    @app.get("/api/majdataview/maidata")
    def api_viewer_maidata(
        root: str | None = None,
        id: str | None = None,
        path: str | None = None,
        index: int = Query(1),
        seek: float = Query(0.0, ge=0.0, le=36000.0),
    ) -> PlainTextResponse:
        if path:
            text = catalog.viewer_maidata_file(path, index, seek)
        elif id:
            text = catalog.viewer_maidata(root, id, index, seek)
        else:
            raise ValueError("缺少谱面 id 或 path")
        return PlainTextResponse(text, media_type="text/plain; charset=utf-8")

    @app.get("/api/majdataview/audio")
    def api_viewer_audio(
        root: str | None = None,
        song: str | None = None,
        path: str | None = None,
        duration: float = Query(60.0, gt=0.0, le=36000.0),
        start: float = Query(0.0, ge=0.0, le=36000.0),
    ) -> FileResponse:
        if path:
            source = catalog.audio_file(
                path_value=path, root_value=None, song_id=None
            )
        elif root and song:
            source = catalog.song_audio(root, song)
        else:
            source = None
        cache_dir = settings.out_root / "webui" / "cache" / "majdataview"
        target = (
            ensure_mp3(source, cache_dir, start=start)
            if source is not None
            else ensure_silence(max(1.0, duration - start), cache_dir)
        )
        return FileResponse(target, media_type="audio/mpeg", filename="track.mp3")

    @app.get("/api/majdataview/bg")
    def api_viewer_bg(
        root: str | None = None, song: str = Query(...)
    ) -> FileResponse:
        target = catalog.background_file(root, song)
        if target is None:
            raise HTTPException(status_code=404, detail="没有背景图")
        mimetype = mimetypes.guess_type(str(target))[0] or "image/png"
        return FileResponse(
            target,
            media_type=mimetype,
            filename=target.name,
            headers={"Cache-Control": "public, max-age=604800"},
        )

    # -- jobs ------------------------------------------------------------
    @app.get("/api/jobs/current")
    def api_job() -> dict[str, Any]:
        return manager.snapshot()

    @app.post("/api/jobs/stop")
    def api_stop() -> dict[str, Any]:
        stopped = manager.stop()
        return {"stopped": stopped, **manager.snapshot(include_log=False)}

    @app.post("/api/train/start")
    def api_train_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return manager.start_train(payload or {})

    @app.post("/api/train/prepare")
    def api_train_prepare(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return manager.start_prepare(payload or {})

    @app.post("/api/infer/start")
    def api_infer_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return manager.start_infer(payload or {})

    @app.post("/api/upload/audio")
    async def api_upload_audio(file: UploadFile = File(...)) -> dict[str, Any]:
        name = Path(file.filename or "upload").name
        suffix = Path(name).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise ValueError(f"不支持的音频格式: {suffix or '(无扩展名)'}")
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        target = settings.upload_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}"
        total = 0
        try:
            with target.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError("音频文件过大(上限 512 MB)")
                    handle.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await file.close()
        return {
            "path": settings.rel(target),
            "name": name,
            "size": total,
        }

    # -- runs ------------------------------------------------------------
    @app.get("/api/runs")
    def api_runs() -> dict[str, Any]:
        return {"runs": list_runs(settings)}

    @app.get("/api/run")
    def api_run(path: str = Query(...)) -> dict[str, Any]:
        return run_detail(settings, path)

    # -- static single-page app -----------------------------------------
    assets = settings.static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    # MajdataView WebGL player (vendored build, GPL-3.0).  Kept outside
    # ``static/`` because ``vite build`` empties that directory.
    mimetypes.add_type("application/wasm", ".wasm")
    mimetypes.add_type("application/octet-stream", ".data")
    if settings.majdataview_dir.is_dir():
        app.mount(
            "/majdataview",
            StaticFiles(directory=settings.majdataview_dir, html=True),
            name="majdataview",
        )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str) -> Any:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        index = settings.static_dir / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(
            "<h1>MaiForge Studio</h1>"
            "<p>前端尚未构建。运行 <code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>,"
            "或在开发时运行 <code>npm run dev</code>。</p>",
            status_code=503,
        )

    return app


def main(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    import uvicorn

    app = create_app()
    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"MaiForge Studio: {url}", flush=True)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        return 0
    return 0
