"""Portable ``.maidb`` dataset bundles.

A bundle is a zip archive with a top-level ``maidb.json`` manifest and one
directory per song under ``songs/<id>/`` holding the raw maidata, audio and
background files.  Import re-runs the normal ingest pipeline for every song,
so absolute paths from the exporting machine never leak into the new dataset.

The archive itself is untrusted input and goes through the same member
validation as :mod:`mainote.dataset.zipimport` (no absolute/traversal paths,
no symlinks, no encrypted members, size and member-count limits).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

from ..maidata.parser import parse_file
from .zipimport import (
    COPY_CHUNK,
    _sanitize_id,
    _unique_id,
    allocate_song_id,
    empty_song_result,
    extract_members,
    import_song_directory,
    locate_song_files,
    validate_members,
)

FORMAT_NAME = "mainote-maidb"
FORMAT_VERSION = 1
MANIFEST_NAME = "maidb.json"
BUNDLE_SUFFIX = ".maidb"

_TEXT_SUFFIXES = {".txt", ".json", ".xml", ".md", ".csv"}
_COMPRESS_LEVEL = 6


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(COPY_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _has_maidata(directory: Path) -> bool:
    """True when ``directory`` is raw-shaped (a real maidata, not IR JSON)."""
    if not directory.is_dir():
        return False
    for name in ("maidata.txt", "maidata"):
        if (directory / name).is_file():
            return True
    return any(
        path.is_file() and path.suffix.lower() not in {".json", ".ir"}
        for path in directory.glob("maidata*")
    )


def resolve_raw_directory(song_dir: str | Path) -> Path:
    """Map a processed song folder to the raw folder a bundle should contain.

    ``processed`` metadata stores an absolute ``raw_dir``; fall back to the
    folder itself when it is already raw-shaped (contains a maidata file).
    """
    song_dir = Path(song_dir)
    meta = _read_json(song_dir / "meta.json", {})
    raw = meta.get("raw_dir") if isinstance(meta, dict) else None
    if isinstance(raw, str) and raw.strip():
        candidate = Path(raw).expanduser()
        if _has_maidata(candidate):
            return candidate
    if _has_maidata(song_dir):
        return song_dir
    raise ValueError(
        f"{song_dir.name}: 找不到原始歌曲目录"
        "（meta.raw_dir 缺失或数据已移动，请重新 ingest 后再导出）"
    )


def bundle_filename(value: str | Path) -> str:
    """Normalize a user-supplied name to a safe ``*.maidb`` basename."""
    name = Path(str(value)).name
    stem = _sanitize_id(Path(name).stem) or "dataset"
    return f"{stem}{BUNDLE_SUFFIX}"


def export_bundle(
    songs: Iterable[tuple[str, str | Path]],
    output: str | Path,
    *,
    include_audio: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Write ``(song_id, raw_dir)`` pairs into one ``.maidb`` archive.

    The archive is built in a temporary file and moved into place only after
    every song has been written, so a failed export never leaves a truncated
    bundle behind.
    """
    output = Path(output)
    output = output.with_name(bundle_filename(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.part")

    entries: list[dict[str, Any]] = []
    songs_summary: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_dirs: set[str] = set()

    # Validate every source before opening the archive; export must not
    # silently drop songs.
    prepared: list[tuple[str, str, Path, Path, Path | None, Path | None, dict[str, Any]]] = []
    for song_id, source in songs:
        source_dir = Path(source)
        maidata, audio, background = locate_song_files(source_dir)
        if maidata is None:
            raise ValueError(f"{song_id}: 原始目录里没有 maidata 文件: {source_dir}")
        if not include_audio:
            audio = None
        dir_name = _unique_id(_sanitize_id(str(song_id)) or "song", used_dirs)
        used_dirs.add(dir_name)

        parsed = None
        try:
            parsed = parse_file(maidata)
        except Exception as exc:  # noqa: BLE001 - keep exporting the raw files
            warnings.append(f"{song_id}: maidata 解析失败，仍按原样导出: {exc}")
        metadata = parsed.metadata if parsed is not None else {}
        entry = {
            "id": str(song_id),
            "dir": f"songs/{dir_name}",
            "title": metadata.get("title") or song_id,
            "artist": metadata.get("artist") or "",
            "chart_count": len(parsed.charts) if parsed is not None else 0,
            "files": [],
        }
        prepared.append(
            (str(song_id), dir_name, source_dir, maidata, audio, background, entry)
        )

    if not prepared:
        raise ValueError("没有可导出的歌曲")

    try:
        with zipfile.ZipFile(
            partial,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=_COMPRESS_LEVEL,
            allowZip64=True,
        ) as archive:
            for song_id, dir_name, _source_dir, maidata, audio, background, entry in prepared:
                if log:
                    log(f"打包 {song_id}")
                files: list[Path] = [maidata]
                if audio is not None:
                    files.append(audio)
                if background is not None:
                    files.append(background)
                seen: set[str] = set()
                for source_file in files:
                    arcname = f"{entry['dir']}/{source_file.name}"
                    if arcname in seen:
                        raise ValueError(f"{song_id}: 文件名重复: {source_file.name}")
                    seen.add(arcname)
                    entry["files"].append(
                        _write_file_entry(archive, source_file, arcname)
                    )
                entries.append(entry)
                songs_summary.append(
                    {
                        "id": entry["id"],
                        "title": entry["title"],
                        "artist": entry["artist"],
                        "chart_count": entry["chart_count"],
                        "has_audio": audio is not None,
                        "has_background": background is not None,
                    }
                )

            manifest = {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "generator": "mainote",
                "song_count": len(entries),
                "songs": entries,
            }
            info = zipfile.ZipInfo(MANIFEST_NAME, date_time=time.localtime()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(
                info,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    file_count = sum(len(entry["files"]) for entry in entries)
    return {
        "ok": True,
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "file": output.name,
        "path": str(output),
        "song_count": len(entries),
        "file_count": file_count,
        "bytes": output.stat().st_size,
        "songs": songs_summary,
        "warnings": warnings,
        "errors": [],
    }


def _write_file_entry(
    archive: zipfile.ZipFile, source: Path, arcname: str
) -> dict[str, Any]:
    info = zipfile.ZipInfo.from_file(source, arcname)
    if source.suffix.lower() not in _TEXT_SUFFIXES:
        info.compress_type = zipfile.ZIP_STORED
    digest = hashlib.sha256()
    with source.open("rb") as src, archive.open(info, "w") as dst:
        while chunk := src.read(COPY_CHUNK):
            digest.update(chunk)
            dst.write(chunk)
    return {
        "path": source.name,
        "size": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _parse_manifest(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("maidb.json 必须是 JSON 对象")
    if data.get("format") != FORMAT_NAME:
        raise ValueError("不是 mainote maidb 数据集（format 不匹配）")
    version = data.get("version")
    if not isinstance(version, int) or version < 1:
        raise ValueError("maidb.json 缺少合法的 version")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"maidb 版本 {version} 过新，当前程序只支持到 {FORMAT_VERSION}，请升级 mainote"
        )
    songs = data.get("songs")
    if not isinstance(songs, list) or not songs:
        raise ValueError("maidb.json 里没有歌曲")
    return data


def _manifest_path(staging: Path) -> Path | None:
    direct = staging / MANIFEST_NAME
    if direct.is_file():
        return direct
    matches = sorted(
        staging.rglob(MANIFEST_NAME),
        key=lambda item: (len(item.relative_to(staging).parts), str(item)),
    )
    return matches[0] if matches else None


def _manifest_song_dir(base: Path, value: Any, song_id: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{song_id}: 清单缺少 dir")
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{song_id}: 清单包含非法路径: {value}")
    target = base.joinpath(*pure.parts) if pure.parts else base
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"{song_id}: 清单路径越界: {value}") from exc
    if not target.is_dir():
        raise ValueError(f"{song_id}: 清单目录不存在: {value}")
    return target


def _verify_manifest_files(song_dir: Path, entry: dict[str, Any], song_id: str) -> None:
    files = entry.get("files")
    if not isinstance(files, list):
        return
    for item in files:
        if not isinstance(item, dict):
            continue
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not expected:
            continue
        pure = PurePosixPath(relative.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"{song_id}: 清单包含非法文件路径: {relative}")
        target = song_dir.joinpath(*pure.parts)
        if not target.is_file():
            raise ValueError(f"{song_id}: 缺少文件: {relative}")
        if _sha256_file(target) != expected:
            raise ValueError(f"{song_id}: 文件校验失败: {relative}")


def import_bundle(
    source: Path,
    *,
    raw_root: Path,
    processed_root: Path,
    filename: str | None = None,
    align: bool = False,
    staging_root: Path | None = None,
) -> dict[str, Any]:
    """Import every song of a ``.maidb`` archive.

    Returns a batch-shaped result with one nested per-song result each.  A
    plain zip renamed to ``.maidb`` (no manifest) is imported as one song.
    """
    display = filename or source.name
    result: dict[str, Any] = {
        "file": display,
        "kind": "bundle",
        "ok": False,
        "format": FORMAT_NAME,
        "version": None,
        "song_id": None,
        "title": None,
        "artist": None,
        "chart_count": 0,
        "has_audio": False,
        "has_background": False,
        "raw": None,
        "processed": None,
        "bundle_song_count": 0,
        "imported": 0,
        "failed": 0,
        "results": [],
        "warnings": [],
        "errors": [],
    }
    try:
        _import_bundle_locked(
            source,
            display,
            raw_root=raw_root,
            processed_root=processed_root,
            align=align,
            staging_root=staging_root,
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - report bundle-level failures
        result["ok"] = False
        result["errors"].append(str(exc))
    return result


def _import_bundle_locked(
    source: Path,
    filename: str,
    *,
    raw_root: Path,
    processed_root: Path,
    align: bool,
    staging_root: Path | None,
    result: dict[str, Any],
) -> None:
    with zipfile.ZipFile(source) as archive:
        members = validate_members(archive)
        staging_parent = str(staging_root) if staging_root is not None else None
        with tempfile.TemporaryDirectory(
            prefix="mainote_bundle_", dir=staging_parent
        ) as tmp:
            staging = Path(tmp)
            extract_members(archive, members, staging)

            manifest_path = _manifest_path(staging)
            if manifest_path is None:
                _import_manifestless(staging, filename, raw_root, processed_root, align, result)
                return
            base = manifest_path.parent
            try:
                manifest = _parse_manifest(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"maidb.json 不是合法 JSON: {exc}") from exc
            result["version"] = manifest["version"]

            batch_ids: set[str] = set()
            for position, entry in enumerate(manifest["songs"]):
                if not isinstance(entry, dict):
                    result["results"].append(
                        _failed_song(filename, f"song_{position + 1}", "清单条目不是对象")
                    )
                    continue
                entry_id = entry.get("id")
                if not isinstance(entry_id, str) or not entry_id.strip():
                    entry_id = f"song_{position + 1}"
                entry_id = entry_id.strip()
                try:
                    song_dir = _manifest_song_dir(base, entry.get("dir", "."), entry_id)
                    _verify_manifest_files(song_dir, entry, entry_id)
                    allocated = allocate_song_id(
                        entry_id, raw_root, processed_root, taken=batch_ids
                    )
                    batch_ids.add(allocated)
                    song = import_song_directory(
                        song_dir,
                        raw_root=raw_root,
                        processed_root=processed_root,
                        filename=filename,
                        align=align,
                        song_id=allocated,
                    )
                    song["source_id"] = entry_id
                except Exception as exc:  # noqa: BLE001 - one bad song must not stop the bundle
                    song = _failed_song(filename, entry_id, str(exc))
                result["results"].append(song)

            imported = sum(1 for song in result["results"] if song.get("ok"))
            failed = len(result["results"]) - imported
            result.update(
                {
                    "ok": imported > 0 and failed == 0,
                    "song_id": None,
                    "bundle_song_count": len(result["results"]),
                    "imported": imported,
                    "failed": failed,
                    "chart_count": sum(
                        int(song.get("chart_count") or 0)
                        for song in result["results"]
                        if song.get("ok")
                    ),
                    "has_audio": any(
                        song.get("has_audio") for song in result["results"]
                    ),
                    "has_background": any(
                        song.get("has_background") for song in result["results"]
                    ),
                }
            )


def _import_manifestless(
    staging: Path,
    filename: str,
    raw_root: Path,
    processed_root: Path,
    align: bool,
    result: dict[str, Any],
) -> None:
    """Treat an archive without ``maidb.json`` as one ordinary song zip."""
    song = import_song_directory(
        staging,
        raw_root=raw_root,
        processed_root=processed_root,
        filename=filename,
        align=align,
    )
    song["source_id"] = Path(filename).stem
    result["version"] = None
    result["results"] = [song]
    result.update(
        {
            "ok": bool(song.get("ok")),
            "bundle_song_count": 1,
            "imported": 1 if song.get("ok") else 0,
            "failed": 0 if song.get("ok") else 1,
            "chart_count": int(song.get("chart_count") or 0),
            "has_audio": bool(song.get("has_audio")),
            "has_background": bool(song.get("has_background")),
            "warnings": list(song.get("warnings") or []),
            "errors": list(song.get("errors") or []),
        }
    )


def _failed_song(filename: str, source_id: str, message: str) -> dict[str, Any]:
    song = empty_song_result(filename)
    song["source_id"] = source_id
    song["errors"].append(message)
    return song


def inspect_bundle(source: Path) -> dict[str, Any]:
    """Read a bundle's manifest without extracting the archives."""
    with zipfile.ZipFile(source) as archive:
        validate_members(archive)
        names = archive.namelist()
        candidates = sorted(
            (name for name in names if name.replace("\\", "/").endswith(MANIFEST_NAME)),
            key=lambda name: (name.count("/"), name),
        )
        if not candidates:
            return {
                "format": None,
                "version": None,
                "song_count": 0,
                "songs": [],
                "files": len(names),
            }
        try:
            data = json.loads(archive.read(candidates[0]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"maidb.json 不是合法 JSON: {exc}") from exc
        manifest = _parse_manifest(data)
        return {
            "format": manifest["format"],
            "version": manifest["version"],
            "song_count": len(manifest["songs"]),
            "songs": [
                {
                    "id": entry.get("id"),
                    "title": entry.get("title"),
                    "artist": entry.get("artist"),
                    "chart_count": entry.get("chart_count"),
                }
                for entry in manifest["songs"]
                if isinstance(entry, dict)
            ],
            "files": len(names),
        }


__all__ = [
    "BUNDLE_SUFFIX",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "bundle_filename",
    "export_bundle",
    "import_bundle",
    "inspect_bundle",
    "resolve_raw_directory",
]
