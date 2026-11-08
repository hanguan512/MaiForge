"""Import one song per zip archive into the raw and processed dataset trees.

Shared by the web upload endpoint and the online-catalogue downloader.  The
archive is untrusted input: extraction is staged, absolute/traversal paths and
symlinks are rejected, and only the located maidata/audio/background files are
copied into ``raw_root/<song_id>/``.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from ..audio.io import AUDIO_EXTENSIONS
from .ingest import ingest_dataset

MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024  # 1 GiB compressed
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB extracted
MAX_ARCHIVE_MEMBERS = 5000
COPY_CHUNK = 1024 * 1024

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
BACKGROUND_STEMS = ("bg", "cover", "jacket")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sanitize_id(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", name, flags=re.UNICODE).strip("._")
    return (cleaned or "song")[:80]


def _unique_id(base: str, taken: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in taken:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _find_maidata(root: Path) -> Path | None:
    exact: list[Path] = []
    fallback: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name in {"maidata.txt", "maidata"}:
            exact.append(path)
        elif name.startswith("maidata") and path.suffix.lower() not in {".json", ".ir"}:
            fallback.append(path)
    candidates = exact or fallback
    if not candidates:
        return None
    return min(candidates, key=lambda item: (len(item.relative_to(root).parts), str(item)))


def _find_audio(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            0 if item.stem.lower() == "track" else 1,
            len(item.relative_to(root).parts),
            str(item),
        ),
    )


def _find_background(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            0 if item.stem.lower() in BACKGROUND_STEMS else 1,
            len(item.relative_to(root).parts),
            str(item),
        ),
    )


def _find_meta(root: Path, maidata: Path) -> Path | None:
    for candidate in (maidata.parent / "meta.json", root / "meta.json"):
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob("meta.json"))
    return matches[0] if matches else None


def validate_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    total = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        normalized = info.filename.replace("\\", "/")
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError(f"压缩包包含非法路径: {info.filename}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError(f"压缩包包含符号链接,已拒绝: {info.filename}")
        if info.flag_bits & 0x1:
            raise ValueError("不支持加密压缩包")
        total += max(0, info.file_size)
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("压缩包解压后过大(上限 2 GB)")
        members.append(info)
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError("压缩包文件数量过多(上限 5000)")
    return members


def extract_members(
    archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], target: Path
) -> None:
    for info in members:
        pure = PurePosixPath(info.filename.replace("\\", "/"))
        destination = target.joinpath(*pure.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, destination.open("wb") as output:
            shutil.copyfileobj(source, output, length=COPY_CHUNK)


def empty_song_result(filename: str) -> dict[str, Any]:
    """The per-song result shape shared by zip and bundle importers."""
    return {
        "file": filename,
        "kind": "song",
        "ok": False,
        "song_id": None,
        "title": None,
        "artist": None,
        "chart_count": 0,
        "has_audio": False,
        "has_background": False,
        "raw": None,
        "processed": None,
        "warnings": [],
        "errors": [],
    }


def locate_song_files(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Locate the maidata, primary audio and background files under ``root``."""
    return _find_maidata(root), _find_audio(root), _find_background(root)


def allocate_song_id(
    base: str,
    raw_root: Path,
    processed_root: Path,
    taken: set[str] | None = None,
) -> str:
    taken = set(taken or ())
    taken.update(
        child.name
        for root in (raw_root, processed_root)
        if root.is_dir()
        for child in root.iterdir()
        if child.is_dir()
    )
    return _unique_id(_sanitize_id(base), taken)


def import_song_directory(
    staged: Path,
    *,
    raw_root: Path,
    processed_root: Path,
    filename: str,
    align: bool = False,
    song_id: str | None = None,
) -> dict[str, Any]:
    """Copy one staged song directory into ``raw_root`` and ingest it.

    Raises on failure; callers turn that into a per-file/per-song result so a
    batch import can keep going.
    """
    result = empty_song_result(filename)
    maidata = _find_maidata(staged)
    if maidata is None:
        raise ValueError("未找到 maidata 文件")
    audio = _find_audio(staged)
    if audio is None:
        result["warnings"].append("未找到音频文件,仅导入谱面")
    background = _find_background(staged)

    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    if song_id is None:
        song_id = allocate_song_id(
            _sanitize_id(Path(filename).stem), raw_root, processed_root
        )
    song_raw = raw_root / song_id
    song_raw.mkdir(parents=True)

    shutil.copy2(maidata, song_raw / "maidata.txt")
    if audio is not None:
        shutil.copy2(audio, song_raw / f"track{audio.suffix.lower()}")
    if background is not None:
        shutil.copy2(background, song_raw / f"bg{background.suffix.lower()}")
    meta = _find_meta(staged, maidata)
    if meta is not None:
        shutil.copy2(meta, song_raw / "meta.json")

    result["song_id"] = song_id
    result["has_background"] = background is not None
    result["raw"] = str(song_raw)

    try:
        ingested = ingest_dataset(song_raw, processed_root, align=align)
    except Exception:
        shutil.rmtree(song_raw, ignore_errors=True)
        raise
    result["warnings"].extend(ingested.warnings)
    result["errors"].extend(ingested.errors)
    if not ingested.records:
        # Do not leave a half-imported raw song behind: a retry would
        # otherwise allocate a new ``_2`` id and keep accumulating.
        shutil.rmtree(song_raw, ignore_errors=True)
        result["raw"] = None
        result["song_id"] = None
        raise ValueError("谱面导入失败,详见错误信息")
    record = ingested.records[0]
    result.update(
        {
            "ok": True,
            "title": record.get("title"),
            "artist": record.get("artist"),
            "chart_count": int(record.get("chart_count") or 0),
            "has_audio": bool(record.get("has_audio")),
            "processed": str(processed_root / song_id),
        }
    )
    return result


def import_zip_archive(
    source: Path,
    *,
    raw_root: Path,
    processed_root: Path,
    filename: str | None = None,
    align: bool = False,
    staging_root: Path | None = None,
) -> dict[str, Any]:
    """Extract ``source`` and ingest it as one song.

    Returns a per-file result dict; failures are reported in ``errors`` instead
    of raising, so batch importers can keep going.
    """
    display = filename or source.name
    result = empty_song_result(display)
    try:
        _import_locked(
            source,
            display,
            raw_root=raw_root,
            processed_root=processed_root,
            align=align,
            staging_root=staging_root,
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - report per-file failures
        result["ok"] = False
        result["errors"].append(str(exc))
    return result


def _import_locked(
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
            prefix="mainote_import_", dir=staging_parent
        ) as tmp:
            staging = Path(tmp)
            extract_members(archive, members, staging)
            result.update(
                import_song_directory(
                    staging,
                    raw_root=raw_root,
                    processed_root=processed_root,
                    filename=filename,
                    align=align,
                )
            )
