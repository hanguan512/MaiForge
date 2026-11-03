"""Training-data selection, manifest filtering and config overrides."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .settings import WebSettings

SAMPLES_REL = Path("data/samples")
FEATURES_REL = Path("data/features")


def samples_dir(settings: WebSettings) -> Path:
    return settings.repo_root / SAMPLES_REL


def features_dir(settings: WebSettings) -> Path:
    return settings.repo_root / FEATURES_REL


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_sample_path(settings: WebSettings, row: dict[str, Any]) -> Path | None:
    raw = row.get("sample")
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = settings.repo_root / path
        if path.is_file():
            return path
    song_id = str(row.get("song_id") or "")
    chart_index = row.get("chart_index")
    if song_id and chart_index is not None:
        path = samples_dir(settings) / song_id / f"inote_{int(chart_index)}.json"
        if path.is_file():
            return path
    return None


def _chart_briefs(meta: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for chart in meta.get("charts") or []:
        if not isinstance(chart, dict):
            continue
        items.append(
            {
                "index": chart.get("index"),
                "level": chart.get("level"),
                "difficulty": chart.get("difficulty"),
            }
        )
    return items


def _audio_exists(song_dir: Path, meta: dict[str, Any]) -> bool:
    raw = meta.get("audio")
    if not raw:
        return False
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = song_dir / path
    return path.is_file()


def training_catalog(
    settings: WebSettings, root_value: str | None = None
) -> dict[str, Any]:
    """List songs in one processed root with sample/feature readiness."""
    root = settings.data_root_by_value(root_value)
    manifest_path = samples_dir(settings) / "manifest.jsonl"
    rows = read_manifest(manifest_path)
    rows_by_song: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_song.setdefault(str(row.get("song_id") or ""), []).append(row)

    feature_dir = features_dir(settings)
    songs: list[dict[str, Any]] = []
    if root.is_dir():
        for song_dir in sorted(root.iterdir()):
            meta_path = song_dir / "meta.json"
            if not song_dir.is_dir() or not meta_path.is_file():
                continue
            meta = _read_json(meta_path, {}) or {}
            song_id = str(meta.get("song_id") or song_dir.name)
            song_rows = rows_by_song.get(song_id, [])
            # Training only uses valid rows, so readiness must ignore sample
            # files that belong to charts rejected by validation.
            valid_rows = [row for row in song_rows if bool(row.get("valid", True))]
            sample_ready = bool(valid_rows) and all(
                _row_sample_path(settings, row) is not None for row in valid_rows
            )
            features_ready = (feature_dir / f"{song_id}.npz").is_file()
            has_audio = _audio_exists(song_dir, meta)
            has_ir = (song_dir / "maidata.ir.json").is_file()
            songs.append(
                {
                    "id": song_id,
                    "title": str(meta.get("title") or song_id),
                    "artist": str(meta.get("artist") or ""),
                    "chart_count": len(meta.get("charts") or []),
                    "charts": _chart_briefs(meta),
                    "has_audio": has_audio,
                    "has_ir": has_ir,
                    "sample_count": len(song_rows),
                    "sample_ready": sample_ready,
                    "features_ready": features_ready,
                    "trainable": has_audio and has_ir and sample_ready and features_ready,
                }
            )

    return {
        "root": settings.rel(root),
        "roots": [
            {
                "value": item["value"],
                "label": item["label"],
                "exists": item["exists"],
                "song_count": item["song_count"],
            }
            for item in _roots(settings, root)
        ],
        "songs": songs,
        "manifest": {
            "path": settings.rel(manifest_path),
            "exists": manifest_path.is_file(),
            "rows": len(rows),
        },
        "samples_dir": settings.rel(samples_dir(settings)),
        "features_dir": settings.rel(feature_dir),
        "summary": {
            "songs": len(songs),
            "sample_ready": sum(1 for song in songs if song["sample_ready"]),
            "features_ready": sum(1 for song in songs if song["features_ready"]),
            "trainable": sum(1 for song in songs if song["trainable"]),
        },
    }


def _roots(settings: WebSettings, current: Path) -> list[dict[str, Any]]:
    items = []
    for root in settings.data_roots:
        count = 0
        if root.is_dir():
            count = sum(
                1
                for child in root.iterdir()
                if child.is_dir() and (child / "meta.json").is_file()
            )
        relative = settings.rel(root)
        if relative.startswith("data/"):
            label = "主数据集"
        elif "demo" in relative:
            label = "演示数据"
        elif "audio-fm" in relative:
            label = "音频测试数据"
        else:
            label = root.name
        items.append(
            {
                "value": relative,
                "label": label,
                "exists": root.is_dir(),
                "song_count": count,
            }
        )
    return items


def normalize_song_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = [str(item) for item in value]
    else:
        raise ValueError("song_ids 必须是字符串或列表")
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        song_id = item.strip()
        if not song_id or song_id in seen:
            continue
        if "," in song_id or Path(song_id).name != song_id or song_id in (".", ".."):
            raise ValueError(f"非法歌曲 id: {song_id}")
        seen.add(song_id)
        result.append(song_id)
    if len(result) > 500:
        raise ValueError("一次最多选择 500 首歌")
    return result


def validate_song_ids(
    settings: WebSettings, root: Path, song_ids: Iterable[str]
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for song_id in song_ids:
        song_dir = root / song_id
        meta_path = song_dir / "meta.json"
        if not song_dir.is_dir() or not meta_path.is_file():
            raise ValueError(f"歌曲不存在于 {settings.rel(root)}: {song_id}")
        meta = _read_json(meta_path, {}) or {}
        specs.append(
            {
                "id": song_id,
                "title": str(meta.get("title") or song_id),
                "has_audio": _audio_exists(song_dir, meta),
                "has_ir": (song_dir / "maidata.ir.json").is_file(),
            }
        )
    return specs


def features_missing(
    settings: WebSettings, song_ids: Iterable[str], directory: Path | None = None
) -> list[str]:
    directory = directory or features_dir(settings)
    return [
        song_id
        for song_id in song_ids
        if not (directory / f"{song_id}.npz").is_file()
    ]


def assign_selected_splits(
    song_ids: Iterable[str], *, seed: int = 0, val_ratio: float = 0.1
) -> dict[str, str]:
    """Assign train/val by song.

    Small selections keep at least one training song.  Two or more songs always
    produce a validation song when ``val_ratio > 0`` so note-level evaluation
    remains available; a single song trains without validation.
    """
    ordered = sorted(
        set(song_ids),
        key=lambda song_id: (
            hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).hexdigest(),
            song_id,
        ),
    )
    if len(ordered) < 2 or val_ratio <= 0:
        return {song_id: "train" for song_id in ordered}
    n_val = max(1, int(round(len(ordered) * float(val_ratio))))
    n_val = min(n_val, len(ordered) - 1)
    validation = set(ordered[:n_val])
    return {
        song_id: "val" if song_id in validation else "train"
        for song_id in ordered
    }


def build_selection_manifest(
    settings: WebSettings,
    song_ids: Iterable[str],
    destination: Path,
    *,
    seed: int = 0,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    """Write a song-filtered manifest with train/val splits by song."""
    song_ids = list(dict.fromkeys(song_ids))
    if not song_ids:
        raise ValueError("请选择至少一首训练歌曲")
    manifest_path = samples_dir(settings) / "manifest.jsonl"
    rows = read_manifest(manifest_path)
    rows_by_song: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_song.setdefault(str(row.get("song_id") or ""), []).append(row)

    missing_rows: list[str] = []
    missing_files: list[str] = []
    selected: list[dict[str, Any]] = []
    for song_id in song_ids:
        usable = [
            row
            for row in rows_by_song.get(song_id, [])
            if bool(row.get("has_audio", True)) and bool(row.get("valid", True))
        ]
        if not usable:
            missing_rows.append(song_id)
            continue
        for row in usable:
            if _row_sample_path(settings, row) is None:
                missing_files.append(song_id)
                break
        selected.extend(usable)

    if missing_rows:
        shown = "、".join(missing_rows[:6])
        suffix = " 等" if len(missing_rows) > 6 else ""
        raise ValueError(
            f"以下歌曲还没有可用的训练样本: {shown}{suffix}。请先点击“准备所选数据”。"
        )
    if missing_files:
        shown = "、".join(missing_files[:6])
        suffix = " 等" if len(missing_files) > 6 else ""
        raise ValueError(f"样本文件缺失: {shown}{suffix}。请重新准备数据。")

    split_of = assign_selected_splits(song_ids, seed=seed, val_ratio=val_ratio)
    output_rows: list[dict[str, Any]] = []
    for row in selected:
        item = dict(row)
        song_id = str(item.get("song_id") or "")
        item["split"] = split_of[song_id]
        sample_path = _row_sample_path(settings, item)
        if sample_path is not None:
            item["sample"] = str(sample_path.resolve())
        output_rows.append(item)

    output_rows.sort(
        key=lambda row: (
            str(row.get("song_id") or ""),
            int(row.get("chart_index") or 0),
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    train_songs = sorted(
        song_id for song_id, split in split_of.items() if split == "train"
    )
    val_songs = sorted(
        song_id for song_id, split in split_of.items() if split == "val"
    )
    return {
        "path": destination,
        "rows": len(output_rows),
        "train_songs": train_songs,
        "val_songs": val_songs,
        "split_of": split_of,
    }


# -- config -------------------------------------------------------------

_INT_SPECS: dict[tuple[str, str], tuple[int, int | None]] = {
    ("data", "window_bars"): (1, 1024),
    ("data", "stride_bars"): (1, 1024),
    ("data", "context_bars"): (0, 128),
    ("data", "min_events"): (0, 100000),
    ("data", "max_windows_per_chart"): (1, 100000),
    ("data", "max_audio_frames"): (128, 65536),
    ("model", "hidden"): (8, 8192),
    ("model", "audio_layers"): (1, 128),
    ("model", "decoder_layers"): (1, 128),
    ("model", "heads"): (1, 256),
    ("model", "ffn"): (1, 65536),
    ("model", "max_audio_frames"): (128, 65536),
    ("model", "max_tokens"): (128, 32768),
    ("model", "n_mels"): (1, 1024),
    ("model", "audio_stride"): (1, 1024),
    ("train", "batch_size"): (1, 8192),
    ("train", "grad_accum"): (1, 1024),
    ("train", "epochs"): (1, 100000),
    ("train", "warmup_steps"): (0, 10_000_000),
    ("train", "patience"): (0, 100000),
    ("train", "save_every"): (1, 100000),
    ("train", "log_every"): (1, 100000),
    ("train", "num_workers"): (0, 256),
}

_FLOAT_SPECS: dict[tuple[str, str], tuple[float, float]] = {
    ("model", "dropout"): (0.0, 0.999),
    ("train", "lr"): (0.0, 10.0),
    ("train", "weight_decay"): (0.0, 10.0),
    ("train", "grad_clip"): (0.0, 1000.0),
    ("train", "label_smoothing"): (0.0, 1.0),
}

_BOOL_SPECS = {("train", "amp")}


def _int_value(section: str, key: str, value: Any) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{section}.{key} 不能为空")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section}.{key} 必须是整数") from exc
    minimum, maximum = _INT_SPECS[(section, key)]
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError(
            f"{section}.{key} 超出范围 [{minimum}, {maximum}]: {number}"
        )
    return number


def apply_hyperparams(cfg: dict[str, Any], hyperparams: Any) -> None:
    """Merge whitelisted UI overrides into a parsed training config."""
    if hyperparams is None:
        return
    if not isinstance(hyperparams, dict):
        raise ValueError("hyperparams 必须是对象")
    known_sections = {"data", "model", "train"}
    unknown_sections = set(hyperparams) - known_sections
    if unknown_sections:
        raise ValueError(f"未知超参分组: {', '.join(sorted(unknown_sections))}")

    for section in sorted(known_sections):
        values = hyperparams.get(section)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"hyperparams.{section} 必须是对象")
        target = cfg.setdefault(section, {})
        if not isinstance(target, dict):
            raise ValueError(f"配置中的 {section} 不是对象")

        for key, value in values.items():
            if (section, key) in _INT_SPECS:
                if (
                    (section, key) == ("data", "max_windows_per_chart")
                    and (value is None or str(value).strip() == "")
                ):
                    target[key] = None
                else:
                    target[key] = _int_value(section, key, value)
            elif (section, key) in _FLOAT_SPECS:
                if value is None or str(value).strip() == "":
                    raise ValueError(f"{section}.{key} 不能为空")
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{section}.{key} 必须是数字") from exc
                minimum, maximum = _FLOAT_SPECS[(section, key)]
                if not minimum <= number <= maximum:
                    raise ValueError(
                        f"{section}.{key} 超出范围 [{minimum}, {maximum}]: {number}"
                    )
                target[key] = number
            elif (section, key) in _BOOL_SPECS:
                if not isinstance(value, bool):
                    raise ValueError(f"{section}.{key} 必须是布尔值")
                target[key] = value
            elif (section, key) == ("data", "spec_augment_prob"):
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("data.spec_augment_prob 必须是数字") from exc
                if not 0.0 <= number <= 1.0:
                    raise ValueError("data.spec_augment_prob 必须在 [0, 1] 内")
                augment = target.setdefault("spec_augment", {})
                if not isinstance(augment, dict):
                    raise ValueError("配置中的 data.spec_augment 不是对象")
                augment["prob"] = number
            else:
                raise ValueError(f"未知超参数: {section}.{key}")

    model = cfg.get("model") or {}
    hidden = model.get("hidden")
    heads = model.get("heads")
    stride = model.get("audio_stride")
    if isinstance(hidden, int) and isinstance(heads, int) and heads > 0:
        if hidden % heads:
            raise ValueError("model.hidden 必须能被 model.heads 整除")
    if isinstance(stride, int) and stride > 0 and stride & (stride - 1):
        raise ValueError("model.audio_stride 必须是 2 的幂")
    if model.get("max_audio_frames") is not None:
        data_max = (cfg.get("data") or {}).get("max_audio_frames")
        if data_max is not None and int(data_max) > int(model["max_audio_frames"]):
            raise ValueError(
                "data.max_audio_frames 不能大于 model.max_audio_frames"
            )


def read_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ValueError(
            "读取/生成训练配置需要 PyYAML: pip install pyyaml"
        ) from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface YAML/IO errors
        raise ValueError(f"无法读取训练配置 {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("训练配置必须是 YAML 映射")
    return data


def write_config(cfg: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ValueError(
            "生成训练配置需要 PyYAML: pip install pyyaml"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
