"""Filesystem configuration and path validation for the local web UI.

The web UI only ever reads files under explicit roots and only starts the
documented ``mainote`` CLI entry points.  Everything defaults to localhost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Roots scanned by the browse page.  ``data/processed`` is the canonical
# location; the demo and smoke-test outputs are included when present so a
# fresh checkout still has something to look at.
DEFAULT_DATA_ROOTS = (
    "data/processed",
    "out/demo/processed",
    "out/audio-fm/dataset/processed",
)


def _env_paths(name: str) -> list[Path]:
    raw = os.environ.get(name, "")
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def _resolve(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


@dataclass
class WebSettings:
    """Resolved roots for one app instance.

    Tests construct ``WebSettings(repo_root=tmp_path)`` to run the app against
    a temporary tree without touching the real repository.
    """

    repo_root: Path = REPO_ROOT
    data_roots: list[Path] = field(default_factory=list)
    runs_root: Path | None = None
    out_root: Path | None = None
    config_root: Path | None = None
    static_dir: Path | None = None
    majdataview_dir: Path | None = None

    def __post_init__(self) -> None:
        self.repo_root = _resolve(self.repo_root, Path.cwd())
        if not self.data_roots:
            configured = _env_paths("MAINOTE_WEB_DATA_ROOTS")
            if configured:
                self.data_roots = [_resolve(path, self.repo_root) for path in configured]
            else:
                self.data_roots = [
                    _resolve(Path(relative), self.repo_root)
                    for relative in DEFAULT_DATA_ROOTS
                ]
        else:
            self.data_roots = [_resolve(path, self.repo_root) for path in self.data_roots]
        self.runs_root = _resolve(self.runs_root or Path("runs"), self.repo_root)
        self.out_root = _resolve(self.out_root or Path("out"), self.repo_root)
        self.config_root = _resolve(
            self.config_root or Path("configs"), self.repo_root
        )
        self.static_dir = _resolve(
            self.static_dir or Path("src/mainote/webapp/static"), self.repo_root
        )
        self.majdataview_dir = _resolve(
            self.majdataview_dir or Path("src/mainote/webapp/majdataview"),
            self.repo_root,
        )

    # -- helpers ---------------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.out_root / "webui" / "uploads"

    @property
    def job_dir(self) -> Path:
        return self.out_root / "webui" / "jobs"

    def rel(self, path: Path | str) -> str:
        try:
            return Path(path).resolve().relative_to(self.repo_root).as_posix()
        except (OSError, ValueError):
            return str(path)

    def resolve_under(
        self,
        value: str | Path,
        roots: list[Path] | tuple[Path, ...],
        *,
        label: str = "路径",
        must_exist: bool = False,
    ) -> Path:
        raw = Path(str(value)).expanduser()
        if not raw.is_absolute():
            raw = self.repo_root / raw
        try:
            path = raw.resolve()
        except OSError as exc:
            raise ValueError(f"{label}无法解析: {value}") from exc
        if must_exist and not path.exists():
            raise ValueError(f"{label}不存在: {path}")
        for root in roots:
            try:
                path.relative_to(Path(root).resolve())
            except ValueError:
                continue
            return path
        allowed = ", ".join(self.rel(root) for root in roots)
        raise ValueError(f"{label}不在允许目录内({allowed}): {path}")

    def data_root_by_value(self, value: str | None) -> Path:
        """Map a root query value (repo-relative or absolute) to a known root."""
        if not value:
            for root in self.data_roots:
                if root.is_dir():
                    return root
            return self.data_roots[0]
        resolved = _resolve(Path(value), self.repo_root)
        for root in self.data_roots:
            if root == resolved:
                return root
        raise ValueError(f"未知的数据目录: {value}")
