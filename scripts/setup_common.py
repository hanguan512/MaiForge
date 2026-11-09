#!/usr/bin/env python3
"""Shared implementation for the Linux and Windows setup scripts.

``setup_linux.py`` and ``setup_win.py`` are thin platform entrypoints; this
module owns dependency detection, the per-venv install state, and the Web UI
launcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
STATE_FILE = VENV_DIR / ".mainote_setup_state.json"

TORCH_VERSION = "2.8.0"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DEFAULT_PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEFAULT_PORT = 8765
PROGRESS_INTERVAL = 10.0
SETUP_STATE_VERSION = 2

REQUIREMENTS_FILES = ("requirements-train.txt", "requirements.txt", "pyproject.toml")


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def run(
    command: list[str],
    *,
    capture: bool = False,
    check: bool = True,
    label: str | None = None,
    heartbeat: float = PROGRESS_INTERVAL,
) -> subprocess.CompletedProcess:
    """Run a command, streaming its output and printing a periodic heartbeat."""
    parts = [str(part) for part in command]
    log("$ " + " ".join(parts))
    if capture:
        return subprocess.run(
            parts,
            cwd=str(REPO_ROOT),
            check=check,
            capture_output=True,
            text=True,
        )

    started = time.monotonic()
    if label:
        log(f"开始: {label}")
    process = subprocess.Popen(parts, cwd=str(REPO_ROOT), text=True)
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(max(heartbeat, 0.1)):
            elapsed = time.monotonic() - started
            log(f"进行中: {label or '命令'}（已用 {elapsed:.0f}s）")

    thread = None
    if label and heartbeat > 0:
        thread = threading.Thread(target=report, daemon=True)
        thread.start()
    try:
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)

    elapsed = time.monotonic() - started
    if check and returncode != 0:
        if label:
            log(f"失败: {label}（退出码 {returncode}，{elapsed:.1f}s）")
        raise subprocess.CalledProcessError(returncode, parts)
    if label:
        if returncode == 0:
            log(f"完成: {label}（{elapsed:.1f}s）")
        else:
            log(f"结束: {label}（退出码 {returncode}，{elapsed:.1f}s）")
    return subprocess.CompletedProcess(parts, returncode)


# ---------------------------------------------------------------------------
# virtual environment


def platform_label(platform: str) -> str:
    return "Linux" if platform == "linux" else "Windows"


def venv_python(platform: str) -> Path:
    if platform == "win":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def python_version(command: list[str]) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [
                *command,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        major, minor = result.stdout.strip().split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def choose_python(platform: str) -> list[str]:
    """Return the command for a Python 3.11+ interpreter."""
    candidates: list[list[str]] = []

    def add(command: list[str]) -> None:
        if command not in candidates:
            candidates.append(command)

    if platform == "win":
        if sys.version_info >= (3, 11):
            add([sys.executable])
        py = shutil.which("py")
        if py:
            for version in ("3.12", "3.11"):
                add([py, f"-{version}"])
        for name in ("python3.12", "python3.11", "python"):
            path = shutil.which(name)
            if path:
                add([path])
    else:
        for name in ("python3.12", "python3.11", "python3"):
            path = shutil.which(name)
            if path:
                add([path])
        add([sys.executable])

    valid = [
        (command, version)
        for command in candidates
        if (version := python_version(command)) is not None and version >= (3, 11)
    ]
    if not valid:
        if platform == "win":
            raise SystemExit(
                "[setup] 需要 Python 3.11+；请安装 Python 3.12（python.org 或 Microsoft Store）。"
            )
        raise SystemExit(
            "[setup] 需要 Python 3.11+；请先安装 python3.12（推荐）或 python3.11。"
        )

    if platform == "linux":
        # Prefer 3.12, then the lowest supported version.
        return min(
            valid,
            key=lambda item: (0 if item[1] == (3, 12) else 1, item[1]),
        )[0]
    # Windows: keep candidate order (current interpreter first, then py -3.12).
    return valid[0][0]


def ensure_virtualenv(platform: str, script_path: Path) -> None:
    if os.environ.get("MAINOTE_SETUP_REEXEC") == "1":
        return
    if in_virtualenv():
        if sys.version_info < (3, 11):
            raise SystemExit(
                f"[setup] 当前虚拟环境是 Python {sys.version.split()[0]}，需要 3.11+。"
            )
        log(f"使用当前虚拟环境: {sys.prefix}")
        return

    target = venv_python(platform)
    if target.is_file():
        version = python_version([str(target)])
        if not version or version < (3, 11):
            raise SystemExit(
                f"[setup] {VENV_DIR} 不是 Python 3.11+ 环境，请删除后重试。"
            )
        log(f"使用已有虚拟环境: {VENV_DIR}")
    else:
        python = choose_python(platform)
        log(f"创建虚拟环境: {VENV_DIR}（{' '.join(python)}）")
        try:
            run([*python, "-m", "venv", str(VENV_DIR)], label="创建虚拟环境")
        except subprocess.CalledProcessError as exc:
            if platform == "win":
                hint = "请确认已安装 Python 3.12，并尝试：py -3.12 -m venv .venv"
            else:
                hint = "Debian/Ubuntu 请先安装：sudo apt-get install -y python3-venv"
            raise SystemExit(f"[setup] 创建虚拟环境失败。{hint}") from exc

    os.environ["MAINOTE_SETUP_REEXEC"] = "1"
    log(f"切换到虚拟环境解释器: {target}")
    if platform == "win":
        # os.execv works on Windows, but subprocess gives clearer exit codes and
        # handles paths with spaces without relying on CreateProcess quoting.
        completed = subprocess.run(
            [str(target), str(script_path), *sys.argv[1:]],
            executable=str(target),
            cwd=str(REPO_ROOT),
            check=False,
        )
        raise SystemExit(completed.returncode)
    os.execv(str(target), [str(target), str(script_path), *sys.argv[1:]])


# ---------------------------------------------------------------------------
# GPU / PyTorch detection


def detect_gpu() -> tuple[bool, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, "未找到 nvidia-smi"
    result = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        detail = (result.stderr or result.stdout or "nvidia-smi 没有返回 GPU").strip()
        return False, detail
    return True, "; ".join(lines)


def torch_info() -> dict | None:
    code = (
        "import json, torch; "
        "print(json.dumps({"
        "'version': torch.__version__, "
        "'cuda': torch.version.cuda, "
        "'available': bool(torch.cuda.is_available()), "
        "'device_count': torch.cuda.device_count()"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def torch_version_tuple(version: str) -> tuple[int, int]:
    parts = version.split("+", 1)[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def plan_torch(info: dict | None, gpu: bool) -> tuple[str, str]:
    """Return ``(action, reason)`` where action is install / warn / ok."""
    if info is None:
        return "install", "未安装 PyTorch"
    version = torch_version_tuple(str(info.get("version", "")))
    if version < (2, 2):
        return "install", f"torch {info.get('version')} 低于 2.2"
    if gpu and info.get("cuda") is None:
        return "install", f"torch {info.get('version')} 是 CPU 版，但检测到 GPU"
    if gpu and not info.get("available"):
        return (
            "warn",
            f"torch {info.get('version')} 检测到 CUDA {info.get('cuda')} 但不可用，"
            "请检查 NVIDIA 驱动；继续使用当前 PyTorch",
        )
    backend = f"CUDA {info.get('cuda')}" if info.get("cuda") else "CPU"
    return "ok", f"torch {info.get('version')} ({backend})"


# ---------------------------------------------------------------------------
# installed-dependency state


def requirements_fingerprint(platform: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"setup-state-v{SETUP_STATE_VERSION}".encode())
    digest.update(f"{sys.version_info.major}.{sys.version_info.minor}".encode())
    digest.update(platform.encode())
    for name in REQUIREMENTS_FILES:
        path = REPO_ROOT / name
        digest.update(name.encode())
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
    return digest.hexdigest()


def read_state() -> dict:
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_state(fingerprint: str, torch_version: str | None = None) -> None:
    data = {
        "schema": SETUP_STATE_VERSION,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sys.platform,
        "fingerprint": fingerprint,
        "torch": torch_version,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp = STATE_FILE.with_suffix(".tmp")
        temp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(STATE_FILE)
    except OSError as exc:
        log(f"警告: 无法写入状态文件 {STATE_FILE}: {exc}")


_MODULE_CHECK = r"""
import importlib.metadata

missing = []
for name in ("numpy", "mutagen", "yaml", "fastapi", "uvicorn", "httpx"):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
try:
    import multipart  # noqa: F401
except Exception:
    try:
        import python_multipart  # noqa: F401
    except Exception:
        missing.append("python-multipart")
try:
    import mainote  # noqa: F401
except Exception:
    missing.append("mainote(import)")
try:
    importlib.metadata.version("mainote")
except Exception:
    missing.append("mainote(dist)")
print(";".join(missing))
"""


def missing_modules() -> list[str]:
    log("检查已安装依赖 ...")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _MODULE_CHECK],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"<依赖检查失败: {exc}>"]
    if result.returncode != 0:
        lines = (result.stderr or result.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else str(result.returncode)
        return [f"<依赖检查失败: {detail}>"]
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return []
    return [item for item in lines[-1].split(";") if item]


def requirements_status(
    fingerprint: str,
    *,
    write_state_if_ready: bool = True,
) -> tuple[bool, str]:
    state = read_state()
    missing = missing_modules()
    if missing:
        return False, "缺少或无法导入: " + ", ".join(missing)
    if not state:
        if write_state_if_ready:
            write_state(fingerprint)
            return True, "依赖已就绪，已写入状态文件"
        return True, "依赖已就绪（检查模式不写状态）"
    if state.get("fingerprint") != fingerprint:
        return False, "依赖清单或 Python 版本有变化"
    return True, "依赖已安装且清单未变化"


# ---------------------------------------------------------------------------
# dependency installation


def pip_status() -> tuple[bool, str]:
    """Return whether the venv's pip can run, plus a short diagnostic."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return True, result.stdout.strip() or "pip"
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return False, lines[-1] if lines else f"退出码 {result.returncode}"


def _run_ensurepip() -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True, result.stdout.strip()
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return False, lines[-1] if lines else f"退出码 {result.returncode}"


def _pip_artifacts() -> list[Path]:
    """Return the installed ``pip`` package and dist-info directories."""
    purelib = Path(sysconfig.get_paths()["purelib"])
    paths: list[Path] = []
    package = purelib / "pip"
    if package.exists():
        paths.append(package)
    paths.extend(sorted(purelib.glob("pip-*.dist-info")))
    return paths


def ensure_pip(platform: str) -> None:
    """Self-heal a broken pip inside the venv before any install commands.

    A venv copied between machines, an interrupted ``pip install -U pip`` or a
    full disk can leave pip unable to import its vendored packages. In that
    state ``python -m pip install ...`` fails immediately, so bootstrap the
    bundled pip wheel with ``ensurepip`` first. If the stale dist-info makes
    ``ensurepip`` skip the reinstall, move the broken package aside to force a
    clean bootstrap, restoring it if the repair fails.
    """
    healthy, detail = pip_status()
    if healthy:
        return
    log(f"虚拟环境内的 pip 不可用（{detail}），尝试用 ensurepip 修复 ...")
    ok, error = _run_ensurepip()
    healthy, repaired = pip_status()
    if healthy:
        log(f"pip 已修复: {repaired}")
        return

    log("ensurepip 未能替换现有 pip，改为强制重新安装 ...")
    backup = Path(sys.prefix) / ".mainote-pip-backup"
    moved: list[tuple[Path, Path]] = []
    success = False
    try:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        backup.mkdir(parents=True)
        for source in _pip_artifacts():
            target = backup / source.name
            shutil.move(str(source), str(target))
            moved.append((source, target))
        ok, error = _run_ensurepip()
        healthy, repaired = pip_status()
        success = healthy
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if not success:
            for source, target in moved:
                if target.exists() and not source.exists():
                    shutil.move(str(target), str(source))
        shutil.rmtree(backup, ignore_errors=True)

    if success:
        log(f"pip 已修复: {repaired}")
        return

    if platform == "win":
        hint = "请删除 .venv 后重新运行 scripts\\setup_win.py。"
    else:
        hint = "请删除 .venv 后重新运行 scripts/setup_linux.py。"
    raise SystemExit(f"[setup] pip 修复失败：{error}。{hint}")


def install_dependencies(
    args: argparse.Namespace,
    *,
    platform: str,
    gpu: bool,
    info: dict | None,
    fingerprint: str,
    deps_ready: bool,
    deps_detail: str,
) -> None:
    requirements = REPO_ROOT / "requirements-train.txt"
    if not requirements.is_file():
        raise SystemExit(f"[setup] 缺少 {requirements}")

    mirror_args = ["-i", args.pip_index] if args.pip_index else []
    action, reason = plan_torch(info, gpu)
    if action == "install" or args.force_install or not deps_ready:
        ensure_pip(platform)
    if action == "install":
        index_url = args.cuda_index_url if gpu else args.cpu_index_url
        backend = "CUDA 12.8" if gpu else "CPU"
        log(f"安装 PyTorch {args.torch_version}（{backend}）：{reason}")
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                f"torch=={args.torch_version}",
                "--index-url",
                index_url,
            ],
            label=f"安装 PyTorch {args.torch_version}（{backend}）",
            heartbeat=args.progress_interval,
        )
        info = torch_info()
    elif action == "warn":
        log(f"警告: {reason}")
    else:
        log(f"PyTorch 已就绪: {reason}")

    if deps_ready and not args.force_install:
        log(f"依赖已就绪，跳过 pip install（{deps_detail}）。")
        write_state(fingerprint, (info or {}).get("version"))
        return

    log(f"安装 Web/音频/训练依赖：{deps_detail}")
    run(
        [sys.executable, "-m", "pip", "install", "-U", "pip", *mirror_args],
        label="升级 pip",
        heartbeat=args.progress_interval,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
            *mirror_args,
        ],
        label="安装 Web/音频/训练依赖（可能需要几分钟）",
        heartbeat=args.progress_interval,
    )
    write_state(fingerprint, (info or {}).get("version"))


# ---------------------------------------------------------------------------
# ffmpeg


def find_ffmpeg(platform: str) -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if platform != "win":
        return None

    candidates: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.extend(
            Path(local).glob(
                "Microsoft/WinGet/Packages/Gyan.FFmpeg_*/**/bin/ffmpeg.exe"
            )
        )
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if base:
            candidates.extend(Path(base).glob("ffmpeg*/bin/ffmpeg.exe"))
            candidates.append(Path(base) / "ffmpeg" / "bin" / "ffmpeg.exe")
    choco = os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey")
    candidates.append(Path(choco) / "bin" / "ffmpeg.exe")
    candidates.append(Path(r"C:\ffmpeg\bin\ffmpeg.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _add_to_path(executable: str) -> None:
    directory = str(Path(executable).parent)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if directory not in entries:
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


def ensure_ffmpeg(args: argparse.Namespace, platform: str) -> bool:
    found = find_ffmpeg(platform)
    if found:
        log(f"ffmpeg: {found}")
        _add_to_path(found)
        return True

    log("警告: 未找到 ffmpeg；音频分析、预览和带对齐的导入会失败。")
    if args.check:
        log("--check: 跳过 ffmpeg 安装。")
        return False
    if not args.install_ffmpeg:
        if platform == "win":
            log(
                "提示: 重新运行并加 --install-ffmpeg，或手动执行 "
                "winget install --id Gyan.FFmpeg -e"
            )
        else:
            log(
                "提示: 重新运行并加 --install-ffmpeg，或手动执行 "
                "sudo apt-get install -y ffmpeg"
            )
        return False

    if platform == "win":
        winget = shutil.which("winget")
        choco = shutil.which("choco")
        if winget:
            run(
                [
                    winget,
                    "install",
                    "--id",
                    "Gyan.FFmpeg",
                    "-e",
                    "--accept-source-agreements",
                    "--accept-package-agreements",
                    "--silent",
                ],
                check=False,
                label="安装 ffmpeg（winget）",
                heartbeat=args.progress_interval,
            )
        elif choco:
            run(
                [choco, "install", "ffmpeg", "-y"],
                check=False,
                label="安装 ffmpeg（choco）",
                heartbeat=args.progress_interval,
            )
        else:
            log("未找到 winget 或 choco，请手动安装 ffmpeg：")
            log("  winget install --id Gyan.FFmpeg -e")
            log(
                "  或从 https://www.gyan.dev/ffmpeg/builds/ 下载，"
                "并把 bin 目录加入 PATH"
            )
            return False
    else:
        if not shutil.which("apt-get"):
            log("未找到 apt-get，请手动安装 ffmpeg。")
            return False
        prefix = [] if os.geteuid() == 0 else (["sudo"] if shutil.which("sudo") else [])
        if not prefix and os.geteuid() != 0:
            log("需要 root 或 sudo 才能安装 ffmpeg。")
            return False
        run(
            [*prefix, "apt-get", "update"],
            check=False,
            label="apt-get update",
            heartbeat=args.progress_interval,
        )
        run(
            [*prefix, "apt-get", "install", "-y", "ffmpeg"],
            check=False,
            label="安装 ffmpeg",
            heartbeat=args.progress_interval,
        )

    found = find_ffmpeg(platform)
    if found:
        log(f"ffmpeg 安装完成: {found}")
        _add_to_path(found)
        return True
    log("ffmpeg 仍未找到；请重开终端或手动把 ffmpeg 的 bin 目录加入 PATH。")
    return False


# ---------------------------------------------------------------------------
# web UI


def find_free_port(host: str, preferred: int) -> int:
    bind_host = "0.0.0.0" if host in ("0.0.0.0", "") else host
    last = min(preferred + 19, 65535)
    for port in range(preferred, last + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
            except (OSError, OverflowError):
                continue
            return port
    raise SystemExit(f"[setup] {preferred}-{last} 端口都被占用。")


def local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def web_urls(host: str, port: int) -> list[str]:
    if host in ("0.0.0.0", "::"):
        urls = [f"http://127.0.0.1:{port}"]
        ip = local_ip()
        if ip != "127.0.0.1":
            urls.append(f"http://{ip}:{port}")
        return urls
    if host in ("127.0.0.1", "localhost"):
        return [f"http://127.0.0.1:{port}"]
    return [f"http://{host}:{port}"]


def start_webui(args: argparse.Namespace) -> int:
    port = find_free_port(args.host, args.port)
    log(f"Web UI 端口: {port}")
    for url in web_urls(args.host, port):
        log(f"访问地址: {url}")

    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-m",
        "mainote",
        "web",
        "--host",
        args.host,
        "--port",
        str(port),
        "--no-browser",
    ]
    try:
        return subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            check=False,
        ).returncode
    except KeyboardInterrupt:
        return 0


# ---------------------------------------------------------------------------
# CLI


def build_parser(platform: str) -> argparse.ArgumentParser:
    label = platform_label(platform)
    parser = argparse.ArgumentParser(
        description=(
            f"{label} 下检测/安装虚拟环境、PyTorch 和 Web 依赖，"
            "然后启动 mainote Web UI。"
        )
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检测环境，不安装依赖、不启动 Web UI",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="跳过依赖检查与安装，直接启动（用于已配置好的环境）",
    )
    parser.add_argument(
        "--force-install",
        action="store_true",
        help="忽略依赖状态文件，强制重新运行 pip install",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="安装完成后不启动 Web UI",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="即使检测到 GPU，也使用 CPU 版 PyTorch",
    )
    parser.add_argument(
        "--install-ffmpeg",
        action="store_true",
        help=(
            "ffmpeg 缺失时尝试自动安装"
            "（Linux 用 apt-get，Windows 用 winget/choco）"
        ),
    )
    parser.add_argument("--torch-version", default=TORCH_VERSION)
    parser.add_argument("--cuda-index-url", default=CUDA_INDEX_URL)
    parser.add_argument("--cpu-index-url", default=CPU_INDEX_URL)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=PROGRESS_INTERVAL,
        help="长任务心跳间隔秒数；0 表示关闭（默认 10）",
    )
    parser.add_argument(
        "--pip-index",
        default=os.environ.get("MAINOTE_PIP_INDEX") or DEFAULT_PIP_INDEX,
        help=(
            "普通 pip 包使用的镜像，默认清华源；传空字符串可回到 PyPI。"
            "PyTorch 始终使用官方 CUDA/CPU wheel 索引"
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    platform: str,
    script_path: Path,
) -> int:
    args = build_parser(platform).parse_args(argv)
    started = time.monotonic()

    if platform == "linux" and not sys.platform.startswith("linux"):
        raise SystemExit("[setup] setup_linux.py 只能在 Linux 上运行。")
    if platform == "win" and os.name != "nt":
        raise SystemExit("[setup] setup_win.py 只能在 Windows 上运行。")
    if not 1 <= args.port <= 65535:
        raise SystemExit("[setup] --port 必须在 1-65535 之间。")
    if args.progress_interval < 0:
        raise SystemExit("[setup] --progress-interval 不能为负数。")

    ensure_virtualenv(platform, script_path)

    gpu, gpu_detail = detect_gpu()
    if args.force_cpu and gpu:
        log("--force-cpu: 忽略检测到的 GPU，使用 CPU 版 PyTorch")
        gpu = False
        gpu_detail = "已强制使用 CPU"

    log(f"Python: {sys.version.split()[0]} ({sys.executable})")
    log(f"GPU: {'检测到 ' + gpu_detail if gpu else '未检测到（' + gpu_detail + '）'}")
    log(f"pip 镜像: {args.pip_index or 'PyPI（未指定镜像）'}")
    if not gpu:
        log("提示: 没有可用 GPU，训练和推理会使用 CPU。")
    ensure_ffmpeg(args, platform)

    info = torch_info()
    action, reason = plan_torch(info, gpu)
    log(f"PyTorch: {reason}")

    fingerprint = requirements_fingerprint(platform)
    if args.skip_install:
        deps_ready = None
        deps_detail = "已跳过（--skip-install）"
        log(f"依赖检查: {deps_detail}")
    else:
        deps_ready, deps_detail = requirements_status(
            fingerprint,
            write_state_if_ready=not args.check,
        )
        log(f"依赖检查: {'已就绪' if deps_ready else '需要安装'}（{deps_detail}）")

    if args.check:
        if action == "install":
            backend = "CUDA 12.8" if gpu else "CPU"
            log(f"安装计划: torch=={args.torch_version}（{backend}）")
        else:
            log("安装计划: 无需安装 PyTorch")
        if args.skip_install:
            log("依赖计划: 已跳过（--skip-install）")
        elif args.force_install:
            log("依赖计划: --force-install，将重新运行 pip install")
        elif deps_ready:
            log("依赖计划: 已就绪，将跳过 pip install")
        else:
            log("依赖计划: 需要安装")
        if not args.no_start:
            log(f"Web UI 计划端口: {find_free_port(args.host, args.port)}")
        log(f"检测完成（{time.monotonic() - started:.1f}s）。")
        return 0

    if not args.skip_install:
        install_dependencies(
            args,
            platform=platform,
            gpu=gpu,
            info=info,
            fingerprint=fingerprint,
            deps_ready=bool(deps_ready),
            deps_detail=deps_detail,
        )
        info = torch_info()
        if info is not None:
            log(
                "安装后 PyTorch: "
                f"torch {info.get('version')} "
                f"({('CUDA ' + str(info.get('cuda'))) if info.get('cuda') else 'CPU'})"
            )
    elif action == "install":
        log("--skip-install: 跳过 PyTorch 安装。")

    if args.no_start:
        log(
            "已跳过 Web UI 启动（--no-start）；"
            f"总用时 {time.monotonic() - started:.1f}s。"
        )
        return 0
    log(f"环境准备完成（{time.monotonic() - started:.1f}s），启动 Web UI ...")
    return start_webui(args)
