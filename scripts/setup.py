#!/usr/bin/env python3
"""One-shot Linux bootstrap for the MaiForge Web UI.

Usage (from the repository root):

    python3 scripts/setup.py
    python3 scripts/setup.py --check          # detect only, install/start nothing
    python3 scripts/setup.py --port 9000 --host 0.0.0.0

What it does:

1. Re-executes itself inside ``.venv``, creating it with Python 3.12 when
   available (3.11+ is accepted).
2. Detects an NVIDIA GPU through ``nvidia-smi``.
3. Checks the installed PyTorch build and installs a Linux wheel when needed:
   ``torch==2.8.0`` from the CUDA 12.8 index on a GPU machine, otherwise from
   the CPU index.
4. Installs ``requirements-train.txt`` (Web + audio + training extras) from
   the Tsinghua PyPI mirror by default; override with ``--pip-index``.
5. Starts ``mainote web`` on the first free port and prints the URL.

This file deliberately lives in ``scripts/`` rather than the repository root.
A root-level ``setup.py`` would be executed by setuptools as packaging
metadata during ``pip install -e .``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"

TORCH_VERSION = "2.8.0"
CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
DEFAULT_PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
DEFAULT_PORT = 8765
PROGRESS_INTERVAL = 10.0


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
        log(f"完成: {label}（{elapsed:.1f}s）")
    return subprocess.CompletedProcess(parts, returncode)


# ---------------------------------------------------------------------------
# virtual environment


def in_virtualenv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def python_version(executable: str) -> tuple[int, int] | None:
    try:
        result = subprocess.run(
            [
                executable,
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


def choose_python() -> str:
    """Prefer Python 3.12, then the oldest available 3.11+ interpreter."""
    candidates: list[tuple[str, tuple[int, int]]] = []
    seen: set[str] = set()
    for name in ("python3.12", "python3.11", "python3"):
        path = shutil.which(name)
        if not path or path in seen:
            continue
        seen.add(path)
        version = python_version(path)
        if version and version >= (3, 11):
            candidates.append((path, version))
    if sys.executable not in seen:
        version = python_version(sys.executable)
        if version and version >= (3, 11):
            candidates.append((sys.executable, version))
    if not candidates:
        raise SystemExit(
            "[setup] 需要 Python 3.11+；请先安装 python3.12（推荐）或 python3.11。"
        )
    # 3.12 first; otherwise the lowest supported version.
    return min(
        candidates,
        key=lambda item: (0 if item[1] == (3, 12) else 1, item[1]),
    )[0]


def ensure_virtualenv() -> None:
    if os.environ.get("MAINOTE_SETUP_REEXEC") == "1":
        return
    if in_virtualenv():
        if sys.version_info < (3, 11):
            raise SystemExit(
                f"[setup] 当前虚拟环境是 Python {sys.version.split()[0]}，需要 3.11+。"
            )
        log(f"使用当前虚拟环境: {sys.prefix}")
        return

    if VENV_PYTHON.is_file():
        version = python_version(str(VENV_PYTHON))
        if not version or version < (3, 11):
            raise SystemExit(
                f"[setup] {VENV_DIR} 不是 Python 3.11+ 环境，请删除后重试。"
            )
        log(f"使用已有虚拟环境: {VENV_DIR}")
    else:
        python = choose_python()
        log(f"创建虚拟环境: {VENV_DIR}（{python}）")
        try:
            run([python, "-m", "venv", str(VENV_DIR)], label="创建虚拟环境")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                "[setup] 创建虚拟环境失败。Debian/Ubuntu 请先安装："
                "sudo apt-get install -y python3-venv"
            ) from exc

    os.environ["MAINOTE_SETUP_REEXEC"] = "1"
    log(f"切换到虚拟环境解释器: {VENV_PYTHON}")
    os.execv(
        str(VENV_PYTHON),
        [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )


# ---------------------------------------------------------------------------
# GPU / PyTorch detection


def detect_gpu() -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, f"当前平台 {sys.platform} 不是 Linux"
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
# dependency installation


def install_dependencies(args: argparse.Namespace, gpu: bool, info: dict | None) -> None:
    requirements = REPO_ROOT / "requirements-train.txt"
    if not requirements.is_file():
        raise SystemExit(f"[setup] 缺少 {requirements}")

    mirror_args = ["-i", args.pip_index] if args.pip_index else []
    log("升级 pip")
    run(
        [sys.executable, "-m", "pip", "install", "-U", "pip", *mirror_args],
        label="升级 pip",
        heartbeat=args.progress_interval,
    )

    action, reason = plan_torch(info, gpu)
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
    elif action == "warn":
        log(f"警告: {reason}")
    else:
        log(f"PyTorch 已就绪: {reason}")

    log("安装 Web/音频/训练依赖（requirements-train.txt）")
    # torch is already installed above; pip keeps it because it satisfies
    # the project's ``torch>=2.2`` requirement.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "检测/安装 Linux 下的虚拟环境、PyTorch 和 Web 依赖，"
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
        help="跳过 pip 安装，直接启动（用于已配置好的环境）",
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
        help="ffmpeg 缺失时尝试用 apt-get 安装（需要 root 或 sudo）",
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


def ensure_ffmpeg(args: argparse.Namespace) -> bool:
    if shutil.which("ffmpeg"):
        log(f"ffmpeg: {shutil.which('ffmpeg')}")
        return True
    log("警告: 未找到 ffmpeg；音频分析、预览和带对齐的导入会失败。")
    if args.check:
        log("--check: 跳过 ffmpeg 安装。")
        return False
    if not args.install_ffmpeg:
        log("提示: 重新运行并加 --install-ffmpeg，或手动执行 sudo apt-get install -y ffmpeg")
        return False
    if not shutil.which("apt-get"):
        log("未找到 apt-get，请手动安装 ffmpeg。")
        return False
    prefix = [] if os.geteuid() == 0 else (["sudo"] if shutil.which("sudo") else [])
    if not prefix and os.geteuid() != 0:
        log("需要 root 或 sudo 才能安装 ffmpeg。")
        return False
    try:
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
    except OSError as exc:
        log(f"安装 ffmpeg 失败: {exc}")
        return False
    installed = bool(shutil.which("ffmpeg"))
    log("ffmpeg 安装完成" if installed else "ffmpeg 仍未找到，请手动安装。")
    return installed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.monotonic()
    if not sys.platform.startswith("linux"):
        raise SystemExit(
            "[setup] 此脚本只支持 Linux；PyTorch wheel 按 Linux CUDA 12.8/CPU 选择。"
        )
    if not 1 <= args.port <= 65535:
        raise SystemExit("[setup] --port 必须在 1-65535 之间。")
    if args.progress_interval < 0:
        raise SystemExit("[setup] --progress-interval 不能为负数。")

    ensure_virtualenv()

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
    ensure_ffmpeg(args)

    info = torch_info()
    action, reason = plan_torch(info, gpu)
    log(f"PyTorch: {reason}")

    if args.check:
        if action == "install":
            backend = "CUDA 12.8" if gpu else "CPU"
            log(f"安装计划: torch=={args.torch_version}（{backend}）")
        else:
            log("安装计划: 无需安装 PyTorch")
        if not args.no_start:
            log(f"Web UI 计划端口: {find_free_port(args.host, args.port)}")
        log(f"检测完成（{time.monotonic() - started:.1f}s）。")
        return 0

    if not args.skip_install:
        install_dependencies(args, gpu, info)
        info = torch_info()
        if info is not None:
            log(
                "安装后 PyTorch: "
                f"torch {info.get('version')} "
                f"({('CUDA ' + str(info.get('cuda'))) if info.get('cuda') else 'CPU'})"
            )

    if args.no_start:
        log(
            "已跳过 Web UI 启动（--no-start）；"
            f"总用时 {time.monotonic() - started:.1f}s。"
        )
        return 0
    log(f"环境准备完成（{time.monotonic() - started:.1f}s），启动 Web UI ...")
    return start_webui(args)


if __name__ == "__main__":
    raise SystemExit(main())
