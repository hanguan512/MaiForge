"""AWMC-MaiArchive (download.wmc.pub) catalogue and chart downloads.

The site documents a public JSON API (``/api/v1/charts``) and free single-chart
downloads; no guard cookie is needed for these endpoints.  We only ever talk to
``download.wmc.pub`` (redirects may land on its ``oss.awmc.cc`` object store),
never to a user-supplied URL.

Typical use::

    results = search_charts("apollo", limit=10)
    chart = fetch_chart("011661")
    download_chart_zip(chart, Path("song.zip"))

Downloads are user-initiated; callers should keep request rates modest.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WMC_BASE = "https://download.wmc.pub"
WMC_API = f"{WMC_BASE}/api/v1/charts"
# Fixed download hosts only.  ``download.wmc.pub`` redirects chart zips to its
# object store; both ``oss.awmc.cc`` and ``when.axql.love`` are hosts the
# official API has pointed at over time.  The browser can never supply a URL.
WMC_HOSTS = {"download.wmc.pub", "oss.awmc.cc", "when.axql.love"}
USER_AGENT = "mainote/0.1 (+local dataset tool)"
MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024  # 1 GiB
MAX_IMAGE_BYTES = 16 * 1024 * 1024  # 16 MiB
SHORTID_RE = re.compile(r"^[0-9]{5,7}$")
IMAGE_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\.(?:png|jpe?g|webp|gif)$", re.IGNORECASE
)


class WmcError(RuntimeError):
    """Raised when the catalogue or a download cannot be used."""


@dataclass
class WmcClient:
    """Small stdlib-only client with an injectable base URL for tests."""

    base_url: str = WMC_BASE
    timeout: float = 30.0
    user_agent: str = USER_AGENT
    # Tests point ``base_url`` at a local HTTP server and override this set.
    allowed_hosts: frozenset[str] | None = None

    # -- HTTP ------------------------------------------------------------
    def _request(self, url: str, *, timeout: float | None = None):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json, application/zip, */*",
            },
        )
        return urllib.request.urlopen(request, timeout=timeout or self.timeout)

    def _json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        if params:
            clean = {key: value for key, value in params.items() if value not in (None, "")}
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        last_error: Exception | None = None
        payload: bytes | None = None
        for attempt in range(3):
            try:
                with self._request(url) as response:
                    payload = response.read()
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    raise WmcError(
                        f"AWMC-MaiArchive 请求失败: HTTP {exc.code}"
                    ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            if attempt < 2:
                _retry_sleep(attempt, last_error, 1.5)
        if payload is None:
            raise WmcError(f"无法访问 AWMC-MaiArchive: {last_error}") from last_error
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WmcError("AWMC-MaiArchive 返回了无法解析的数据") from exc
        if not isinstance(data, dict):
            raise WmcError("AWMC-MaiArchive 返回了意外的数据格式")
        return data

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
        if page < 1:
            raise ValueError("page 必须大于 0")
        if not 1 <= limit <= 500:
            raise ValueError("limit 必须在 1 到 500 之间")
        if chart_type is not None:
            chart_type = str(chart_type).lower()
            if chart_type not in {"st", "dx", "utg"}:
                raise ValueError("chart_type 必须是 st/dx/utg")
        if sort not in {"id", "title", "version", "popular"}:
            raise ValueError("sort 必须是 id/title/version/popular")
        data = self._json(
            "/api/v1/charts",
            {
                "q": query.strip() or None,
                "version": version,
                "type": chart_type,
                "genre": genre,
                "sort": sort,
                "page": page,
                "limit": limit,
            },
        )
        charts = data.get("charts")
        if not isinstance(charts, list):
            raise WmcError("AWMC-MaiArchive 没有返回曲目列表")
        data["charts"] = [chart for chart in charts if isinstance(chart, dict)]
        return data

    def chart(self, shortid: str) -> dict[str, Any]:
        shortid = str(shortid).strip()
        if not SHORTID_RE.match(shortid):
            raise ValueError("谱面 ID 必须是 5-7 位数字")
        data = self._json(f"/api/v1/charts/{shortid}")
        chart = data.get("chart")
        if not isinstance(chart, dict):
            raise WmcError(f"未找到谱面 {shortid}")
        return chart

    # -- covers ----------------------------------------------------------
    def image_url(self, shortid: str, image: str) -> str:
        """Build the fixed-host URL for a chart's ``bg.png``-style cover."""
        shortid = str(shortid).strip()
        if not SHORTID_RE.match(shortid):
            raise ValueError("谱面 ID 必须是 5-7 位数字")
        name = str(image or "").strip()
        if not IMAGE_NAME_RE.match(name):
            raise ValueError("非法的封面文件名")
        return (
            f"{self.base_url.rstrip('/')}/s/{shortid}/"
            f"{urllib.parse.quote(name, safe='')}"
        )

    def download_image(
        self,
        shortid: str,
        image: str,
        target: Path,
        *,
        timeout: float = 60.0,
    ) -> Path:
        """Download a cover image (the site's ``bg.png``) to ``target``."""
        url = self.image_url(shortid, image)
        _ensure_wmc_url(url, self.allowed_hosts)

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            total = 0
            try:
                with self._request(url, timeout=timeout) as response:
                    _ensure_wmc_url(response.geturl(), self.allowed_hosts)
                    content_type = (
                        (response.headers.get("Content-Type") or "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if content_type and not content_type.startswith("image/"):
                        raise WmcError(
                            f"封面返回了非图片内容: {content_type}"
                        )
                    with temporary.open("wb") as output:
                        while True:
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > MAX_IMAGE_BYTES:
                                raise WmcError("封面文件超过 16 MB,已中止")
                            output.write(chunk)
                if total == 0:
                    raise WmcError("封面文件为空")
                os.replace(temporary, target)
                return target
            except WmcError as exc:
                last_error = exc
                if "超过 16 MB" in str(exc):
                    raise
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    raise WmcError(f"下载封面失败: HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            finally:
                temporary.unlink(missing_ok=True)
            if attempt < 2:
                _retry_sleep(attempt, last_error, 1.5)
        raise WmcError(f"下载封面失败: {last_error}") from last_error

    # -- download --------------------------------------------------------
    def download_zip(
        self,
        chart: dict[str, Any],
        target: Path,
        *,
        include_video: bool = False,
        timeout: float = 600.0,
    ) -> Path:
        """Download a chart zip (no background video by default) to ``target``."""
        links = chart.get("download")
        key = "zip" if include_video else "zipNoVideo"
        url = links.get(key) if isinstance(links, dict) else None
        if not url:
            folder = chart.get("folderName")
            version = chart.get("versionFolder")
            if not folder or not version:
                raise WmcError("谱面信息缺少下载地址")
            url = (
                f"{self.base_url.rstrip('/')}/api/download/"
                f"{urllib.parse.quote(str(version), safe='')}/"
                f"{urllib.parse.quote(str(folder), safe='')}/zip"
                + ("" if include_video else "?novideo=1")
            )
        _ensure_wmc_url(url, self.allowed_hosts)

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.part")
        last_error: Exception | None = None
        for attempt in range(3):
            total = 0
            try:
                with self._request(url, timeout=timeout) as response:
                    _ensure_wmc_url(response.geturl(), self.allowed_hosts)
                    with temporary.open("wb") as output:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > MAX_DOWNLOAD_BYTES:
                                raise WmcError("谱面包超过 1 GB,已中止")
                            output.write(chunk)
                with zipfile.ZipFile(temporary) as archive:
                    if archive.testzip() is not None:
                        raise WmcError("下载的谱面包已损坏")
                os.replace(temporary, target)
                return target
            except WmcError as exc:
                last_error = exc
                if "超过 1 GB" in str(exc):
                    raise
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    raise WmcError(f"下载谱面包失败: HTTP {exc.code}") from exc
            except zipfile.BadZipFile as exc:
                last_error = exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
            finally:
                temporary.unlink(missing_ok=True)
            if attempt < 2:
                _retry_sleep(attempt, last_error, 2.0)
        raise WmcError(f"下载谱面包失败: {last_error}") from last_error


def _retry_sleep(attempt: int, exc: Exception | None, base: float) -> None:
    """Back off between retries and honour ``Retry-After`` on HTTP 429.

    Bulk imports routinely hit the archive's rate limit; retrying a few seconds
    later just burns another request, so wait noticeably longer for 429.
    """
    delay = base * (attempt + 1)
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        delay = max(delay, 5.0 * (attempt + 1))
        header = exc.headers.get("Retry-After") if exc.headers else None
        try:
            delay = max(delay, float(header))
        except (TypeError, ValueError):
            pass
    time.sleep(min(delay, 60.0))


def _ensure_wmc_url(url: str, hosts: frozenset[str] | None = None) -> None:
    parsed = urllib.parse.urlparse(url)
    allowed = hosts or WMC_HOSTS
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in allowed:
        raise WmcError(f"拒绝下载非 AWMC 地址: {url}")


def search_charts(
    query: str = "",
    *,
    version: str | int | None = None,
    chart_type: str | None = None,
    genre: str | None = None,
    sort: str = "id",
    page: int = 1,
    limit: int = 30,
    client: WmcClient | None = None,
) -> dict[str, Any]:
    return (client or WmcClient()).search(
        query,
        version=version,
        chart_type=chart_type,
        genre=genre,
        sort=sort,
        page=page,
        limit=limit,
    )


def fetch_chart(shortid: str, *, client: WmcClient | None = None) -> dict[str, Any]:
    return (client or WmcClient()).chart(shortid)


def download_chart_zip(
    chart: dict[str, Any],
    target: Path,
    *,
    include_video: bool = False,
    client: WmcClient | None = None,
) -> Path:
    return (client or WmcClient()).download_zip(
        chart, target, include_video=include_video
    )


def polite_sleep(seconds: float = 1.0) -> None:
    """Separate catalogue downloads so the free service is not hammered."""
    if seconds > 0:
        time.sleep(seconds)
