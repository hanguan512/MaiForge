"""Tests for the AWMC-MaiArchive client and zip import core."""

import io
import json
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from mainote.dataset.wmc import WmcClient, WmcError
from mainote.dataset.zipimport import import_zip_archive

MAIDATA = """\
&title=Online Demo
&artist=Tester
&wholebpm=120
&inote_5=(120){4}8>2[4:1]*s4<5[4:1],6h[1:0],2-6[4:3],E
"""

_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-cover"


def _zip_bytes(*, maidata: str = MAIDATA, background: bool = True) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("song/maidata.txt", maidata)
        archive.writestr("song/track.mp3", b"fake-mp3")
        if background:
            archive.writestr("song/bg.png", b"\x89PNG\r\n\x1a\n")
    return buffer.getvalue()


class _Handler(BaseHTTPRequestHandler):
    flaky_attempts = 0

    def log_message(self, *args):  # noqa: D401 - silence test server
        pass

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        host = self.headers.get("Host", "127.0.0.1")
        path = self.path.split("?", 1)[0]
        if path == "/api/v1/charts":
            payload = {
                "total": 1,
                "matched": 1,
                "page": 1,
                "pages": 1,
                "limit": 30,
                "count": 1,
                "charts": [{"shortid": "012345", "title": "Online Demo"}],
                "available": {
                    "versions": {"23": "PRiSM"},
                    "genres": ["POPS＆アニメ"],
                },
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if path == "/api/v1/charts/012345":
            payload = {
                "chart": {
                    "shortid": "012345",
                    "title": "Online Demo",
                    "version": "PRiSM",
                    "versionFolder": "PRiSM",
                    "folderName": "Online Demo [DX]",
                    "bpm": 120,
                    "download": {
                        "zipNoVideo": f"http://{host}/files/chart.zip",
                        "zip": f"http://{host}/files/chart.zip",
                    },
                }
            }
            self._send(200, json.dumps(payload).encode(), "application/json")
            return
        if path == "/files/chart.zip":
            self._send(200, _zip_bytes(), "application/zip")
            return
        if path == "/files/bad.zip":
            self._send(200, b"not a zip", "application/zip")
            return
        if path == "/files/retry.zip":
            type(self).flaky_attempts += 1
            if type(self).flaky_attempts <= 2:
                self._send(503, b"busy", "text/plain")
                return
            self._send(200, _zip_bytes(), "application/zip")
            return
        if path == "/s/012345/bg.png":
            self._send(200, _PNG_BYTES, "image/png")
            return
        if path == "/s/012345/not-image.png":
            self._send(200, b"<html>nope</html>", "text/html")
            return
        self._send(404, b"missing", "text/plain")


class WmcClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.client = WmcClient(
            base_url=self.base,
            timeout=5,
            allowed_hosts=frozenset({"127.0.0.1"}),
        )
        sleep_patch = mock.patch("mainote.dataset.wmc.time.sleep", lambda *_: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_search_and_detail(self):
        result = self.client.search("demo", chart_type="dx", sort="popular", limit=30)
        self.assertEqual(result["charts"][0]["shortid"], "012345")
        self.assertEqual(result["available"]["versions"]["23"], "PRiSM")

        chart = self.client.chart("012345")
        self.assertEqual(chart["title"], "Online Demo")
        self.assertIn("zipNoVideo", chart["download"])

    def test_search_validation(self):
        for kwargs in (
            {"limit": 0},
            {"limit": 501},
            {"page": 0},
            {"chart_type": "other"},
            {"sort": "random"},
        ):
            with self.assertRaises(ValueError):
                self.client.search(**kwargs)
        with self.assertRaises(ValueError):
            self.client.chart("abc")
        with self.assertRaises(ValueError):
            self.client.chart("../../etc")

    def test_cover_download_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "cover.png"
            self.client.download_image("012345", "bg.png", target)
            self.assertEqual(target.read_bytes(), _PNG_BYTES)

        for shortid, image in (
            ("abc", "bg.png"),
            ("012345", "../bg.png"),
            ("012345", "evil.txt"),
        ):
            with self.assertRaises(ValueError):
                self.client.image_url(shortid, image)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WmcError):
                self.client.download_image(
                    "012345", "not-image.png", Path(tmp) / "bad.png"
                )

        foreign = WmcClient(
            base_url="https://example.com",
            timeout=5,
            allowed_hosts=frozenset({"download.wmc.pub"}),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WmcError):
                foreign.download_image("012345", "bg.png", Path(tmp) / "x.png")

    def test_download_and_import(self):
        chart = self.client.chart("012345")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "chart.zip"
            self.client.download_zip(chart, target)
            with zipfile.ZipFile(target) as archive:
                self.assertIn("song/maidata.txt", archive.namelist())

            result = import_zip_archive(
                target,
                raw_root=Path(tmp) / "raw",
                processed_root=Path(tmp) / "processed",
                filename="online_012345.zip",
            )
            self.assertTrue(result["ok"], result["errors"])
            self.assertEqual(result["title"], "Online Demo")
            self.assertTrue(result["has_background"])
            song_raw = Path(tmp) / "raw" / "online_012345"
            self.assertTrue((song_raw / "maidata.txt").is_file())
            self.assertTrue((song_raw / "track.mp3").is_file())
            self.assertTrue((song_raw / "bg.png").is_file())
            self.assertTrue((Path(tmp) / "processed" / "online_012345" / "meta.json").is_file())

    def test_download_retries_transient_errors(self):
        _Handler.flaky_attempts = 0
        chart = {"download": {"zipNoVideo": f"{self.base}/files/retry.zip"}}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "retry.zip"
            self.client.download_zip(chart, target)
            self.assertGreaterEqual(_Handler.flaky_attempts, 3)
            self.assertTrue(zipfile.is_zipfile(target))

    def test_download_rejects_bad_zip_and_foreign_host(self):
        chart = {"download": {"zipNoVideo": f"{self.base}/files/bad.zip"}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WmcError):
                self.client.download_zip(chart, Path(tmp) / "bad.zip")

        evil = {"download": {"zipNoVideo": "https://example.com/chart.zip"}}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(WmcError):
                self.client.download_zip(evil, Path(tmp) / "evil.zip")

    def test_failed_parse_cleans_partial_raw_song(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "broken.zip"
            archive.write_bytes(_zip_bytes(maidata="&inote_5=(120){4}1h[bad],E"))
            result = import_zip_archive(
                archive,
                raw_root=Path(tmp) / "raw",
                processed_root=Path(tmp) / "processed",
                filename="broken.zip",
            )
            self.assertFalse(result["ok"])
            self.assertIsNone(result["raw"])
            self.assertFalse((Path(tmp) / "raw" / "broken").exists())


if __name__ == "__main__":
    unittest.main()
