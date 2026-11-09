"""Tests for the local FastAPI web UI (browse / infer / train control)."""

import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from mainote.maidata.parser import parse_maidata_text

try:
    from fastapi.testclient import TestClient

    from mainote.webapp.app import create_app
    from mainote.webapp.settings import WebSettings

    HAS_WEB = True
except ImportError:  # fastapi/httpx live in the `web` extra
    HAS_WEB = False

MAIDATA_TEXT = """\
&title=Demo Song
&artist=Tester
&wholebpm=120
&lv_5=12+
&des_5=designer
&inote_5=(120){4}1,2-6[4:3],3,4,E
"""


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


ONLINE_MAIDATA = """\
&title=Online Demo
&artist=Tester
&wholebpm=120
&inote_5=(120){4}8>2[4:1]*s4<5[4:1],6h[1:0],2-6[4:3],E
"""


class _FakeWmcClient:
    """Stands in for the network so the import job can be tested offline."""

    def search(self, query="", **kwargs):
        return {
            "total": 1,
            "matched": 1,
            "page": 1,
            "pages": 1,
            "limit": 30,
            "count": 1,
            "charts": [
                {
                    "shortid": "012345",
                    "title": "Online Demo",
                    "image": "bg.png",
                }
            ],
            "available": {"versions": {"23": "PRiSM"}, "genres": ["POPS＆アニメ"]},
        }

    def chart(self, shortid):
        return {
            "shortid": shortid,
            "title": "Online Demo",
            "version": "PRiSM",
            "versionFolder": "PRiSM",
            "folderName": "Online Demo [DX]",
            "bpm": 120,
        }

    def download_zip(self, chart, target, *, include_video=False, timeout=600.0):
        target.write_bytes(
            _zip_bytes(
                {
                    "song/maidata.txt": ONLINE_MAIDATA.encode(),
                    "song/track.mp3": b"fake-mp3",
                    "song/bg.png": b"\x89PNG\r\n\x1a\n",
                }
            )
        )
        return target

    def download_image(self, shortid, image, target, *, timeout=60.0):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x89PNG\r\n\x1a\nfake-cover")
        return target


def _write_repo(repo: Path, *, with_static: bool = False) -> Path:
    """Create a miniature repository: config, one processed song, one run."""
    configs = repo / "configs"
    configs.mkdir(parents=True)
    (configs / "p3_base.yaml").write_text("train:\n  epochs: 2\n", encoding="utf-8")

    song = repo / "data" / "processed" / "demo"
    song.mkdir(parents=True)
    maidata = parse_maidata_text(MAIDATA_TEXT)
    (song / "maidata.ir.json").write_text(
        json.dumps(maidata.to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    (song / "audio.mp3").write_bytes(b"fake-mp3-bytes")
    (song / "bg.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (song / "meta.json").write_text(
        json.dumps(
            {
                "song_id": "demo",
                "title": "Demo Song",
                "artist": "Tester",
                "audio": "audio.mp3",
                "timing": {"bpm": 120.0, "first": 0.0, "duration": 40.0},
                "charts": [
                    {
                        "index": 5,
                        "level": "12+",
                        "designer": "designer",
                        "notes_total": 5,
                        "notes_per_second": 2.0,
                    }
                ],
                "warnings": [],
                "errors": [],
            }
        ),
        encoding="utf-8",
    )

    samples = repo / "data" / "samples" / "demo"
    samples.mkdir(parents=True)
    (samples / "inote_5.json").write_text(
        json.dumps(
            {
                "sample_id": "demo:inote_5",
                "song_id": "demo",
                "chart_index": 5,
                "difficulty": "master",
            }
        ),
        encoding="utf-8",
    )
    (samples.parent / "manifest.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "demo:inote_5",
                "song_id": "demo",
                "chart_index": 5,
                "difficulty": "master",
                "has_audio": True,
                "valid": True,
                "split": "train",
                "sample": "data/samples/demo/inote_5.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    features = repo / "data" / "features"
    features.mkdir(parents=True)
    (features / "demo.npz").write_bytes(b"fake-npz")

    run = repo / "runs" / "run_a"
    run.mkdir(parents=True)
    rows = [
        {"epoch": 1, "step": 10, "train_loss": 3.0, "val_loss": 2.5},
        {
            "epoch": 2,
            "step": 20,
            "train_loss": 2.0,
            "val_loss": 1.5,
            "note_eval": {"note_score": 0.4, "onset_f1": 0.5},
        },
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (run / "best.pt").write_bytes(b"checkpoint")
    (run / "config.json").write_text(
        json.dumps({"train": {"epochs": 2}}), encoding="utf-8"
    )

    if with_static:
        static = repo / "src" / "mainote" / "webapp" / "static"
        (static / "assets").mkdir(parents=True)
        (static / "index.html").write_text("<html>studio</html>", encoding="utf-8")
        (static / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        majdataview = repo / "src" / "mainote" / "webapp" / "majdataview"
        (majdataview / "Build").mkdir(parents=True)
        (majdataview / "index.html").write_text("<html>majdataview</html>", encoding="utf-8")
        (majdataview / "Build" / "1.wasm").write_bytes(b"\0asm")
    return repo


@unittest.skipUnless(HAS_WEB, "fastapi/httpx are required for web tests")
class WebAppTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = _write_repo(Path(self._tmp.name) / "repo")
        self.settings = WebSettings(repo_root=self.repo)
        self.app = create_app(self.settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self._tmp.cleanup()

    # -- meta / browse ---------------------------------------------------
    def test_health_roots_and_overview(self):
        self.assertEqual(self.client.get("/api/health").json()["service"], "mainote-webapp")

        roots = self.client.get("/api/roots").json()["roots"]
        self.assertEqual(len(roots), len(WebSettings(repo_root=self.repo).data_roots))
        self.assertTrue(roots[0]["exists"])
        self.assertEqual(roots[0]["song_count"], 1)

        overview = self.client.get("/api/overview").json()
        self.assertEqual(overview["songs"], 1)
        self.assertEqual(overview["charts"], 1)
        self.assertEqual(overview["runs"], 1)
        self.assertEqual(overview["checkpoints"], 1)
        self.assertEqual(overview["configs"], 1)
        self.assertFalse(overview["active_job"]["active"])

    def test_songs_and_song_detail(self):
        songs = self.client.get("/api/songs").json()["songs"]
        self.assertEqual(len(songs), 1)
        self.assertEqual(songs[0]["title"], "Demo Song")
        self.assertTrue(songs[0]["has_audio"])
        self.assertTrue(songs[0]["has_ir"])
        self.assertEqual(songs[0]["chart_count"], 1)

        detail = self.client.get("/api/song", params={"id": "demo"}).json()
        self.assertEqual(detail["id"], "demo")
        self.assertTrue(detail["ir_available"])
        self.assertEqual(detail["charts"][0]["index"], 5)
        self.assertEqual(detail["charts"][0]["difficulty"], "master")
        self.assertIn("/api/audio?", detail["audio_url"])

    def test_chart_payload_events_and_slides(self):
        payload = self.client.get(
            "/api/chart", params={"id": "demo", "index": 5}
        ).json()
        self.assertEqual(payload["difficulty"], "master")
        self.assertAlmostEqual(payload["initial_bpm"], 120.0)
        self.assertTrue(payload["validation"]["ok"], payload["validation"]["errors"])
        self.assertIn("&inote_5", payload["maidata"])
        self.assertEqual(payload["source"]["kind"], "song")
        self.assertEqual(payload["charts"][0]["index"], 5)
        self.assertEqual(payload["charts"][0]["difficulty"], "master")
        self.assertAlmostEqual(payload["audio_duration_seconds"], 40.0)

        slides = [event for event in payload["events"] if event["kind"] == "slide"]
        self.assertEqual(len(slides), 1)
        segments = slides[0]["segments"]
        self.assertEqual(segments[0]["end"], "6")
        self.assertGreater(segments[0]["end_seconds"], segments[0]["start_seconds"])
        self.assertAlmostEqual(
            segments[0]["end_seconds"] - segments[0]["start_seconds"],
            1.5,
            places=3,
        )
        for event in payload["events"]:
            self.assertIn("seconds", event)
            self.assertGreaterEqual(event["end_seconds"], event["seconds"])

        missing = self.client.get("/api/chart", params={"id": "demo", "index": 2})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("不存在", missing.json()["error"])

    def test_audio_endpoint(self):
        response = self.client.get(
            "/api/audio", params={"root": "data/processed", "song": "demo"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-mp3-bytes")
        missing = self.client.get(
            "/api/audio", params={"root": "data/processed", "song": "nope"}
        )
        self.assertEqual(missing.status_code, 400)

    # -- MajdataView preview ---------------------------------------------
    def test_majdataview_maidata_uses_first_slot(self):
        response = self.client.get(
            "/api/majdataview/maidata",
            params={"root": "data/processed", "id": "demo", "index": 5},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        text = response.text
        self.assertIn("&inote_1=", text)
        self.assertIn("&lv_1=12+", text)
        self.assertNotIn("&inote_5=", text)
        chart = parse_maidata_text(text).chart(1)
        self.assertIsNotNone(chart)
        self.assertGreater(len(chart.events), 0)

        missing = self.client.get(
            "/api/majdataview/maidata",
            params={"root": "data/processed", "id": "demo", "index": 2},
        )
        self.assertEqual(missing.status_code, 400)

    def test_majdataview_maidata_seeks_by_shifting_first(self):
        response = self.client.get(
            "/api/majdataview/maidata",
            params={
                "root": "data/processed",
                "id": "demo",
                "index": 5,
                "seek": 1.0,
            },
        )
        self.assertEqual(response.status_code, 200)
        text = response.text
        self.assertIn("&first=-1", text)
        self.assertNotIn("&first_1=", text)
        chart = parse_maidata_text(text).chart(1)
        self.assertIsNotNone(chart)
        self.assertAlmostEqual(chart.first, -1.0)
        # At 120 BPM the first bar is 2s long; seeking to 1.0s drops the
        # events at 0.0s and 0.5s but keeps the ones at 1.0s and 1.5s.
        self.assertEqual(len(chart.events), 2)

        negative = self.client.get(
            "/api/majdataview/maidata",
            params={
                "root": "data/processed",
                "id": "demo",
                "index": 5,
                "seek": -1,
            },
        )
        self.assertEqual(negative.status_code, 422)

    def test_majdataview_audio_and_background(self):
        audio = self.client.get(
            "/api/majdataview/audio",
            params={"root": "data/processed", "song": "demo", "duration": 3},
        )
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.headers["content-type"], "audio/mpeg")
        self.assertEqual(audio.content, b"fake-mp3-bytes")

        background = self.client.get(
            "/api/majdataview/bg", params={"root": "data/processed", "song": "demo"}
        )
        self.assertEqual(background.status_code, 200)
        self.assertEqual(background.headers["content-type"], "image/png")

        missing = self.client.get(
            "/api/majdataview/bg", params={"root": "data/processed", "song": "nope"}
        )
        self.assertEqual(missing.status_code, 400)

    # -- path validation -------------------------------------------------
    def test_rejects_path_traversal_and_unknown_roots(self):
        for params in (
            {"id": ".."},
            {"root": "data/processed", "id": "../../../etc"},
        ):
            response = self.client.get("/api/song", params=params)
            self.assertEqual(response.status_code, 400, params)

        response = self.client.get("/api/files", params={"path": "../../../../etc/passwd"})
        self.assertEqual(response.status_code, 400)

        response = self.client.get(
            "/api/chart/file", params={"path": "../../../../etc/passwd"}
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.get("/api/songs", params={"root": "/tmp/elsewhere"})
        self.assertEqual(response.status_code, 400)

    def test_file_preview_and_download(self):
        path = self.repo / "data" / "processed" / "demo" / "meta.json"
        preview = self.client.get("/api/files", params={"path": str(path)})
        self.assertEqual(preview.status_code, 200)
        self.assertIn("Demo Song", preview.text)
        download = self.client.get(
            "/api/files", params={"path": str(path), "download": True}
        )
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download.headers.get("content-disposition", ""))

    # -- runs ------------------------------------------------------------
    def test_runs_and_run_detail(self):
        runs = self.client.get("/api/runs").json()["runs"]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["path"], "runs/run_a")
        self.assertEqual(runs[0]["checkpoint_count"], 1)
        self.assertAlmostEqual(runs[0]["best_val_loss"], 1.5)
        self.assertAlmostEqual(runs[0]["best_note_score"], 0.4)

        detail = self.client.get("/api/run", params={"path": "runs/run_a"}).json()
        self.assertEqual(len(detail["metrics"]), 2)
        self.assertEqual(detail["config"]["train"]["epochs"], 2)
        self.assertTrue(any(item["name"] == "best.pt" for item in detail["files"]))

        outside = self.client.get("/api/run", params={"path": "data/processed/demo"})
        self.assertEqual(outside.status_code, 400)

    def test_options_include_configs_checkpoints_and_audio(self):
        options = self.client.get("/api/options").json()
        self.assertTrue(any(item["value"] == "configs/p3_base.yaml" for item in options["configs"]))
        self.assertTrue(any(item["value"] == "runs/run_a/best.pt" for item in options["checkpoints"]))
        self.assertTrue(any(item["value"].endswith("demo/audio.mp3") for item in options["audio"]))
        self.assertEqual(len(options["runs"]), 1)

    # -- training data / model architecture ------------------------------
    def test_invalid_samples_are_not_marked_ready(self):
        manifest = self.repo / "data" / "samples" / "manifest.jsonl"
        row = json.loads(manifest.read_text(encoding="utf-8").strip())
        row["valid"] = False
        row["errors"] = ["inote_5: broken"]
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

        data = self.client.get("/api/train/data").json()
        song = data["songs"][0]
        self.assertFalse(song["sample_ready"])
        self.assertFalse(song["trainable"])
        self.assertEqual(data["summary"]["trainable"], 0)

    def test_train_data_catalog_and_config(self):
        data = self.client.get("/api/train/data").json()
        self.assertEqual(data["root"], "data/processed")
        self.assertEqual(data["samples_dir"], "data/samples")
        self.assertEqual(data["features_dir"], "data/features")
        self.assertEqual(data["manifest"]["rows"], 1)
        self.assertEqual(len(data["songs"]), 1)
        song = data["songs"][0]
        self.assertEqual(song["id"], "demo")
        self.assertTrue(song["has_audio"])
        self.assertTrue(song["has_ir"])
        self.assertTrue(song["sample_ready"])
        self.assertTrue(song["features_ready"])
        self.assertTrue(song["trainable"])
        self.assertEqual(data["summary"]["trainable"], 1)

        labelled = self.client.get(
            "/api/train/data", params={"root": "data/processed"}
        ).json()
        self.assertEqual(labelled["root"], "data/processed")

        config = self.client.get(
            "/api/train/config", params={"path": "configs/p3_base.yaml"}
        ).json()
        self.assertEqual(config["path"], "configs/p3_base.yaml")
        self.assertEqual(config["config"]["train"]["epochs"], 2)

        missing = self.client.get(
            "/api/train/config", params={"path": "configs/missing.yaml"}
        )
        self.assertEqual(missing.status_code, 400)

    def test_model_architecture_endpoint(self):
        payload = self.client.get("/api/model/architecture").json()
        self.assertGreater(payload["total_params"], 0)
        self.assertGreater(payload["vocab_size"], 0)
        self.assertAlmostEqual(payload["mel_fps"], 22050.0 / 256.0)
        self.assertAlmostEqual(payload["encoded_fps"], payload["mel_fps"] / 2)
        kinds = {layer["kind"] for group in payload["groups"] for layer in group["layers"]}
        self.assertIn("conv", kinds)
        self.assertIn("encoder", kinds)
        self.assertIn("decoder", kinds)
        self.assertIn("embed", kinds)
        columns = {group["column"] for group in payload["groups"]}
        self.assertEqual(columns, {"encoder", "decoder"})

        smaller = self.client.get(
            "/api/model/architecture", params={"audio_layers": 2}
        ).json()
        self.assertLess(smaller["total_params"], payload["total_params"])

        invalid = self.client.get(
            "/api/model/architecture", params={"hidden": 10, "heads": 3}
        )
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("整除", invalid.json()["error"])

        bad_stride = self.client.get(
            "/api/model/architecture", params={"audio_stride": 3}
        )
        self.assertEqual(bad_stride.status_code, 400)

    def test_train_selection_payload_validation(self):
        base = {
            "config": "configs/p3_base.yaml",
            "features_dir": "data/features",
            "song_ids": ["demo"],
        }
        unknown = self.client.post(
            "/api/train/start", json={**base, "song_ids": ["ghost"]}
        )
        self.assertEqual(unknown.status_code, 400)
        self.assertIn("不存在", unknown.json()["error"])

        no_audio = self.repo / "data" / "processed" / "mute"
        no_audio.mkdir()
        (no_audio / "meta.json").write_text(
            json.dumps({"song_id": "mute", "title": "Mute"}), encoding="utf-8"
        )
        silent = self.client.post(
            "/api/train/start", json={**base, "song_ids": ["mute"]}
        )
        self.assertEqual(silent.status_code, 400)
        self.assertIn("没有音频", silent.json()["error"])

        bad_hidden = self.client.post(
            "/api/train/start",
            json={
                **base,
                "hyperparams": {"model": {"hidden": 10, "heads": 3}},
            },
        )
        self.assertEqual(bad_hidden.status_code, 400)
        self.assertIn("整除", bad_hidden.json()["error"])

        bad_val = self.client.post(
            "/api/train/start", json={**base, "val_ratio": 1.0}
        )
        self.assertEqual(bad_val.status_code, 400)
        self.assertIn("val_ratio", bad_val.json()["error"])

    def test_prepare_payload_validation(self):
        empty = self.client.post("/api/train/prepare", json={})
        self.assertEqual(empty.status_code, 400)
        self.assertIn("至少一首", empty.json()["error"])

        unknown = self.client.post(
            "/api/train/prepare",
            json={"root": "data/processed", "song_ids": ["ghost"]},
        )
        self.assertEqual(unknown.status_code, 400)
        self.assertIn("不存在", unknown.json()["error"])

        no_audio = self.repo / "data" / "processed" / "mute"
        no_audio.mkdir()
        (no_audio / "meta.json").write_text(
            json.dumps({"song_id": "mute", "title": "Mute"}), encoding="utf-8"
        )
        silent = self.client.post(
            "/api/train/prepare",
            json={"root": "data/processed", "song_ids": ["mute"]},
        )
        self.assertEqual(silent.status_code, 400)
        self.assertIn("没有音频", silent.json()["error"])

        outside = self.client.post(
            "/api/train/prepare",
            json={
                "root": "data/processed",
                "song_ids": ["demo"],
                "samples_out": "../outside",
            },
        )
        self.assertEqual(outside.status_code, 400)

    # -- job validation (no subprocess is spawned) -----------------------
    def test_train_payload_validation(self):
        bad_config = self.client.post(
            "/api/train/start", json={"config": "configs/missing.yaml"}
        )
        self.assertEqual(bad_config.status_code, 400)
        self.assertIn("配置", bad_config.json()["error"])

        bad_device = self.client.post(
            "/api/train/start",
            json={"config": "configs/p3_base.yaml", "device": "tpu"},
        )
        self.assertEqual(bad_device.status_code, 400)
        self.assertIn("device", bad_device.json()["error"])

        bad_output = self.client.post(
            "/api/train/start",
            json={"config": "configs/p3_base.yaml", "output_dir": "../../escape"},
        )
        self.assertEqual(bad_output.status_code, 400)

        bad_resume = self.client.post(
            "/api/train/start",
            json={
                "config": "configs/p3_base.yaml",
                "resume": "data/processed/demo/meta.json",
            },
        )
        self.assertEqual(bad_resume.status_code, 400)

    def test_infer_payload_validation(self):
        missing_audio = self.client.post("/api/infer/start", json={})
        self.assertEqual(missing_audio.status_code, 400)

        wrong_checkpoint = self.client.post(
            "/api/infer/start",
            json={
                "audio": "data/processed/demo/audio.mp3",
                "checkpoint": "runs/run_a/config.json",
            },
        )
        self.assertEqual(wrong_checkpoint.status_code, 400)
        self.assertIn(".pt", wrong_checkpoint.json()["error"])

        bad_difficulty = self.client.post(
            "/api/infer/start",
            json={
                "audio": "data/processed/demo/audio.mp3",
                "checkpoint": "runs/run_a/best.pt",
                "difficulty": "ultra",
            },
        )
        self.assertEqual(bad_difficulty.status_code, 400)
        self.assertIn("难度", bad_difficulty.json()["error"])

        bad_output = self.client.post(
            "/api/infer/start",
            json={
                "audio": "data/processed/demo/audio.mp3",
                "checkpoint": "runs/run_a/best.pt",
                "output_dir": "../escape",
            },
        )
        self.assertEqual(bad_output.status_code, 400)

    # -- uploads ---------------------------------------------------------
    def test_audio_upload(self):
        accepted = self.client.post(
            "/api/upload/audio",
            files={"file": ("song.mp3", b"audio-bytes", "audio/mpeg")},
        )
        self.assertEqual(accepted.status_code, 200)
        body = accepted.json()
        self.assertTrue(body["path"].startswith("out/webui/uploads/"))
        self.assertTrue((self.settings.repo_root / body["path"]).is_file())

        rejected = self.client.post(
            "/api/upload/audio",
            files={"file": ("notes.txt", b"nope", "text/plain")},
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("不支持", rejected.json()["error"])

    # -- dataset import --------------------------------------------------
    def test_dataset_import_one_zip_per_song(self):
        payload = _zip_bytes(
            {
                "myosotis/track.mp3": b"fake-audio",
                "myosotis/maidata.txt": MAIDATA_TEXT.encode("utf-8"),
                "myosotis/bg.png": b"ignored",
            }
        )
        response = self.client.post(
            "/api/datasets/import",
            files=[("files", ("myosotis.zip", payload, "application/zip"))],
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["ok"])
        self.assertEqual(len(body["results"]), 1)
        item = body["results"][0]
        self.assertTrue(item["ok"], item)
        self.assertEqual(item["song_id"], "myosotis")
        self.assertEqual(item["title"], "Demo Song")
        self.assertEqual(item["chart_count"], 1)
        self.assertTrue(item["has_audio"])
        self.assertTrue((self.repo / "data" / "raw" / "myosotis" / "track.mp3").is_file())
        self.assertTrue(
            (self.repo / "data" / "processed" / "myosotis" / "meta.json").is_file()
        )

        songs = self.client.get(
            "/api/songs", params={"root": "data/processed"}
        ).json()["songs"]
        imported = next(song for song in songs if song["id"] == "myosotis")
        self.assertEqual(imported["title"], "Demo Song")
        self.assertIsNotNone(imported["cover_url"])

        chart = self.client.get(
            "/api/chart", params={"root": "data/processed", "id": "myosotis", "index": 5}
        ).json()
        self.assertTrue(chart["validation"]["ok"])
        self.assertIn("/api/audio?", chart["audio_url"])
        self.assertEqual(chart["cover_url"], imported["cover_url"])

        summary = self.client.get("/api/datasets").json()
        self.assertEqual(summary["processed_root"], "data/processed")
        self.assertEqual(summary["processed_count"], 2)
        self.assertEqual(summary["raw_count"], 1)
        self.assertTrue(any(song["id"] == "myosotis" for song in summary["songs"]))
        summary_song = next(song for song in summary["songs"] if song["id"] == "myosotis")
        self.assertIsNotNone(summary_song["cover_url"])
        self.assertFalse(summary_song["sample_ready"])
        self.assertFalse(summary_song["features_ready"])
        demo_song = next(song for song in summary["songs"] if song["id"] == "demo")
        self.assertTrue(demo_song["sample_ready"])
        self.assertTrue(demo_song["features_ready"])

    def test_dataset_batch_import_continues_after_failures(self):
        good = _zip_bytes(
            {
                "track.mp3": b"fake-audio",
                "maidata.txt": MAIDATA_TEXT.encode("utf-8"),
            }
        )
        missing = _zip_bytes({"readme.txt": b"no chart here"})
        response = self.client.post(
            "/api/datasets/import",
            files=[
                ("files", ("first.zip", good, "application/zip")),
                ("files", ("second.zip", missing, "application/zip")),
                ("files", ("notes.txt", b"nope", "text/plain")),
            ],
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["ok"])
        results = body["results"]
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0]["ok"], results[0])
        self.assertEqual(results[0]["song_id"], "first")
        self.assertFalse(results[1]["ok"])
        self.assertTrue(any("maidata" in error for error in results[1]["errors"]))
        self.assertFalse(results[2]["ok"])
        self.assertTrue(any("zip" in error for error in results[2]["errors"]))

    def test_dataset_import_job_reports_progress(self):
        payload = _zip_bytes(
            {
                "track.mp3": b"fake-audio",
                "maidata.txt": MAIDATA_TEXT.encode("utf-8"),
            }
        )
        response = self.client.post(
            "/api/datasets/import-jobs",
            files=[("files", ("progressive.zip", payload, "application/zip"))],
        )
        self.assertEqual(response.status_code, 200)
        job = response.json()
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["total_files"], 1)

        deadline = time.time() + 20
        while job["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
            polled = self.client.get(f"/api/datasets/import-jobs/{job['id']}")
            self.assertEqual(polled.status_code, 200)
            job = polled.json()

        self.assertEqual(job["status"], "done")
        self.assertEqual(job["completed_files"], 1)
        self.assertEqual(job["progress"], 1.0)
        self.assertEqual(len(job["results"]), 1)
        self.assertTrue(job["results"][0]["ok"], job["results"][0])
        self.assertEqual(job["results"][0]["song_id"], "progressive")

        jobs = self.client.get("/api/datasets/import-jobs").json()["jobs"]
        self.assertTrue(any(item["id"] == job["id"] for item in jobs))
        self.assertEqual(
            self.client.get("/api/datasets/import-jobs/does-not-exist").status_code,
            400,
        )
        self.assertTrue(
            (self.repo / "data" / "processed" / "progressive" / "meta.json").is_file()
        )

    def test_dataset_export_and_bundle_reimport(self):
        payload = _zip_bytes(
            {
                "myosotis/track.mp3": b"fake-audio",
                "myosotis/maidata.txt": MAIDATA_TEXT.encode("utf-8"),
                "myosotis/bg.png": b"\x89PNG\r\n\x1a\n",
            }
        )
        imported = self.client.post(
            "/api/datasets/import",
            files=[("files", ("myosotis.zip", payload, "application/zip"))],
        ).json()["results"][0]
        self.assertTrue(imported["ok"], imported)

        exported = self.client.post(
            "/api/datasets/export",
            json={"song_ids": ["myosotis"], "filename": "corpus"},
        )
        self.assertEqual(exported.status_code, 200)
        body = exported.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["file"], "corpus.maidb")
        self.assertEqual(body["song_count"], 1)
        self.assertEqual(body["songs"][0]["id"], "myosotis")
        self.assertTrue(body["download_url"].startswith("/api/datasets/exports/"))
        export_path = self.repo / body["path"]
        self.assertTrue(export_path.is_file())
        with zipfile.ZipFile(export_path) as archive:
            self.assertIn("maidb.json", archive.namelist())
            self.assertIn("songs/myosotis/track.mp3", archive.namelist())

        download = self.client.get(body["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertIn("attachment", download.headers.get("content-disposition", ""))
        self.assertTrue(download.content.startswith(b"PK"))
        self.assertEqual(self.client.get("/api/datasets/exports/nope.maidb").status_code, 400)

        reimported = self.client.post(
            "/api/datasets/import",
            files=[("files", ("corpus.maidb", download.content, "application/zip"))],
        ).json()["results"][0]
        self.assertTrue(reimported["ok"], reimported)
        self.assertEqual(reimported["kind"], "bundle")
        self.assertEqual(reimported["bundle_song_count"], 1)
        self.assertEqual(reimported["imported"], 1)
        self.assertEqual(reimported["failed"], 0)
        self.assertEqual(reimported["results"][0]["song_id"], "myosotis_2")
        self.assertEqual(reimported["results"][0]["title"], "Demo Song")

    def test_dataset_export_validation(self):
        empty = self.client.post("/api/datasets/export", json={"song_ids": []})
        self.assertEqual(empty.status_code, 400)
        self.assertIn("至少一首", empty.json()["error"])

        traversal = self.client.post(
            "/api/datasets/export", json={"song_ids": ["../demo"]}
        )
        self.assertEqual(traversal.status_code, 400)

        # ``demo`` has no raw folder in this miniature repo, so the raw
        # source needed for a portable bundle is missing.
        missing = self.client.post(
            "/api/datasets/export", json={"song_ids": ["demo"]}
        )
        self.assertEqual(missing.status_code, 400)
        self.assertIn("原始歌曲目录", missing.json()["error"])

    def test_dataset_import_duplicate_names_and_zip_slip(self):
        good = _zip_bytes(
            {
                "track.mp3": b"fake-audio",
                "maidata.txt": MAIDATA_TEXT.encode("utf-8"),
            }
        )
        response = self.client.post(
            "/api/datasets/import",
            files=[
                ("files", ("same.zip", good, "application/zip")),
                ("files", ("same.zip", good, "application/zip")),
            ],
        )
        ids = [item["song_id"] for item in response.json()["results"]]
        self.assertEqual(ids, ["same", "same_2"])

        escaping = _zip_bytes(
            {
                "../evil.txt": b"escape",
                "track.mp3": b"fake-audio",
                "maidata.txt": MAIDATA_TEXT.encode("utf-8"),
            }
        )
        rejected = self.client.post(
            "/api/datasets/import",
            files=[("files", ("evil.zip", escaping, "application/zip"))],
        )
        item = rejected.json()["results"][0]
        self.assertFalse(item["ok"])
        self.assertTrue(any("非法路径" in error for error in item["errors"]))
        self.assertFalse((self.repo / "evil.txt").exists())
        self.assertFalse((self.repo / "data" / "evil.txt").exists())

    # -- online catalogue ------------------------------------------------
    def _wait_wmc_job(self, job_id: str) -> dict:
        deadline = time.time() + 10
        while time.time() < deadline:
            job = self.client.get(f"/api/wmc/imports/{job_id}").json()
            if job["status"] not in ("running", "stopping"):
                return job
            time.sleep(0.05)
        self.fail("online import job did not finish")

    def test_wmc_search_and_import(self):
        self.app.state.wmc.client = _FakeWmcClient()

        search = self.client.get("/api/wmc/search", params={"q": "online"})
        self.assertEqual(search.status_code, 200)
        body = search.json()
        self.assertEqual(body["charts"][0]["shortid"], "012345")
        self.assertEqual(
            body["charts"][0]["cover_url"],
            "/api/wmc/cover/012345?image=bg.png",
        )
        self.assertTrue(body["source"].startswith("AWMC-MaiArchive"))

        cover = self.client.get(
            "/api/wmc/cover/012345", params={"image": "bg.png"}
        )
        self.assertEqual(cover.status_code, 200)
        self.assertEqual(cover.headers["content-type"], "image/png")
        self.assertTrue(cover.content.startswith(b"\x89PNG"))
        invalid_cover = self.client.get(
            "/api/wmc/cover/012345", params={"image": "../bg.png"}
        )
        self.assertEqual(invalid_cover.status_code, 400)

        invalid = self.client.post("/api/wmc/import", json={"ids": ["abc"]})
        self.assertEqual(invalid.status_code, 400)
        empty = self.client.post("/api/wmc/import", json={"ids": []})
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(self.client.get("/api/wmc/imports/nope").status_code, 404)

        started = self.client.post("/api/wmc/import", json={"ids": ["012345"]})
        self.assertEqual(started.status_code, 200)
        job = self._wait_wmc_job(started.json()["id"])
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["completed"], 1)
        self.assertTrue(job["results"][0]["ok"], job["results"][0])
        self.assertEqual(job["results"][0]["title"], "Online Demo")
        self.assertTrue(job["results"][0]["has_background"])
        self.assertTrue(
            (self.repo / "data" / "raw" / "wmc_012345" / "maidata.txt").is_file()
        )
        self.assertTrue(
            (self.repo / "data" / "processed" / "wmc_012345" / "meta.json").is_file()
        )

        songs = self.client.get(
            "/api/songs", params={"root": "data/processed"}
        ).json()["songs"]
        online = next(song for song in songs if song["id"] == "wmc_012345")
        self.assertIsNotNone(online["cover_url"])
        local_cover = self.client.get(online["cover_url"])
        self.assertEqual(local_cover.status_code, 200)
        self.assertEqual(local_cover.headers["content-type"], "image/png")

        # The mixed-chain and [1:0] chart must have survived parsing.
        chart = self.client.get(
            "/api/chart",
            params={"root": "data/processed", "id": "wmc_012345", "index": 5},
        ).json()
        self.assertTrue(chart["validation"]["ok"], chart["validation"]["errors"])
        self.assertEqual(chart["cover_url"], online["cover_url"])
        self.assertTrue(any(event["chain_type"] == "mixed" for event in chart["events"]))

        again = self.client.post("/api/wmc/import", json={"ids": ["012345"]})
        self.assertEqual(again.status_code, 200)
        skipped = self._wait_wmc_job(again.json()["id"])
        self.assertTrue(skipped["results"][0]["skipped"])

        jobs = self.client.get("/api/wmc/imports").json()["jobs"]
        self.assertEqual(len(jobs), 2)

    def test_wmc_failed_result_keeps_browser_shape(self):
        """Failed downloads must still carry ``errors``/``warnings`` arrays.

        The web UI spreads both lists while rendering; a missing ``warnings``
        key used to crash the React root and leave a blank page.
        """

        class _FailingClient(_FakeWmcClient):
            def download_zip(self, chart, target, **kwargs):
                raise RuntimeError("download exploded")

        self.app.state.wmc.client = _FailingClient()
        started = self.client.post("/api/wmc/import", json={"ids": ["099999"]})
        self.assertEqual(started.status_code, 200)
        job = self._wait_wmc_job(started.json()["id"])
        self.assertEqual(job["status"], "done")
        result = job["results"][0]
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"], ["download exploded"])
        self.assertEqual(result["warnings"], [])

    # -- SPA fallback ----------------------------------------------------
    def test_spa_fallback_without_build(self):
        index = self.client.get("/")
        self.assertEqual(index.status_code, 503)
        self.assertIn("npm run build", index.text)
        self.assertEqual(self.client.get("/api/does-not-exist").status_code, 404)

    def test_spa_fallback_with_build(self):
        index = self.client.get("/")
        self.assertEqual(index.status_code, 503)

        with tempfile.TemporaryDirectory() as tmp:
            repo = _write_repo(Path(tmp) / "repo", with_static=True)
            with self._client(repo) as client:
                response = client.get("/viewer")
                self.assertEqual(response.status_code, 200)
                self.assertIn("studio", response.text)
                asset = client.get("/assets/app.js")
                self.assertEqual(asset.status_code, 200)
                self.assertEqual(asset.text, "console.log(1)")

                player = client.get("/majdataview/")
                self.assertEqual(player.status_code, 200)
                self.assertIn("majdataview", player.text)
                wasm = client.get("/majdataview/Build/1.wasm")
                self.assertEqual(wasm.status_code, 200)
                self.assertEqual(wasm.headers["content-type"], "application/wasm")

    def _client(self, repo: Path):
        return TestClient(create_app(WebSettings(repo_root=repo)))


class TrainingSelectionTest(unittest.TestCase):
    """Training-data selection helpers do not need the web extra."""

    def test_assign_selected_splits_by_song(self):
        from mainote.webapp.training import assign_selected_splits

        splits = assign_selected_splits(["song_a", "song_b"], seed=7, val_ratio=0.1)
        self.assertEqual(sorted(splits), ["song_a", "song_b"])
        self.assertEqual(sorted(splits.values()), ["train", "val"])
        self.assertEqual(
            splits, assign_selected_splits(["song_b", "song_a"], seed=7, val_ratio=0.1)
        )

        single = assign_selected_splits(["solo"], seed=0, val_ratio=0.1)
        self.assertEqual(single, {"solo": "train"})

        no_val = assign_selected_splits(["a", "b", "c"], seed=0, val_ratio=0)
        self.assertEqual(set(no_val.values()), {"train"})

        many = assign_selected_splits([f"song_{i}" for i in range(10)], val_ratio=0.2)
        self.assertEqual(sum(1 for split in many.values() if split == "val"), 2)
        self.assertEqual(sum(1 for split in many.values() if split == "train"), 8)

    def test_build_selection_manifest_single_song(self):
        from mainote.webapp.training import build_selection_manifest

        with tempfile.TemporaryDirectory() as tmp:
            repo = _write_repo(Path(tmp) / "repo")
            settings = WebSettings(repo_root=repo)
            destination = repo / "runs" / "run_a" / "training_manifest.jsonl"
            info = build_selection_manifest(
                settings, ["demo"], destination, seed=0, val_ratio=0.1
            )
            self.assertEqual(info["rows"], 1)
            self.assertEqual(info["train_songs"], ["demo"])
            self.assertEqual(info["val_songs"], [])
            rows = [
                json.loads(line)
                for line in destination.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(rows[0]["split"], "train")
            self.assertTrue(Path(rows[0]["sample"]).is_absolute())
            self.assertTrue(Path(rows[0]["sample"]).is_file())

            with self.assertRaises(ValueError):
                build_selection_manifest(settings, ["ghost"], destination)

    def test_apply_hyperparams_whitelist_and_validation(self):
        from mainote.webapp.training import apply_hyperparams

        cfg = {
            "data": {"window_bars": 16, "spec_augment": {"prob": 0.5}},
            "model": {"hidden": 256, "heads": 8, "audio_stride": 2},
            "train": {"lr": 0.0003, "amp": True},
        }
        apply_hyperparams(
            cfg,
            {
                "data": {
                    "window_bars": 8,
                    "max_windows_per_chart": None,
                    "spec_augment_prob": 0.25,
                },
                "model": {"decoder_layers": 4},
                "train": {"lr": 0.001, "amp": False},
            },
        )
        self.assertEqual(cfg["data"]["window_bars"], 8)
        self.assertIsNone(cfg["data"]["max_windows_per_chart"])
        self.assertAlmostEqual(cfg["data"]["spec_augment"]["prob"], 0.25)
        self.assertEqual(cfg["model"]["decoder_layers"], 4)
        self.assertAlmostEqual(cfg["train"]["lr"], 0.001)
        self.assertFalse(cfg["train"]["amp"])

        with self.assertRaises(ValueError):
            apply_hyperparams(cfg, {"train": {"learning_rate": 0.1}})
        with self.assertRaises(ValueError):
            apply_hyperparams(cfg, {"train": {"lr": -1}})
        with self.assertRaises(ValueError):
            apply_hyperparams(cfg, {"mystery": {"x": 1}})
        with self.assertRaises(ValueError):
            apply_hyperparams(cfg, {"model": {"hidden": 10, "heads": 3}})
        with self.assertRaises(ValueError):
            apply_hyperparams(cfg, {"model": {"audio_stride": 3}})


@unittest.skipUnless(HAS_WEB, "fastapi/httpx are required for web tests")
class WebSettingsTest(unittest.TestCase):
    def test_relative_defaults_resolve_under_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            settings = WebSettings(repo_root=repo)
            self.assertEqual(settings.out_root, repo / "out")
            self.assertEqual(settings.runs_root, repo / "runs")
            self.assertEqual(settings.config_root, repo / "configs")
            self.assertEqual(settings.upload_dir, repo / "out" / "webui" / "uploads")
            self.assertIn(repo / "data" / "processed", settings.data_roots)

    def test_resolve_under_rejects_outside_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp).resolve()
            settings = WebSettings(repo_root=repo)
            allowed = settings.resolve_under("data", [repo])
            self.assertEqual(allowed, repo / "data")
            with self.assertRaises(ValueError):
                settings.resolve_under("..", [repo / "runs"])
            with self.assertRaises(ValueError):
                settings.resolve_under("missing.txt", [repo], must_exist=True)

    def test_unknown_data_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = WebSettings(repo_root=Path(tmp))
            with self.assertRaises(ValueError):
                settings.data_root_by_value("/some/where/else")


if __name__ == "__main__":
    unittest.main()
