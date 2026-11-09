"""Tests for the portable ``.maidb`` dataset bundle format."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mainote.dataset.bundle import (
    FORMAT_NAME,
    FORMAT_VERSION,
    export_bundle,
    import_bundle,
    inspect_bundle,
    resolve_raw_directory,
)

MAIDATA_A = """\
&title=Bundle A
&artist=Tester
&wholebpm=143
&first=0
&lv_5=12.0
&des_5=test
&inote_5=(143){4}1,2,3,4,1,2,3,4,E
"""

MAIDATA_B = """\
&title=Bundle B
&artist=Tester
&wholebpm=120
&first=0
&lv_4=9.0
&des_4=test
&inote_4=(120){4}8>2[4:1]*s4<5[4:1],E
"""


def _write_raw(root: Path, name: str, maidata: str, *, audio: bytes | None = b"fake-audio"):
    song = root / name
    song.mkdir(parents=True)
    (song / "maidata.txt").write_text(maidata, encoding="utf-8")
    if audio is not None:
        (song / "track.mp3").write_bytes(audio)
    (song / "bg.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return song


def _manifest(bundle: Path) -> dict:
    with zipfile.ZipFile(bundle) as archive:
        return json.loads(archive.read("maidb.json"))


class TestMaidbExportImport(unittest.TestCase):
    def test_roundtrip_preserves_songs_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            _write_raw(raw, "song_a", MAIDATA_A)
            _write_raw(raw, "song_b", MAIDATA_B, audio=None)

            bundle = tmp / "exports" / "corpus.maidb"
            summary = export_bundle(
                [("song_a", raw / "song_a"), ("song_b", raw / "song_b")], bundle
            )
            self.assertTrue(summary["ok"])
            self.assertEqual(summary["song_count"], 2)
            self.assertEqual(summary["file_count"], 5)  # 3 files + 2 files, no audio on b
            self.assertTrue(bundle.is_file())
            self.assertEqual(bundle.suffix, ".maidb")

            manifest = _manifest(bundle)
            self.assertEqual(manifest["format"], FORMAT_NAME)
            self.assertEqual(manifest["version"], FORMAT_VERSION)
            self.assertEqual(manifest["song_count"], 2)
            self.assertEqual(
                [song["id"] for song in manifest["songs"]], ["song_a", "song_b"]
            )
            self.assertEqual(manifest["songs"][0]["title"], "Bundle A")
            self.assertEqual(manifest["songs"][0]["chart_count"], 1)
            self.assertTrue(
                all(len(item["sha256"]) == 64 for item in manifest["songs"][0]["files"])
            )

            info = inspect_bundle(bundle)
            self.assertEqual(info["format"], FORMAT_NAME)
            self.assertEqual(info["song_count"], 2)

            out_raw = tmp / "imported" / "raw"
            out_processed = tmp / "imported" / "processed"
            result = import_bundle(
                bundle, raw_root=out_raw, processed_root=out_processed
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["bundle_song_count"], 2)
            self.assertEqual(result["imported"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["chart_count"], 2)
            self.assertTrue((out_raw / "song_a" / "maidata.txt").is_file())
            self.assertEqual(
                (out_raw / "song_a" / "track.mp3").read_bytes(), b"fake-audio"
            )
            self.assertTrue((out_raw / "song_a" / "bg.png").is_file())
            self.assertFalse((out_raw / "song_b" / "track.mp3").exists())
            meta = json.loads(
                (out_processed / "song_a" / "meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["title"], "Bundle A")
            self.assertTrue(meta["has_audio"])
            self.assertEqual(
                [song["song_id"] for song in result["results"]], ["song_a", "song_b"]
            )

    def test_import_reports_bundle_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            _write_raw(raw, "song_a", MAIDATA_A)
            _write_raw(raw, "song_b", MAIDATA_B, audio=None)
            bundle = tmp / "corpus.maidb"
            export_bundle(
                [("song_a", raw / "song_a"), ("song_b", raw / "song_b")], bundle
            )

            events: list[dict] = []
            result = import_bundle(
                bundle,
                raw_root=tmp / "imported" / "raw",
                processed_root=tmp / "imported" / "processed",
                progress=events.append,
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(events[0]["phase"], "extract")
            songs = [event for event in events if event["phase"] == "songs"]
            self.assertEqual(songs[0]["done"], 0)
            self.assertEqual(songs[0]["total"], 2)
            self.assertEqual(songs[-1]["done"], 2)
            self.assertEqual(
                {event["song_id"] for event in songs if event["song_id"]},
                {"song_a", "song_b"},
            )

    def test_import_duplicate_ids_get_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            _write_raw(raw, "song_a", MAIDATA_A)
            bundle = tmp / "one.maidb"
            export_bundle([("song_a", raw / "song_a")], bundle)

            out_raw = tmp / "out" / "raw"
            out_processed = tmp / "out" / "processed"
            first = import_bundle(
                bundle, raw_root=out_raw, processed_root=out_processed
            )
            second = import_bundle(
                bundle, raw_root=out_raw, processed_root=out_processed
            )
            self.assertEqual(first["results"][0]["song_id"], "song_a")
            self.assertEqual(second["results"][0]["song_id"], "song_a_2")
            self.assertTrue((out_processed / "song_a" / "meta.json").is_file())
            self.assertTrue((out_processed / "song_a_2" / "meta.json").is_file())

    def test_export_from_processed_meta_raw_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            song = _write_raw(raw, "song_a", MAIDATA_A)
            processed = tmp / "processed" / "song_a"
            processed.mkdir(parents=True)
            (processed / "meta.json").write_text(
                json.dumps({"song_id": "song_a", "raw_dir": str(song)}),
                encoding="utf-8",
            )
            self.assertEqual(resolve_raw_directory(processed), song.resolve())

            missing = tmp / "processed" / "broken"
            missing.mkdir()
            (missing / "meta.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_raw_directory(missing)

    def test_no_audio_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            song = _write_raw(raw, "song_a", MAIDATA_A)
            bundle = tmp / "charts.maidb"
            summary = export_bundle([("song_a", song)], bundle, include_audio=False)
            self.assertFalse(summary["songs"][0]["has_audio"])
            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
            self.assertNotIn("songs/song_a/track.mp3", names)
            self.assertIn("songs/song_a/maidata.txt", names)
            self.assertIn("songs/song_a/bg.png", names)

    def test_manifestless_archive_imports_as_one_song(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "plain.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("maidata.txt", MAIDATA_A)
                archive.writestr("track.mp3", b"fake-audio")
            renamed = tmp / "plain.maidb"
            source.rename(renamed)

            result = import_bundle(
                renamed,
                raw_root=tmp / "raw",
                processed_root=tmp / "processed",
            )
            self.assertTrue(result["ok"], result)
            self.assertIsNone(result["version"])
            self.assertEqual(result["imported"], 1)
            self.assertEqual(result["results"][0]["song_id"], "plain")

    def test_checksum_mismatch_fails_that_song_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            _write_raw(raw, "song_a", MAIDATA_A)
            _write_raw(raw, "song_b", MAIDATA_B)
            bundle = tmp / "corpus.maidb"
            export_bundle(
                [("song_a", raw / "song_a"), ("song_b", raw / "song_b")], bundle
            )

            tampered = tmp / "tampered.maidb"
            with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(
                tampered, "w"
            ) as target:
                for info in source.infolist():
                    data = source.read(info)
                    if info.filename == "songs/song_a/track.mp3":
                        data = b"tampered"
                    target.writestr(info, data)

            result = import_bundle(
                tampered,
                raw_root=tmp / "out" / "raw",
                processed_root=tmp / "out" / "processed",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["imported"], 1)
            self.assertEqual(result["failed"], 1)
            failed = next(song for song in result["results"] if not song["ok"])
            self.assertIn("校验失败", " ".join(failed["errors"]))
            self.assertFalse(
                (tmp / "out" / "raw" / "song_a" / "maidata.txt").exists()
            )
            self.assertTrue(
                (tmp / "out" / "raw" / "song_b" / "maidata.txt").is_file()
            )

    def test_rejects_zip_slip_and_manifest_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            escaping = tmp / "evil.maidb"
            with zipfile.ZipFile(escaping, "w") as archive:
                archive.writestr("../evil.txt", "escape")
                archive.writestr("maidata.txt", MAIDATA_A)
            result = import_bundle(
                escaping,
                raw_root=tmp / "raw",
                processed_root=tmp / "processed",
            )
            self.assertFalse(result["ok"])
            self.assertIn("非法路径", " ".join(result["errors"]))
            self.assertFalse((tmp / "evil.txt").exists())

            traversal = tmp / "traversal.maidb"
            manifest = {
                "format": FORMAT_NAME,
                "version": FORMAT_VERSION,
                "songs": [{"id": "evil", "dir": "../outside", "files": []}],
            }
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("maidb.json", json.dumps(manifest))
            result = import_bundle(
                traversal,
                raw_root=tmp / "raw2",
                processed_root=tmp / "processed2",
            )
            self.assertFalse(result["ok"])
            self.assertEqual(result["failed"], 1)
            self.assertIn("非法路径", " ".join(result["results"][0]["errors"]))

    def test_rejects_unknown_format_and_newer_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for manifest, expected in (
                ({"format": "other", "version": 1, "songs": []}, "format"),
                (
                    {
                        "format": FORMAT_NAME,
                        "version": FORMAT_VERSION + 1,
                        "songs": [{"id": "a", "dir": "songs/a"}],
                    },
                    "版本",
                ),
                ({"format": FORMAT_NAME, "version": 1, "songs": []}, "没有歌曲"),
            ):
                path = tmp / f"bad_{expected}.maidb"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("maidb.json", json.dumps(manifest))
                result = import_bundle(
                    path,
                    raw_root=tmp / "raw",
                    processed_root=tmp / "processed",
                )
                self.assertFalse(result["ok"])
                self.assertIn(expected, " ".join(result["errors"]))


if __name__ == "__main__":
    unittest.main()
