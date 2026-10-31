import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mainote.dataset.ingest import ingest_dataset
from mainote.dataset.report import report_dataset
from mainote.dataset.split import split_dataset

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


SIMPLE_MAIDATA = """\
&title=simple
&artist=mainote
&wholebpm=143
&first=0
&lv_5=12.0
&des_5=test
&inote_5=(143){4}1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,E
"""


class TestDatasetIngest(unittest.TestCase):
    def test_ingest_chart_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "song_a"
            raw.mkdir(parents=True)
            (raw / "maidata.txt").write_text(SIMPLE_MAIDATA, encoding="utf-8")
            out = Path(tmp) / "processed"
            result = ingest_dataset(raw.parent, out)
            self.assertEqual(len(result.records), 1)
            record = result.records[0]
            self.assertFalse(record["has_audio"])
            self.assertEqual(record["chart_count"], 1)
            self.assertTrue((out / "song_a" / "meta.json").exists())
            self.assertTrue((out / "song_a" / "maidata.ir.json").exists())

            report = report_dataset(out)
            self.assertEqual(report["song_count"], 1)
            self.assertEqual(report["audio_coverage"], 0.0)
            self.assertIn("master", report["difficulties"])

            meta = json.loads((out / "song_a" / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["title"], "simple")
            self.assertEqual(meta["charts"][0]["notes_total"], 16)

    def test_split_by_song_is_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            for i in range(10):
                song = raw / f"song_{i}"
                song.mkdir(parents=True)
                text = SIMPLE_MAIDATA.replace("title=simple", f"title=song_{i}")
                (song / "maidata.txt").write_text(text, encoding="utf-8")
            out = Path(tmp) / "processed"
            ingest_dataset(raw, out)
            a = split_dataset(out, seed=7)
            b = split_dataset(out, seed=7)
            self.assertEqual(a, b)
            self.assertEqual(
                a["counts"]["train"] + a["counts"]["val"] + a["counts"]["test"], 10
            )
            self.assertEqual(a["counts"]["val"], 1)
            self.assertEqual(a["counts"]["test"], 1)
            all_ids = (
                a["splits"]["train"] + a["splits"]["val"] + a["splits"]["test"]
            )
            self.assertEqual(len(all_ids), len(set(all_ids)))

    @unittest.skipUnless(HAS_NUMPY, "numpy is required for audio alignment")
    def test_ingest_with_alignment(self):
        from mainote.demo import make_click_track

        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw" / "song_b"
            raw.mkdir(parents=True)
            audio = Path(tmp) / "click.wav"
            make_click_track(audio, bpm=143.0, bars=16)
            shutil.copy2(audio, raw / "audio.wav")
            (raw / "maidata.txt").write_text(SIMPLE_MAIDATA, encoding="utf-8")
            out = Path(tmp) / "processed"
            result = ingest_dataset(raw.parent, out, align=True)
            self.assertEqual(len(result.records), 1)
            record = result.records[0]
            self.assertTrue(record["has_audio"])
            self.assertIsNotNone(record["timing"])
            alignment = record["charts"][0]["alignment"]
            self.assertIsNotNone(alignment)
            self.assertGreater(alignment["confidence"], 0.3)
            self.assertLess(abs(alignment["best_offset_seconds"]), 0.1)


if __name__ == "__main__":
    unittest.main()
