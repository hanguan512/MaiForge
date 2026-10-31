import json
import tempfile
import unittest
from pathlib import Path

from mainote.dataset.ingest import ingest_dataset
from mainote.dataset.samples import (
    build_manifest,
    export_samples,
    parse_level,
)
from mainote.dataset.split import split_dataset

TWO_CHART_MAIDATA = """\
&title={title}
&artist=test
&wholebpm=120
&lv_2=3.0
&des_2=-
&inote_2=(120){{4}}1,2,3,4,E
&lv_5=12.0
&des_5=-
&inote_5=(120){{4}}1,2,3,4,5,6,7,8,E
"""


class TestSampleExport(unittest.TestCase):
    def _make_processed(self, root: Path, songs: int = 10) -> Path:
        raw = root / "raw"
        for i in range(songs):
            song = raw / f"song_{i:02d}"
            song.mkdir(parents=True)
            (song / "maidata.txt").write_text(
                TWO_CHART_MAIDATA.format(title=f"song_{i:02d}"), encoding="utf-8"
            )
        processed = root / "processed"
        ingest_dataset(raw, processed)
        return processed

    def test_parse_level(self):
        self.assertEqual(parse_level("12"), 12.0)
        self.assertEqual(parse_level("12.9"), 12.9)
        self.assertEqual(parse_level("12+"), 12.5)
        self.assertEqual(parse_level("12-"), 11.5)
        self.assertIsNone(parse_level("-"))
        self.assertIsNone(parse_level("?"))

    def test_manifest_one_row_per_difficulty(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = self._make_processed(Path(tmp), songs=10)
            result = build_manifest(processed, seed=0, train=0.6, val=0.2, test=0.2)
            rows = result["rows"]
            self.assertEqual(len(rows), 20)
            self.assertEqual(len({r["sample_id"] for r in rows}), 20)
            self.assertEqual(
                {r["chart_index"] for r in rows}, {2, 5}
            )
            self.assertEqual(
                {r["difficulty"] for r in rows}, {"basic", "master"}
            )
            self.assertEqual(
                {r["level_num"] for r in rows}, {3.0, 12.0}
            )
            self.assertTrue(all(r["valid"] for r in rows))
            self.assertTrue(all(r["alignment_confidence"] is None for r in rows))

            by_song = {}
            for row in rows:
                by_song.setdefault(row["song_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in by_song.values()))
            self.assertEqual(
                result["summary"]["split_song_counts"]["train"]
                + result["summary"]["split_song_counts"]["val"]
                + result["summary"]["split_song_counts"]["test"],
                10,
            )

    def test_export_creates_per_difficulty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = self._make_processed(root, songs=4)
            out = root / "samples"
            summary = export_samples(processed, out, seed=0, train=0.5, val=0.5, test=0.0)
            self.assertEqual(summary["chart_count"], 8)
            self.assertTrue((out / "manifest.jsonl").exists())
            self.assertTrue((out / "summary.json").exists())
            self.assertTrue((out / "splits.json").exists())

            lines = [
                json.loads(line)
                for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 8)
            for row in lines:
                sample_path = out / row["song_id"] / f"inote_{row['chart_index']}.json"
                self.assertTrue(sample_path.exists(), sample_path)
                self.assertEqual(Path(row["sample"]), sample_path.resolve())
                sample = json.loads(sample_path.read_text(encoding="utf-8"))
                self.assertEqual(sample["sample_id"], row["sample_id"])
                self.assertEqual(sample["chart_index"], row["chart_index"])
                self.assertEqual(sample["chart"]["index"], row["chart_index"])
                self.assertIn("song_metadata", sample)
                self.assertGreater(sample["stats"]["notes_total"], 0)

    def test_require_audio_filters_chart_only_songs(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = self._make_processed(Path(tmp), songs=3)
            result = build_manifest(processed, require_audio=True)
            self.assertEqual(result["rows"], [])
            self.assertEqual(len(result["summary"]["skipped_no_audio"]), 3)

    def test_reuse_existing_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = self._make_processed(root, songs=6)
            splits_path = root / "splits.json"
            splits_path.write_text(
                json.dumps(split_dataset(processed, seed=3)), encoding="utf-8"
            )
            result = build_manifest(processed, splits=splits_path)
            expected = json.loads(splits_path.read_text(encoding="utf-8"))
            for name in ("train", "val", "test"):
                for song_id in expected["splits"][name]:
                    song_rows = [r for r in result["rows"] if r["song_id"] == song_id]
                    self.assertTrue(song_rows)
                    self.assertTrue(all(r["split"] == name for r in song_rows))


if __name__ == "__main__":
    unittest.main()
