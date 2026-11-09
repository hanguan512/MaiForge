from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from mainote.maidata import parse_maidata_text

if HAS_DEPS:
    from mainote.ml.dataset import ChartWindowDataset
    from mainote.ml.tokenizer import EventTokenizer


@unittest.skipUnless(HAS_DEPS, "numpy and torch are required")
class TestChartWindowDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.features_dir = root / "features"
        self.features_dir.mkdir()
        np.savez_compressed(
            self.features_dir / "song.npz",
            mel=np.zeros((1000, 16), dtype=np.float32),
            fps=np.float32(10.0),
            n_mels=np.int32(16),
        )
        chart = parse_maidata_text(
            "&title=t\n&wholebpm=120\n&inote_5=(120){4}"
            "1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,E\n"
        ).chart(5)
        sample_path = root / "song" / "inote_5.json"
        sample_path.parent.mkdir()
        sample_path.write_text(
            json.dumps({"chart": chart.to_dict()}), encoding="utf-8"
        )
        self.manifest = root / "manifest.jsonl"
        self.manifest.write_text(
            json.dumps(
                {
                    "sample_id": "song_inote_5",
                    "song_id": "song",
                    "chart_index": 5,
                    "level_num": 10.0,
                    "split": "train",
                    "has_audio": True,
                    "valid": True,
                    "sample": str(sample_path),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, **kwargs) -> ChartWindowDataset:
        features_dir = kwargs.pop("features_dir", self.features_dir)
        options = {
            "window_bars": 2,
            "stride_bars": 2,
            "context_bars": 0,
            "max_audio_frames": 4096,
            "min_events": 1,
        }
        options.update(kwargs)
        return ChartWindowDataset(
            self.manifest,
            features_dir,
            "train",
            EventTokenizer(),
            **options,
        )

    def test_item_shapes_and_caches(self):
        dataset = self._dataset()
        self.assertGreater(len(dataset), 0)
        item = dataset[0]
        self.assertEqual(item["mel"].ndim, 2)
        self.assertEqual(item["mel"].shape[1], 16)
        self.assertGreater(len(item["tokens"]), 0)
        self.assertIs(dataset.chart_for(0), dataset.chart_for(0))
        self.assertEqual(len(dataset._feature_cache), 1)
        self.assertEqual(len(dataset._chart_cache), 1)

    def test_overlong_window_is_skipped_at_init(self):
        dataset = self._dataset(max_audio_frames=8)
        self.assertEqual(len(dataset), 0)
        self.assertGreater(dataset.skipped_windows.get("overlong_audio", 0), 0)

    def test_missing_features_are_skipped_at_init(self):
        dataset = self._dataset(features_dir=self.root / "missing-features")
        self.assertEqual(len(dataset), 0)
        self.assertGreater(
            dataset.skipped_windows.get("missing_features", 0), 0
        )

    def test_overlong_token_window_is_skipped_at_init(self):
        dataset = self._dataset(max_tokens=1)
        self.assertEqual(len(dataset), 0)
        self.assertGreater(
            dataset.skipped_windows.get("overlong_tokens", 0), 0
        )

    def test_same_tick_shuffle_is_deterministic_and_keeps_tokens(self):
        plain = self._dataset()
        shuffled = self._dataset(shuffle_same_tick_events=True)
        self.assertGreater(len(shuffled), 0)
        self.assertTrue(shuffled.shuffle_same_tick_events)
        for index in range(min(2, len(shuffled))):
            self.assertEqual(
                sorted(plain[index]["tokens"]),
                sorted(shuffled[index]["tokens"]),
            )
        # Each window gets a stable seed, so two dataset instances tokenize
        # identically (training, validation and the usability check agree).
        again = self._dataset(shuffle_same_tick_events=True)
        self.assertEqual(shuffled[0]["tokens"], again[0]["tokens"])


if __name__ == "__main__":
    unittest.main()
