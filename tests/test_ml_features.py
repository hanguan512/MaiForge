import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from mainote.audio.io import write_wav
from mainote.ml.features import log_mel_spectrogram
from mainote.ml.prepare import prepare_features


@unittest.skipUnless(HAS_NUMPY, "numpy is required")
class TestMelFeatures(unittest.TestCase):
    def test_log_mel_shape_and_finite(self):
        sr = 22050
        t = np.arange(sr, dtype=np.float32) / sr
        samples = np.sin(2 * np.pi * 440.0 * t).astype(np.float32) * 0.5
        mel, fps = log_mel_spectrogram(samples, sr, n_mels=40, n_fft=512, hop=256)
        self.assertEqual(mel.ndim, 2)
        self.assertEqual(mel.shape[1], 40)
        self.assertGreater(mel.shape[0], 0)
        self.assertAlmostEqual(fps, sr / 256)
        self.assertTrue(np.isfinite(mel).all())

    def test_prepare_features_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "song.wav"
            sr = 22050
            t = np.arange(sr, dtype=np.float32) / sr
            samples = np.sin(2 * np.pi * 440.0 * t).astype(np.float32) * 0.5
            write_wav(audio_path, samples, sr)
            processed = root / "processed" / "song"
            processed.mkdir(parents=True)
            (processed / "meta.json").write_text(
                json.dumps(
                    {
                        "song_id": "song",
                        "audio": str(audio_path),
                        "has_audio": True,
                    }
                ),
                encoding="utf-8",
            )
            out = root / "features"
            summary = prepare_features(
                processed.parent,
                out,
                n_mels=40,
                n_fft=512,
                hop=256,
            )
            self.assertEqual(summary["written"], 1)
            self.assertTrue((out / "song.npz").exists())
            data = np.load(out / "song.npz")
            self.assertEqual(tuple(data["mel"].shape[1:]), (40,))
            self.assertAlmostEqual(float(data["fps"]), sr / 256)
            self.assertTrue(np.isfinite(data["mel"]).all())


if __name__ == "__main__":
    unittest.main()
