import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mainote.ml.model import ChartModelConfig, build_model


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestChartModel(unittest.TestCase):
    def test_forward_shapes(self):
        torch.manual_seed(0)
        config = ChartModelConfig(
            hidden=32,
            audio_layers=1,
            decoder_layers=1,
            heads=4,
            ffn=64,
            dropout=0.0,
            max_audio_frames=128,
            max_tokens=64,
            n_mels=80,
        )
        model = build_model(100, config, pad_id=0)
        mel = torch.randn(2, 64, 80)
        mel_mask = torch.zeros(2, 64, dtype=torch.bool)
        mel_mask[1, 40:] = True
        tokens = torch.randint(1, 100, (2, 12))
        token_mask = torch.zeros(2, 12, dtype=torch.bool)
        logits = model(mel, mel_mask, tokens, token_mask)
        self.assertEqual(tuple(logits.shape), (2, 12, 100))
        self.assertGreater(model.parameter_count(), 0)

    def test_padding_does_not_change_valid_logits(self):
        torch.manual_seed(0)
        config = ChartModelConfig(
            hidden=32,
            audio_layers=1,
            decoder_layers=1,
            heads=4,
            ffn=64,
            dropout=0.0,
            max_audio_frames=128,
            max_tokens=64,
            n_mels=80,
        )
        model = build_model(50, config, pad_id=0)
        model.eval()
        mel = torch.randn(1, 48, 80)
        mel_mask = torch.zeros(1, 48, dtype=torch.bool)
        tokens = torch.randint(1, 50, (1, 10))
        token_mask = torch.zeros(1, 10, dtype=torch.bool)
        with torch.no_grad():
            a = model(mel, mel_mask, tokens, token_mask)
            b = model(mel, mel_mask, tokens, token_mask)
        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
