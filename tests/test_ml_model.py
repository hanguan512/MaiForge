import math
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

    def test_initial_loss_is_near_uniform(self):
        # Tied embeddings must start small; PyTorch's N(0, 1) default makes the
        # first cross-entropy ~sqrt(hidden) times too large.
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
        mel = torch.randn(2, 48, 80)
        mel_mask = torch.zeros(2, 48, dtype=torch.bool)
        tokens = torch.randint(1, 100, (2, 12))
        token_mask = torch.zeros(2, 12, dtype=torch.bool)
        with torch.no_grad():
            logits = model(mel, mel_mask, tokens, token_mask)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 100), torch.randint(1, 100, (2 * 12,))
        )
        self.assertLess(float(loss), 10.0)

    def test_v2_architecture_embedding_scale_and_stride(self):
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
            embed_scale=True,
            final_norm=True,
            audio_stride=2,
        )
        model = build_model(100, config, pad_id=0)
        self.assertAlmostEqual(model.embed_scale, math.sqrt(32))
        model.eval()
        mel = torch.randn(1, 65, 80)
        mel_mask = torch.zeros(1, 65, dtype=torch.bool)
        mel_mask[:, 60:] = True
        memory, memory_mask = model.audio(mel, mel_mask)
        self.assertEqual(memory.shape[1], 33)  # ceil(65 / 2)
        self.assertEqual(memory_mask.shape[1], 33)
        self.assertFalse(bool(memory_mask[0, 0]))
        self.assertTrue(bool(memory_mask[0, -1]))
        tokens = torch.randint(1, 100, (1, 8))
        expected = model.token_pos(
            model.token_embed(tokens) * math.sqrt(config.hidden)
        )
        self.assertTrue(
            torch.allclose(model.token_embeddings(tokens), expected)
        )

    def test_audio_stride_variants_match_mask_length(self):
        for stride in (1, 2, 4):
            config = ChartModelConfig(
                hidden=16,
                audio_layers=1,
                decoder_layers=1,
                heads=4,
                ffn=32,
                dropout=0.0,
                max_audio_frames=128,
                max_tokens=32,
                n_mels=16,
                audio_stride=stride,
            )
            model = build_model(50, config, pad_id=0)
            mel = torch.randn(1, 65, 16)
            mel_mask = torch.zeros(1, 65, dtype=torch.bool)
            memory, memory_mask = model.audio(mel, mel_mask)
            expected = (65 + stride - 1) // stride
            self.assertEqual(memory.shape[1], expected, stride)
            self.assertEqual(memory_mask.shape[1], expected, stride)

    def test_invalid_audio_stride_rejected(self):
        with self.assertRaises(ValueError):
            ChartModelConfig(hidden=16, audio_stride=3).validate()

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
