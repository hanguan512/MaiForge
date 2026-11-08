import math
import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mainote.ml.model import ChartModelConfig, build_model
    from mainote.ml.token_features import (
        build_duration_features,
        build_position_features,
    )
    from mainote.ml.tokenizer import ChartVocabulary


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

    def test_position_features_encode_ring_geometry(self):
        tokens = [
            "P_1",
            "P_2",
            "P_3",
            "P_5",
            "P_9",
            "S_A1",
            "S_E8",
            "S_C",
            "S_C1",
            "TAP",
        ]
        features = build_position_features(tokens)
        # Button 1 sits at angle 0, button 3 at pi/2, button 5 opposite.
        self.assertAlmostEqual(float(features[0, 0]), 0.0)
        self.assertAlmostEqual(float(features[0, 1]), 1.0)
        self.assertAlmostEqual(float(features[2, 0]), 1.0)
        self.assertAlmostEqual(float(features[2, 1]), 0.0)
        self.assertAlmostEqual(float(features[3, 1]), -1.0)
        # The sin/cos part alone yields ring adjacency.
        adjacent = float(features[0, :2] @ features[1, :2])
        opposite = float(features[0, :2] @ features[3, :2])
        self.assertAlmostEqual(adjacent, math.cos(math.pi / 4))
        self.assertAlmostEqual(opposite, -1.0)
        # Ring one-hots after [sin, cos, button].
        for index in range(4):
            self.assertEqual(float(features[index, 2]), 1.0)
        self.assertEqual(float(features[5, 3]), 1.0)  # S_A1 -> ring A
        self.assertEqual(float(features[6, 7]), 1.0)  # S_E8 -> ring E
        self.assertEqual(float(features[7, 5]), 1.0)  # S_C -> ring C
        self.assertEqual(float(features[7, 0]), 0.0)
        self.assertEqual(float(features[7, 1]), 0.0)
        # Centre sensors have no angular component.
        self.assertEqual(float(features[8, 5]), 1.0)
        self.assertEqual(float(features[8, 0]), 0.0)
        self.assertEqual(float(features[8, 1]), 0.0)
        # Unknown / unrelated tokens stay zero.
        self.assertEqual(float(features[4].abs().sum()), 0.0)
        self.assertEqual(float(features[9].abs().sum()), 0.0)

    def test_duration_features_are_monotonic(self):
        tokens = [
            "DUR_NONE",
            "DUR_0",
            "DUR_96",
            "DUR_2048",
            "DUR_9999",
            "TAP",
        ]
        features = build_duration_features(tokens)
        self.assertEqual(float(features[0].abs().sum()), 0.0)
        self.assertEqual(float(features[1, 0]), 0.0)
        self.assertEqual(float(features[1, 2]), 1.0)
        self.assertGreater(float(features[2, 0]), 0.0)
        self.assertLess(float(features[2, 0]), 1.0)
        self.assertAlmostEqual(float(features[3, 0]), 1.0)
        self.assertAlmostEqual(float(features[3, 1]), 1.0)
        self.assertEqual(float(features[3, 3]), 1.0)
        # Values above MAX_DUR_UNITS clamp to the same feature row.
        self.assertTrue(torch.equal(features[3], features[4]))
        self.assertEqual(float(features[5].abs().sum()), 0.0)
        log_columns = features[1:4, 0].tolist()
        self.assertEqual(log_columns, sorted(log_columns))

    def test_structured_features_start_as_baseline(self):
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
            position_features=True,
            duration_features=True,
        )
        vocab = ChartVocabulary.default()
        model = build_model(
            len(vocab.tokens),
            config,
            pad_id=vocab.pad_id,
            vocab_tokens=vocab.tokens,
        )
        ids = torch.tensor(
            [vocab.encode(["BOS", "P_1", "DUR_96", "TAP", "END_EVENT", "EOS"])]
        )
        expected = model.token_pos(model.token_embed(ids) * model.embed_scale)
        self.assertTrue(torch.allclose(model.token_embeddings(ids), expected))
        # The projections are the only extra parameters.
        self.assertIsNotNone(model.position_proj)
        self.assertIsNotNone(model.duration_proj)
        self.assertEqual(model.position_proj.weight.shape, (32, 8))
        self.assertEqual(model.duration_proj.weight.shape, (32, 4))

    def test_structured_features_receive_gradients(self):
        torch.manual_seed(0)
        config = ChartModelConfig(
            hidden=16,
            audio_layers=1,
            decoder_layers=1,
            heads=4,
            ffn=32,
            dropout=0.0,
            max_audio_frames=64,
            max_tokens=32,
            n_mels=16,
            position_features=True,
            duration_features=True,
        )
        vocab = ChartVocabulary.default()
        model = build_model(
            len(vocab.tokens),
            config,
            pad_id=vocab.pad_id,
            vocab_tokens=vocab.tokens,
        )
        ids = torch.tensor([vocab.encode(["P_3", "DUR_96", "P_7"])])
        model.token_embeddings(ids).sum().backward()
        for projection in (model.position_proj, model.duration_proj):
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)

    def test_forward_and_backward_with_structured_features(self):
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
            position_features=True,
            duration_features=True,
        )
        vocab = ChartVocabulary.default()
        model = build_model(
            len(vocab.tokens),
            config,
            pad_id=vocab.pad_id,
            vocab_tokens=vocab.tokens,
        )
        mel = torch.randn(1, 32, 80)
        mel_mask = torch.zeros(1, 32, dtype=torch.bool)
        tokens = torch.tensor(
            [
                vocab.encode(
                    ["BOS", "P_1", "DUR_96", "TAP", "END_EVENT", "EOS"]
                )
            ]
        )
        token_mask = torch.zeros(1, tokens.size(1), dtype=torch.bool)
        logits = model(mel, mel_mask, tokens, token_mask)
        self.assertEqual(
            tuple(logits.shape), (1, tokens.size(1), len(vocab.tokens))
        )
        logits.sum().backward()
        for projection in (model.position_proj, model.duration_proj):
            self.assertIsNotNone(projection.weight.grad)
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)

    def test_structured_features_require_vocabulary(self):
        config = ChartModelConfig(
            hidden=16,
            audio_layers=1,
            decoder_layers=1,
            heads=4,
            ffn=32,
            dropout=0.0,
            max_audio_frames=64,
            max_tokens=32,
            n_mels=16,
            position_features=True,
        )
        with self.assertRaises(ValueError):
            build_model(50, config, pad_id=0)
        with self.assertRaises(ValueError):
            build_model(50, config, pad_id=0, vocab_tokens=["PAD"])

    def test_config_round_trip_includes_structured_flags(self):
        config = ChartModelConfig(position_features=True, duration_features=True)
        self.assertEqual(ChartModelConfig.from_dict(config.to_dict()), config)
        self.assertFalse(ChartModelConfig.from_dict({}).position_features)
        self.assertFalse(ChartModelConfig.from_dict({}).duration_features)

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
