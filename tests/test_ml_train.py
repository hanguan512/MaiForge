import unittest

try:
    import torch
    from torch.nn import functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mainote.maidata.model import TICKS_PER_BAR, Event
from mainote.ml.tokenizer import ChartVocabulary

if HAS_TORCH:
    from mainote.ml.model import ChartModelConfig, build_model
    from mainote.ml.train import (
        ClassAccuracy,
        _decode_budget,
        _trim_generated_events,
        build_token_weights,
        load_init_weights,
        sequence_cross_entropy,
    )


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestTrainingLoss(unittest.TestCase):
    def test_sequence_loss_is_window_normalized(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 4, 5)
        labels = torch.tensor([[1, 2, 3, 4], [2, 2, 0, 0]])
        loss, token_loss, valid = sequence_cross_entropy(logits, labels, 0)
        expected_tokens = F.cross_entropy(
            logits.reshape(-1, 5),
            labels.reshape(-1),
            ignore_index=0,
            reduction="none",
        ).view_as(labels)
        expected = (
            expected_tokens[0].mean() + expected_tokens[1, :2].mean()
        ) / 2
        self.assertTrue(torch.allclose(token_loss, expected_tokens))
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        flat = expected_tokens[valid].mean()
        self.assertNotAlmostEqual(float(loss), float(flat), places=5)

    def test_token_weights_apply_per_token(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[1, 2, 3]])
        weights = torch.tensor([1.0, 1.0, 5.0, 1.0, 1.0])
        loss, token_loss, valid = sequence_cross_entropy(
            logits, labels, 0, token_weights=weights
        )
        expected = (token_loss * weights[labels]).sum() / weights[labels].sum()
        self.assertAlmostEqual(float(loss), float(expected), places=5)

    def test_build_token_weights_uses_categories(self):
        vocab = ChartVocabulary.default()
        weights = build_token_weights(vocab, {"type": 2.0, "duration": 0.5})
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(
            float(weights[vocab.token_to_id["TAP"]]), 2.0
        )
        self.assertAlmostEqual(
            float(weights[vocab.token_to_id["DUR_1"]]), 0.5
        )
        self.assertAlmostEqual(
            float(weights[vocab.token_to_id["P_1"]]), 1.0
        )
        exact = build_token_weights(vocab, {"type": 2.0, "SLIDE": 3.0})
        self.assertAlmostEqual(
            float(exact[vocab.token_to_id["SLIDE"]]), 3.0
        )
        self.assertAlmostEqual(
            float(exact[vocab.token_to_id["TAP"]]), 2.0
        )
        with self.assertRaises(ValueError):
            build_token_weights(vocab, {"not-a-category": 2.0})

    def test_load_init_weights_allows_only_new_structured_projections(self):
        vocab = ChartVocabulary.default()
        base_cfg = ChartModelConfig(
            hidden=16,
            audio_layers=1,
            decoder_layers=1,
            heads=4,
            ffn=32,
            dropout=0.0,
            max_audio_frames=64,
            max_tokens=32,
            n_mels=16,
        )
        base = build_model(len(vocab.tokens), base_cfg, pad_id=vocab.pad_id)
        checkpoint = {
            "model": base.state_dict(),
            "model_config": base_cfg.to_dict(),
            "vocab_tokens": vocab.tokens,
        }
        experiment_cfg = ChartModelConfig(
            **{
                **base_cfg.to_dict(),
                "position_features": True,
                "duration_features": True,
            }
        )
        model = build_model(
            len(vocab.tokens),
            experiment_cfg,
            pad_id=vocab.pad_id,
            vocab_tokens=vocab.tokens,
        )
        missing = load_init_weights(model, experiment_cfg, checkpoint, vocab)
        self.assertEqual(
            missing, ["duration_proj.weight", "position_proj.weight"]
        )
        # The new projections keep their zero initialization; every baseline
        # parameter is copied exactly.
        state = model.state_dict()
        self.assertEqual(float(state["position_proj.weight"].abs().sum()), 0.0)
        self.assertEqual(float(state["duration_proj.weight"].abs().sum()), 0.0)
        for name, tensor in base.state_dict().items():
            self.assertTrue(torch.equal(state[name], tensor), name)

    def test_load_init_weights_rejects_architecture_mismatch(self):
        vocab = ChartVocabulary.default()
        base_cfg = ChartModelConfig(
            hidden=16,
            audio_layers=1,
            decoder_layers=1,
            heads=4,
            ffn=32,
            dropout=0.0,
            max_audio_frames=64,
            max_tokens=32,
            n_mels=16,
        )
        base = build_model(len(vocab.tokens), base_cfg, pad_id=vocab.pad_id)
        checkpoint = {
            "model": base.state_dict(),
            "model_config": base_cfg.to_dict(),
            "vocab_tokens": vocab.tokens,
        }
        other_cfg = ChartModelConfig(
            **{**base_cfg.to_dict(), "hidden": 32}
        )
        other = build_model(
            len(vocab.tokens), other_cfg, pad_id=vocab.pad_id
        )
        with self.assertRaises(RuntimeError):
            load_init_weights(other, other_cfg, checkpoint, vocab)

    def test_class_accuracy_by_token_category(self):
        vocab = ChartVocabulary(
            ["PAD", "BOS", "EOS", "UNK", "TAP", "HOLD", "DUR_1", "DUR_2", "P_1", "P_2"]
        )
        accuracy = ClassAccuracy(vocab)
        labels = torch.tensor([[4, 6, 8, 0]])  # TAP, DUR_1, P_1, pad
        preds = torch.tensor([[4, 6, 9, 0]])  # position wrong
        accuracy.update(preds, labels, labels.ne(0))
        result = accuracy.to_dict()
        self.assertEqual(result["type"]["accuracy"], 1.0)
        self.assertEqual(result["duration"]["accuracy"], 1.0)
        self.assertEqual(result["position"]["accuracy"], 0.0)
        self.assertEqual(result["special"]["total"], 0)


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestNoteEvalHelpers(unittest.TestCase):
    def test_decode_budget_tracks_reference_length(self):
        self.assertEqual(_decode_budget(4, 2048), 64)
        self.assertEqual(_decode_budget(100, 2048), 232)
        self.assertEqual(_decode_budget(1000, 2048), 2032)
        self.assertEqual(_decode_budget(1000, 128), 128)
        self.assertEqual(_decode_budget(1000, 0), 0)

    def test_trim_generated_events_uses_generated_bars(self):
        vocab = ChartVocabulary.default()
        # One BAR after the first event means the model covered bar 0 and bar 1;
        # an event at bar 2 is outside the generated range.
        ids = vocab.encode(["BOS", "T_0", "TAP", "P_1", "END_EVENT", "BAR"])
        events = [
            Event(tick=0, kind="tap", position="1"),
            Event(tick=2 * TICKS_PER_BAR, kind="tap", position="2"),
        ]
        kept, bars = _trim_generated_events(
            ids,
            events,
            vocab,
            start_tick=0,
            end_tick=4 * TICKS_PER_BAR,
        )
        self.assertEqual(bars, 2)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].tick, 0)

    def test_trim_generated_events_offsets_absolute_ticks(self):
        vocab = ChartVocabulary.default()
        ids = vocab.encode(["BOS", "T_0", "TAP", "P_1", "END_EVENT"])
        events = [Event(tick=0, kind="tap", position="1")]
        kept, bars = _trim_generated_events(
            ids,
            events,
            vocab,
            start_tick=2 * TICKS_PER_BAR,
            end_tick=4 * TICKS_PER_BAR,
        )
        self.assertEqual(bars, 1)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].tick, 2 * TICKS_PER_BAR)


if __name__ == "__main__":
    unittest.main()
