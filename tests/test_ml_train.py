import unittest

try:
    import torch
    from torch.nn import functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from mainote.ml.tokenizer import ChartVocabulary

if HAS_TORCH:
    from mainote.ml.train import (
        ClassAccuracy,
        build_token_weights,
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


if __name__ == "__main__":
    unittest.main()
