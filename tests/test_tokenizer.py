import json
import tempfile
import unittest
from pathlib import Path

from mainote.maidata import parse_file, parse_maidata_text
from mainote.ml.tokenizer import (
    ChartTokenGrammar,
    ChartVocabulary,
    EventTokenizer,
    TokenizeError,
    level_bucket,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"

EXTRA = """\
&title=token-test
&wholebpm=120
&inote_5=(120){4}1b,2h[4:1],3x,4hx[4:3],5-7[4:1]b,6-1>2-7[16:7],8-3[8:1]*-4[8:1],C,E1,D6f,Chf[1:1],1V3-5[4:1],2p3[8:1],4q2[16:3],1w5[8:1],E
"""


class TestTokenizer(unittest.TestCase):
    def test_sample_roundtrip(self):
        maidata = parse_file(SAMPLE)
        tokenizer = EventTokenizer()
        for chart in maidata.charts:
            tokens = tokenizer.tokenize_chart(chart)
            parsed = tokenizer.detokenize(tokens)
            expected = [e.signature() for e in sorted(chart.events, key=lambda e: e.sort_key())]
            actual = [e.signature() for e in sorted(parsed.events, key=lambda e: e.sort_key())]
            self.assertEqual(expected, actual, f"inote_{chart.index}")

    def test_extra_syntax_roundtrip(self):
        chart = parse_maidata_text(EXTRA).chart(5)
        tokenizer = EventTokenizer()
        tokens = tokenizer.tokenize_chart(chart)
        parsed = tokenizer.detokenize(tokens)
        expected = [e.signature() for e in sorted(chart.events, key=lambda e: e.sort_key())]
        actual = [e.signature() for e in sorted(parsed.events, key=lambda e: e.sort_key())]
        self.assertEqual(expected, actual)
        self.assertIn("CHAIN_SAME", tokens)
        self.assertIn("CHAIN_PATH", tokens)
        self.assertIn("SHAPE_V", tokens)
        self.assertIn("TRACK_BREAK", tokens)
        self.assertIn("S_C", tokens)
        self.assertIn("S_D6", tokens)

    def test_window_relative_time(self):
        chart = parse_file(SAMPLE).chart(5)
        tokenizer = EventTokenizer()
        start = 10 * 3840
        end = 12 * 3840
        tokens = tokenizer.tokenize_chart(chart, start_tick=start, end_tick=end)
        parsed = tokenizer.detokenize(tokens, start_tick=start)
        expected = [
            e.signature()
            for e in chart.events
            if start <= e.tick < end
        ]
        actual = [e.signature() for e in parsed.events]
        self.assertEqual(expected, actual)
        self.assertGreater(len(actual), 0)

    def test_vocabulary_save_load(self):
        vocab = ChartVocabulary.default()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            vocab.save(path)
            loaded = ChartVocabulary.load(path)
        self.assertEqual(vocab.tokens, loaded.tokens)
        self.assertEqual(vocab.token_to_id, loaded.token_to_id)

    def test_level_bucket(self):
        self.assertEqual(level_bucket(12.9), 26)
        self.assertEqual(level_bucket("12+"), 25)
        self.assertEqual(level_bucket(None), 0)
        self.assertEqual(level_bucket(99), 30)

    def test_grammar_allows_expected_prefix(self):
        vocab = ChartVocabulary.default()
        grammar = ChartTokenGrammar(vocab)
        for token in ("BOS", "DIFF_5", "LEVEL_24", "T_0", "TAP", "P_1"):
            grammar.feed(token)
        allowed = grammar.allowed_tokens()
        self.assertIn("END_EVENT", allowed)
        self.assertIn("FLAG_BREAK", allowed)
        self.assertNotIn("T_1", allowed)
        self.assertNotIn("P_2", allowed)

    def test_grammar_rejects_backwards_time(self):
        vocab = ChartVocabulary.default()
        grammar = ChartTokenGrammar(vocab)
        for token in ("BOS", "T_10", "TAP", "P_1", "END_EVENT"):
            grammar.feed(token)
        with self.assertRaises(TokenizeError):
            grammar.feed("T_5")

    def test_encode_decode_ids(self):
        chart = parse_file(SAMPLE).chart(2)
        tokenizer = EventTokenizer()
        ids = tokenizer.encode_chart(chart)
        parsed = tokenizer.decode_ids(ids)
        self.assertEqual(len(parsed.events), chart.note_count)

    def test_empty_chart(self):
        chart = parse_maidata_text(
            "&title=t\n&wholebpm=120\n&inote_5=(120){4},E\n"
        ).chart(5)
        tokenizer = EventTokenizer()
        tokens = tokenizer.tokenize_chart(chart)
        self.assertEqual(tokens, ["BOS", "DIFF_5", "LEVEL_0", "EOS"])
        parsed = tokenizer.detokenize(tokens)
        self.assertEqual(parsed.events, [])


if __name__ == "__main__":
    unittest.main()
