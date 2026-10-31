import copy
import unittest
from pathlib import Path

from mainote.eval.metrics import evaluate_chart
from mainote.maidata import Chart, Duration, Event, parse_file
from mainote.maidata.model import BpmChange, TICKS_PER_BAR

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"


class TestEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maidata = parse_file(SAMPLE)
        cls.master = cls.maidata.chart(5)

    def test_self_eval_is_perfect(self):
        result = evaluate_chart(self.master, self.master)
        self.assertEqual(result["onset"]["f1"], 1.0)
        self.assertEqual(result["position_accuracy"], 1.0)
        self.assertEqual(result["type_accuracy"], 1.0)
        self.assertEqual(result["exact_accuracy"], 1.0)
        self.assertEqual(result["duration_mae_ticks"], 0.0)
        self.assertEqual(result["slide"]["endpoint_accuracy"], 1.0)

    def test_shifted_chart_lowers_f1(self):
        shifted = copy.deepcopy(self.master)
        for event in shifted.events:
            event.tick += 480  # ~0.21s at 143 BPM
        result = evaluate_chart(shifted, self.master)
        self.assertLess(result["onset"]["f1"], 1.0)

    def test_type_mismatch(self):
        ref_events = [Event(tick=0, kind="tap", position="1")]
        pred_events = [
            Event(tick=0, kind="hold", position="1", duration=Duration(ticks=960))
        ]
        ref = Chart(
            index=5,
            initial_bpm=120,
            events=ref_events,
            bpm_changes=[BpmChange(0, 120)],
            end_tick=TICKS_PER_BAR,
        )
        pred = Chart(
            index=5,
            initial_bpm=120,
            events=pred_events,
            bpm_changes=[BpmChange(0, 120)],
            end_tick=TICKS_PER_BAR,
        )
        result = evaluate_chart(pred, ref)
        self.assertEqual(result["onset"]["f1"], 1.0)
        self.assertEqual(result["position_accuracy"], 1.0)
        self.assertEqual(result["type_accuracy"], 0.0)
        self.assertLess(result["exact_accuracy"], 1.0)

    def test_extra_notes_reduce_precision(self):
        pred = copy.deepcopy(self.master)
        pred.events.append(Event(tick=TICKS_PER_BAR * 200, kind="tap", position="1"))
        result = evaluate_chart(pred, self.master)
        self.assertLess(result["onset"]["precision"], 1.0)
        self.assertEqual(result["onset"]["recall"], 1.0)


class TestTempoMap(unittest.TestCase):
    def test_seconds_and_ticks_roundtrip_with_tempo_changes(self):
        chart = Chart(
            index=5,
            initial_bpm=120.0,
            bpm_changes=[BpmChange(0, 120.0), BpmChange(TICKS_PER_BAR, 240.0)],
        )
        # First bar at 120 BPM = 2s; second bar at 240 BPM = 1s.
        self.assertAlmostEqual(chart.seconds_at_tick(TICKS_PER_BAR), 2.0)
        self.assertAlmostEqual(chart.seconds_at_tick(2 * TICKS_PER_BAR), 3.0)
        self.assertEqual(chart.ticks_at_seconds(1.0), TICKS_PER_BAR // 2)
        self.assertEqual(chart.ticks_at_seconds(2.0), TICKS_PER_BAR)
        self.assertEqual(chart.ticks_at_seconds(3.0), 2 * TICKS_PER_BAR)


if __name__ == "__main__":
    unittest.main()
