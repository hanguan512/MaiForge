import unittest
from pathlib import Path

from mainote.maidata import Chart, Duration, Event, SlideSegment, parse_file
from mainote.maidata.model import BpmChange, TICKS_PER_BAR
from mainote.maidata.validate import validate_chart, validate_maidata

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"


def make_chart(events, *, bpm=120.0, index=5):
    end = max((e.end_tick() for e in events), default=0) + TICKS_PER_BAR
    return Chart(
        index=index,
        level="12.0",
        initial_bpm=bpm,
        events=events,
        bpm_changes=[BpmChange(0, bpm)],
        end_tick=end,
    )


class TestValidation(unittest.TestCase):
    def test_sample_valid(self):
        report = validate_maidata(parse_file(SAMPLE))
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.warnings, [])

    def test_duplicate_position(self):
        chart = make_chart(
            [
                Event(tick=0, kind="tap", position="1"),
                Event(tick=0, kind="tap", position="1"),
            ]
        )
        self.assertFalse(validate_chart(chart).ok)

    def test_hold_overlap(self):
        chart = make_chart(
            [
                Event(tick=0, kind="hold", position="1", duration=Duration(ticks=960)),
                Event(tick=480, kind="hold", position="1", duration=Duration(ticks=960)),
            ]
        )
        self.assertFalse(validate_chart(chart).ok)

    def test_tap_inside_hold(self):
        chart = make_chart(
            [
                Event(tick=0, kind="hold", position="1", duration=Duration(ticks=960)),
                Event(tick=480, kind="tap", position="1"),
            ]
        )
        self.assertFalse(validate_chart(chart).ok)

    def test_straight_slide_same_end(self):
        chart = make_chart(
            [
                Event(
                    tick=0,
                    kind="slide",
                    position="1",
                    slide=[
                        SlideSegment(shape="-", end="1", duration=Duration(ticks=480))
                    ],
                )
            ]
        )
        self.assertFalse(validate_chart(chart).ok)

    def test_path_slide_valid(self):
        chart = make_chart(
            [
                Event(
                    tick=0,
                    kind="slide",
                    position="4",
                    duration=Duration(ticks=1680),
                    chain_type="path",
                    slide=[
                        SlideSegment(shape="-", end="1"),
                        SlideSegment(shape=">", end="2"),
                        SlideSegment(shape="-", end="7"),
                    ],
                )
            ]
        )
        self.assertTrue(validate_chart(chart).ok, validate_chart(chart).errors)

    def test_firework_on_tap(self):
        chart = make_chart(
            [Event(tick=0, kind="tap", position="1", flags=["firework"])]
        )
        self.assertFalse(validate_chart(chart).ok)

    def test_invalid_position(self):
        chart = make_chart([Event(tick=0, kind="tap", position="9")])
        self.assertFalse(validate_chart(chart).ok)

    def test_invalid_bpm(self):
        chart = make_chart([Event(tick=0, kind="tap", position="1")], bpm=-1)
        self.assertFalse(validate_chart(chart).ok)

    def test_zero_duration(self):
        chart = make_chart(
            [Event(tick=0, kind="hold", position="1", duration=Duration(ticks=0))]
        )
        self.assertFalse(validate_chart(chart).ok)

    def test_touch_hold_valid(self):
        chart = make_chart(
            [
                Event(
                    tick=0,
                    kind="touch_hold",
                    position="C",
                    duration=Duration(ticks=960),
                    flags=["firework"],
                )
            ]
        )
        self.assertTrue(validate_chart(chart).ok, validate_chart(chart).errors)


if __name__ == "__main__":
    unittest.main()
