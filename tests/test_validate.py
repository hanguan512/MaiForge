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

    def test_slide_shape_endpoint_warnings(self):
        chart = make_chart(
            [
                Event(
                    tick=0,
                    kind="slide",
                    position="7",
                    slide=[
                        SlideSegment(
                            shape="-", end="6", duration=Duration(ticks=960)
                        )
                    ],
                ),
                Event(
                    tick=1920,
                    kind="slide",
                    position="1",
                    slide=[
                        SlideSegment(
                            shape="s", end="2", duration=Duration(ticks=960)
                        )
                    ],
                ),
                Event(
                    tick=3840,
                    kind="slide",
                    position="1",
                    slide=[
                        SlideSegment(
                            shape="w", end="5", duration=Duration(ticks=960)
                        )
                    ],
                ),
            ]
        )
        report = validate_chart(chart)
        self.assertTrue(report.ok, report.errors)
        messages = " ".join(report.warnings)
        self.assertIn("7->6", messages)
        self.assertIn("1->2", messages)
        self.assertNotIn("1->5", messages)

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

    def test_seconds_duration_uses_tempo_map(self):
        # 2 seconds starting at 200 BPM would be 6400 ticks if the tempo never
        # changed.  At tick 960 the BPM halves, so the hold really ends around
        # tick 3680; a tap at 5000 must not be reported as inside the hold.
        chart = Chart(
            index=5,
            level="12.0",
            initial_bpm=200.0,
            events=[
                Event(
                    tick=0,
                    kind="hold",
                    position="1",
                    duration=Duration(seconds=2.0),
                ),
                Event(tick=5000, kind="tap", position="1"),
            ],
            bpm_changes=[BpmChange(0, 200.0), BpmChange(960, 100.0)],
            end_tick=TICKS_PER_BAR * 4,
        )
        report = validate_chart(chart)
        self.assertTrue(report.ok, report.errors)

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

    def test_zero_duration_hold_is_legal(self):
        # simai accepts ``[1:0]``; official charts use it for instant holds.
        chart = make_chart(
            [Event(tick=0, kind="hold", position="1", duration=Duration(ticks=0))]
        )
        report = validate_chart(chart)
        self.assertTrue(report.ok, report.errors)

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
