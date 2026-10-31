import unittest
from pathlib import Path

from mainote.maidata import ParseError, parse_file, parse_maidata_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"


def parse_body(body: str, *, bpm: float = 120, difficulty: int = 5):
    text = (
        f"&title=test\n&wholebpm={bpm}\n&first=0\n"
        f"&inote_{difficulty}=({bpm}){body}\n"
    )
    return parse_maidata_text(text).chart(difficulty)


class TestSampleParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maidata = parse_file(SAMPLE)

    def test_metadata(self):
        self.assertEqual(self.maidata.get("title"), "ちっちゃな私")
        self.assertEqual(self.maidata.get("artist"), "マサラダ")
        self.assertEqual(self.maidata.get("wholebpm"), "143")
        self.assertEqual(self.maidata.get("first"), "0")

    def test_chart_headers(self):
        charts = {c.index: c for c in self.maidata.charts}
        self.assertEqual(sorted(charts), [2, 3, 4, 5])
        self.assertEqual(charts[2].level, "3.0")
        self.assertEqual(charts[3].level, "6.2")
        self.assertEqual(charts[4].level, "9.7")
        self.assertEqual(charts[5].level, "12.9")
        self.assertEqual(charts[4].designer, "はっぴー")
        self.assertEqual(charts[5].designer, "サファ太")

    def test_note_counts(self):
        charts = {c.index: c for c in self.maidata.charts}
        self.assertEqual(charts[2].note_count, 171)
        self.assertEqual(charts[3].note_count, 268)
        self.assertEqual(charts[4].note_count, 367)
        self.assertEqual(charts[5].note_count, 561)

    def test_tempo(self):
        for chart in self.maidata.charts:
            self.assertAlmostEqual(chart.initial_bpm, 143.0)
            self.assertEqual(len(chart.bpm_changes), 1)
            self.assertEqual(chart.bpm_changes[0].tick, 0)

    def test_divisions_and_max_simultaneous(self):
        charts = {c.index: c for c in self.maidata.charts}
        self.assertEqual(
            sorted({d.division for d in charts[5].division_changes}),
            [1, 2, 4, 8, 16, 24, 64],
        )
        counts = {}
        for event in charts[5].events:
            counts[event.tick] = counts.get(event.tick, 0) + 1
        self.assertEqual(max(counts.values()), 12)


class TestTokenSyntax(unittest.TestCase):
    def test_tap_times(self):
        chart = parse_body("{4}1,2,3,4,E")
        self.assertEqual([e.tick for e in chart.events], [0, 960, 1920, 2880])
        self.assertTrue(all(e.kind == "tap" for e in chart.events))

    def test_touch_vs_break(self):
        chart = parse_body("{4}B2,2b,E")
        by_pos = {e.position: e for e in chart.events}
        self.assertEqual(by_pos["B2"].kind, "touch")
        self.assertIn("break", by_pos["2"].flags)

    def test_touch_e_vs_end(self):
        chart = parse_body("{4}E1,E")
        self.assertEqual(len(chart.events), 1)
        self.assertEqual(chart.events[0].kind, "touch")
        self.assertEqual(chart.events[0].position, "E1")

    def test_each_vs_slide(self):
        chart = parse_body("{4}1/8,1-5[4:1],E")
        self.assertEqual(chart.events[0].kind, "tap")
        self.assertEqual(chart.events[1].kind, "tap")
        self.assertEqual(chart.events[2].kind, "slide")
        self.assertEqual(chart.events[2].position, "1")
        self.assertEqual(chart.events[2].slide[0].end, "5")

    def test_hold_duration(self):
        chart = parse_body("{4}7h[4:3],E")
        event = chart.events[0]
        self.assertEqual(event.kind, "hold")
        self.assertEqual(event.duration.ticks, 2880)
        self.assertEqual(event.duration.raw, "4:3")

    def test_same_start_chain(self):
        chart = parse_body("{4}1-3[4:1]*-4[8:1],E")
        event = chart.events[0]
        self.assertEqual(event.kind, "slide")
        self.assertEqual(event.chain_type, "same_start")
        self.assertEqual([s.end for s in event.slide], ["3", "4"])
        self.assertEqual(event.slide[0].duration.ticks, 960)
        self.assertEqual(event.slide[1].duration.ticks, 480)

    def test_path_chain(self):
        chart = parse_body("{4}4-1>2-7[16:7],E")
        event = chart.events[0]
        self.assertEqual(event.kind, "slide")
        self.assertEqual(event.chain_type, "path")
        self.assertEqual([s.end for s in event.slide], ["1", "2", "7"])
        self.assertEqual(event.duration.ticks, 1680)
        self.assertIsNone(event.slide[-1].duration)

    def test_absolute_durations(self):
        chart = parse_body("{4}1h[#1.5],2h[160#2],3h[3##1.5],E")
        self.assertEqual(chart.events[0].duration.seconds, 1.5)
        self.assertEqual(chart.events[1].duration.wait_bpm, 160.0)
        self.assertEqual(chart.events[1].duration.seconds, 2.0)
        self.assertEqual(chart.events[2].duration.wait_seconds, 3.0)
        self.assertEqual(chart.events[2].duration.seconds, 1.5)

    def test_division_change(self):
        chart = parse_body("{4}1,{8}2,3,E")
        self.assertEqual([e.tick for e in chart.events], [0, 960, 1440])

    def test_bpm_change(self):
        chart = parse_body("{4}1,(150){4}2,E")
        self.assertEqual([e.tick for e in chart.events], [0, 960])
        self.assertEqual(chart.bpm_changes[-1].tick, 960)
        self.assertAlmostEqual(chart.bpm_changes[-1].bpm, 150.0)

    def test_comments(self):
        chart = parse_body("{4}1, || comment\n2, >> old comment\nE")
        self.assertEqual(len(chart.events), 2)

    def test_multidigit_taps(self):
        chart = parse_body("{4}12,123,E")
        self.assertEqual(len(chart.events), 5)
        self.assertEqual(
            [e.position for e in chart.events], ["1", "2", "1", "2", "3"]
        )

    def test_break_and_ex_flags(self):
        chart = parse_body("{4}1bx,2x-5[4:1]b,3b-7[4:1]b,E")
        self.assertEqual(set(chart.events[0].flags), {"break", "ex"})
        slide = chart.events[1]
        self.assertIn("ex", slide.flags)
        self.assertIn("break", slide.slide[0].flags)
        slide2 = chart.events[2]
        self.assertIn("break", slide2.flags)
        self.assertIn("break", slide2.slide[0].flags)

    def test_invalid_token(self):
        with self.assertRaises(ParseError):
            parse_body("{4}1Q,E")

    def test_roundtrip_first_per_chart(self):
        text = (
            "&title=t\n&wholebpm=120\n&first=0\n"
            "&lv_5=12.0\n&des_5=x\n&first_5=1.5\n"
            "&inote_5=(120){4}1,E\n"
        )
        chart = parse_maidata_text(text).chart(5)
        self.assertAlmostEqual(chart.first, 1.5)


if __name__ == "__main__":
    unittest.main()
