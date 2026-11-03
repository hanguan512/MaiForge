import tempfile
import unittest
from collections import Counter
from pathlib import Path

from mainote.audio.timing import TimingInfo
from mainote.maidata import Chart, Duration, Event, compile_maidata, parse_file
from mainote.maidata.model import (
    BpmChange,
    TICKS_PER_BAR,
    TICKS_PER_BEAT,
    position_distance,
)
from mainote.maidata.validate import validate_chart
from mainote.generate.rule import (
    MAX_SIMULTANEOUS,
    RuleConfig,
    _choose_end,
    generate_chart,
)

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "numpy is required for audio tests")
class TestAudioPipeline(unittest.TestCase):
    def test_click_tempo_and_beats(self):
        from mainote.audio.features import analyze_audio
        from mainote.demo import make_click_track

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "click.wav"
            make_click_track(path, bpm=143.0, bars=16)
            timing = analyze_audio(str(path))
        self.assertLess(abs(timing.bpm - 143.0), 2.0)
        self.assertGreaterEqual(len(timing.beat_times), 60)
        self.assertGreaterEqual(len(timing.downbeat_times), 15)
        self.assertGreater(len(timing.onset_times), 50)
        self.assertGreater(timing.duration, 25.0)

    def test_high_tempo_is_not_folded_to_subharmonic(self):
        # A 190 BPM click track used to be estimated as ~126 BPM (a 3/2
        # subharmonic) because the old prior was centred on 130 BPM.
        from mainote.audio.features import analyze_audio
        from mainote.demo import make_click_track

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "click.wav"
            make_click_track(path, bpm=190.0, bars=16)
            timing = analyze_audio(str(path))
        self.assertLess(abs(timing.bpm - 190.0), 3.0)
        self.assertTrue(timing.bpm_candidates)
        self.assertLess(abs(timing.bpm_candidates[0]["bpm"] - 190.0), 3.0)

    def test_bpm_override(self):
        from mainote.audio.features import analyze_audio
        from mainote.demo import make_click_track

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "click.wav"
            make_click_track(path, bpm=143.0, bars=8)
            timing = analyze_audio(str(path), bpm=120.0, first=0.5)
        self.assertAlmostEqual(timing.bpm, 120.0)
        self.assertAlmostEqual(timing.first, 0.5)
        self.assertGreater(len(timing.beat_times), 10)

    def test_timing_candidates_roundtrip(self):
        timing = _synthetic_timing()
        timing.bpm_candidates = [
            {"bpm": 190.0, "score": 0.5, "correlation": 0.6},
            {"bpm": 126.0, "score": 0.4, "correlation": 0.5},
        ]
        loaded = TimingInfo.from_dict(timing.to_dict())
        self.assertEqual(loaded.bpm_candidates, timing.bpm_candidates)

    def test_retime(self):
        timing = _synthetic_timing()
        timing.retime(60.0)
        self.assertAlmostEqual(timing.bpm, 60.0)
        self.assertGreater(len(timing.beat_times), len(timing.downbeat_times))

    def test_rule_generation_valid_and_roundtrip(self):
        from mainote.audio.features import analyze_audio
        from mainote.demo import make_click_track
        from mainote.generate import maidata_for_chart

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "click.wav"
            make_click_track(path, bpm=143.0, bars=16)
            timing = analyze_audio(str(path))
        result = generate_chart(timing, difficulty="master", level=12.0, seed=3)
        chart = result.chart
        report = validate_chart(chart)
        self.assertTrue(report.ok, report.errors)
        self.assertGreater(chart.note_count, 50)
        self.assertLess(chart.note_count, 300)
        maidata = maidata_for_chart(
            chart, title="test", artist="mainote", wholebpm=timing.bpm
        )
        text = compile_maidata(maidata)
        parsed = parse_file(_write_temp(text))
        self.assertEqual(maidata.signature(), parsed.signature())

    def test_generation_deterministic(self):
        timing = _synthetic_timing()
        a = generate_chart(timing, difficulty="expert", level=9.5, seed=42)
        b = generate_chart(timing, difficulty="expert", level=9.5, seed=42)
        self.assertEqual(a.chart.signature(), b.chart.signature())

    def test_rule_layer_caps_simultaneous_notes_at_two(self):
        timing = _synthetic_timing()
        cases = [
            generate_chart(timing, difficulty="master", level=13.5, seed=7),
            generate_chart(
                timing,
                difficulty="master",
                level=13.5,
                seed=7,
                config=RuleConfig(
                    difficulty="master",
                    level=13.5,
                    seed=7,
                    max_simultaneous=4,
                ),
            ),
            generate_chart(timing, difficulty="remaster", level=14.5, seed=11),
        ]
        for result in cases:
            by_tick = Counter(event.tick for event in result.chart.events)
            self.assertLessEqual(
                max(by_tick.values(), default=0), MAX_SIMULTANEOUS
            )
            self.assertLessEqual(MAX_SIMULTANEOUS, 2)

    def test_slide_endpoints_follow_shape_rules(self):
        import random

        for pos in "12345678":
            for shape in ("s", "z", "w"):
                end = _choose_end(pos, shape, {pos}, random.Random(0))
                self.assertEqual(position_distance(pos, end), 4)
            for seed in range(20):
                end = _choose_end(pos, "-", {pos}, random.Random(seed))
                self.assertGreaterEqual(position_distance(pos, end), 2)

    def test_slide_prob_override(self):
        timing = _synthetic_timing()
        sparse = generate_chart(
            timing,
            difficulty="master",
            level=13.4,
            seed=7,
            config=RuleConfig(
                difficulty="master",
                level=13.4,
                seed=7,
                slide_prob=0.0,
            ),
        )
        self.assertEqual(
            sum(1 for e in sparse.chart.events if e.kind == "slide"), 0
        )
        with self.assertRaises(ValueError):
            generate_chart(
                timing,
                difficulty="master",
                level=13.4,
                seed=7,
                config=RuleConfig(
                    difficulty="master",
                    level=13.4,
                    seed=7,
                    slide_prob=1.5,
                ),
            )

    def test_density_scale_increases_notes(self):
        timing = _synthetic_timing()
        base = generate_chart(
            timing, difficulty="master", level=13.5, seed=7
        )
        dense = generate_chart(
            timing,
            difficulty="master",
            level=13.5,
            seed=7,
            config=RuleConfig(
                difficulty="master",
                level=13.5,
                seed=7,
                density_scale=1.5,
            ),
        )
        self.assertGreater(dense.chart.note_count, base.chart.note_count)
        self.assertGreater(
            dense.profile["notes_per_second"], 0.0
        )
        with self.assertRaises(ValueError):
            generate_chart(
                timing,
                difficulty="master",
                level=13.5,
                seed=7,
                config=RuleConfig(
                    difficulty="master",
                    level=13.5,
                    seed=7,
                    density_scale=0.0,
                ),
            )

    def test_alignment_offset(self):
        from mainote.dataset.align import score_chart_against_timing

        timing = _synthetic_timing(shift=0.0)
        chart = _synthetic_chart()
        result = score_chart_against_timing(timing, chart)
        self.assertGreater(result["confidence"], 0.5)
        self.assertLess(abs(result["best_offset_seconds"]), 0.02)
        self.assertGreater(result["matched_ratio"], 0.8)

    def test_demo_end_to_end(self):
        from mainote.demo import run_demo

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_demo(Path(tmp) / "demo")
            maidata_path = Path(summary["maidata"])
            self.assertTrue(maidata_path.exists())
            maidata = parse_file(maidata_path)
            self.assertEqual(len(maidata.charts), 1)
            self.assertGreater(summary["chart_notes"], 50)


def _write_temp(text: str) -> str:
    import tempfile

    fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    fh.write(text)
    fh.close()
    return fh.name


def _synthetic_timing(shift: float = 0.0) -> TimingInfo:
    # 120 BPM, one beat = 0.5 s.  Onsets every half second.
    onset_times = [1.0 + i * 0.5 for i in range(16)]
    return TimingInfo(
        path="synthetic.wav",
        sr=22050,
        duration=10.0,
        bpm=120.0,
        bpm_confidence=1.0,
        beat_times=[1.0 + i * 0.5 for i in range(16)],
        downbeat_times=[1.0 + i * 2.0 for i in range(4)],
        onset_times=[t + shift for t in onset_times],
        onset_strengths=[1.0] * len(onset_times),
        first=0.0,
        tags={"title": "synthetic"},
    )


def _synthetic_chart() -> Chart:
    # 1.0 s at 120 BPM = 2 beats = 1920 ticks.
    events = [
        Event(tick=int(round((1.0 + i * 0.5) * 2 * TICKS_PER_BEAT)), kind="tap", position=str(i % 8 + 1))
        for i in range(16)
    ]
    return Chart(
        index=5,
        level="12.0",
        initial_bpm=120.0,
        events=events,
        bpm_changes=[BpmChange(0, 120.0)],
        end_tick=max(e.end_tick() for e in events) + TICKS_PER_BAR,
    )


if __name__ == "__main__":
    unittest.main()
