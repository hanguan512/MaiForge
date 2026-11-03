import unittest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from mainote.maidata import Duration, Event
    from mainote.ml.infer import _repair_events


@unittest.skipUnless(HAS_TORCH, "torch is required")
class TestRepairEvents(unittest.TestCase):
    def test_caps_outer_chords_but_keeps_touch_swipes(self):
        events = [
            Event(tick=0, kind="tap", position="1"),
            Event(tick=0, kind="tap", position="2"),
            Event(tick=0, kind="tap", position="3"),
            Event(tick=0, kind="touch", position="A1"),
            Event(tick=0, kind="touch", position="A2"),
        ]
        repaired, warnings = _repair_events(events)
        kinds = [event.kind for event in repaired]
        self.assertEqual(kinds.count("tap"), 2)
        self.assertEqual(kinds.count("touch"), 2)
        self.assertTrue(any("outer notes at one tick" in w for w in warnings))

    def test_keeps_structural_notes_before_taps(self):
        events = [
            Event(tick=0, kind="tap", position="1"),
            Event(tick=0, kind="tap", position="2"),
            Event(tick=0, kind="hold", position="3", duration=Duration(ticks=960)),
        ]
        repaired, _ = _repair_events(events)
        self.assertEqual(len(repaired), 2)
        self.assertIn("hold", [event.kind for event in repaired])

    def test_cap_is_per_tick(self):
        events = [
            Event(tick=0, kind="tap", position="1"),
            Event(tick=0, kind="tap", position="2"),
            Event(tick=20, kind="tap", position="3"),
            Event(tick=20, kind="tap", position="4"),
        ]
        repaired, warnings = _repair_events(events)
        self.assertEqual(len(repaired), 4)
        self.assertFalse(any("outer notes at one tick" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
