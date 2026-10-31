import unittest
from pathlib import Path

from mainote.maidata import chart_stats, maidata_stats, parse_file
from mainote.maidata.stats import density_for_level, profile_for_difficulty

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"


class TestStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maidata = parse_file(SAMPLE)
        cls.charts = {c.index: chart_stats(c) for c in cls.maidata.charts}

    def test_basic_stats(self):
        stats = self.charts[2]
        self.assertEqual(stats["notes_total"], 171)
        self.assertAlmostEqual(stats["notes_per_second"], 1.178, places=3)
        self.assertEqual(stats["max_simultaneous"], 2)
        self.assertEqual(stats["divisions"], [1, 2, 4])
        self.assertEqual(stats["counts"]["tap"], 141)
        self.assertEqual(stats["counts"]["hold"], 18)
        self.assertEqual(stats["counts"]["slide"], 4)
        self.assertEqual(stats["counts"]["touch"], 8)

    def test_master_stats(self):
        stats = self.charts[5]
        self.assertEqual(stats["notes_total"], 561)
        self.assertEqual(stats["notes_total_adx"], 567)
        self.assertEqual(stats["notes_total_wiki"], 649)
        self.assertEqual(stats["slide_segments"], 90)
        self.assertEqual(stats["max_simultaneous"], 12)
        self.assertEqual(stats["touch_count"], 110)
        self.assertEqual(stats["finest_division"], 64)
        self.assertAlmostEqual(stats["notes_per_second"], 3.876, places=3)

    def test_density_curve(self):
        self.assertAlmostEqual(density_for_level(3.0), 1.20, places=2)
        self.assertAlmostEqual(density_for_level(6.0), 1.85, places=2)
        self.assertAlmostEqual(density_for_level(10.0), 2.65, places=2)
        self.assertAlmostEqual(density_for_level(13.0), 3.90, places=2)
        self.assertGreater(density_for_level(12.0), density_for_level(9.0))

    def test_profile_clamps_to_difficulty(self):
        profile = profile_for_difficulty("basic", 12.9)
        low, high = profile["notes_per_second_range"]
        self.assertLessEqual(profile["notes_per_second"], high)
        self.assertGreaterEqual(profile["notes_per_second"], low)
        master = profile_for_difficulty("master", 12.9)
        self.assertGreater(master["notes_per_second"], profile["notes_per_second"])

    def test_maidata_stats(self):
        data = maidata_stats(self.maidata)
        self.assertEqual(data["title"], "ちっちゃな私")
        self.assertEqual(data["chart_count"], 4)
        self.assertEqual(len(data["charts"]), 4)


if __name__ == "__main__":
    unittest.main()
