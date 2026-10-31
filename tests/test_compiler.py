import unittest
from pathlib import Path

from mainote.maidata import compile_maidata, parse_file, parse_maidata_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"


def roundtrip(text: str):
    maidata = parse_maidata_text(text)
    compiled = compile_maidata(maidata)
    maidata2 = parse_maidata_text(compiled)
    return maidata, compiled, maidata2


class TestSampleRoundtrip(unittest.TestCase):
    def test_sample_roundtrip(self):
        maidata = parse_file(SAMPLE)
        compiled = compile_maidata(maidata)
        parsed = parse_maidata_text(compiled)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("E\n", compiled)
        self.assertEqual(compiled.count("&inote_"), 4)


class TestCompileFeatures(unittest.TestCase):
    def test_minimal_chart(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4}1,2,3,4,E\n"
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("(120){4}1,", compiled)
        self.assertTrue(compiled.rstrip().endswith("E"))

    def test_b2_vs_2b(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4}B2,2b,E\n"
        _, compiled, _ = roundtrip(text)
        self.assertIn("B2", compiled)
        self.assertIn("2b", compiled)
        self.assertNotIn("b2", compiled)

    def test_duration_forms(self):
        text = (
            "&title=t\n&wholebpm=120\n"
            "&inote_5=(120){4}7h[4:3],1-5[8:1],1h[#1.5],"
            "2h[160#2],3h[3##1.5],E\n"
        )
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("[4:3]", compiled)
        self.assertIn("[8:1]", compiled)
        self.assertIn("[#1.5]", compiled)
        self.assertIn("[160#2]", compiled)
        self.assertIn("[3##1.5]", compiled)

    def test_path_slide(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4}4-1>2-7[16:7],E\n"
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("4-1>2-7[16:7]", compiled)

    def test_same_start_chain(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4}1-3[4:1]*-4[8:1],E\n"
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("1-3[4:1]*-4[8:1]", compiled)

    def test_bpm_change(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4}1,(150){4}2,E\n"
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("(150)", compiled)

    def test_empty_chart(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4},E\n"
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("E", compiled)

    def test_special_flags(self):
        text = (
            "&title=t\n&wholebpm=120\n"
            "&inote_5=(120){4}1?-5[4:1],1!-6[4:1],1$-7[4:1],1@-8[4:1],"
            "1$,1`,1bx,1hx[4:1],1bxh[4:1],Cf,Chf[1:1],E\n"
        )
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        for token in ("1?-5", "1!-6", "1$-7", "1@-8", "1$", "1`", "1bx", "1hx", "1bxh", "Cf", "Chf"):
            self.assertIn(token, compiled)

    def test_multiple_charts_and_first(self):
        text = (
            "&title=t\n&wholebpm=120\n&first=0\n"
            "&lv_4=9.0\n&des_4=a\n&inote_4=(120){4}1,E\n"
            "&lv_5=12.0\n&des_5=b\n&first_5=1.25\n&inote_5=(120){4}2,E\n"
        )
        maidata = parse_maidata_text(text)
        compiled = compile_maidata(maidata)
        parsed = parse_maidata_text(compiled)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("&first_5=1.25", compiled)

    def test_no_trailing_rest_needed(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){4}1,E\n"
        _, compiled, _ = roundtrip(text)
        body = compiled.split("&inote_5=", 1)[1]
        self.assertLess(body.count(","), 8)

    def test_compiler_chooses_fine_division(self):
        text = "&title=t\n&wholebpm=120\n&inote_5=(120){16}1,,,,2,,,3,E\n"
        maidata, compiled, parsed = roundtrip(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertIn("{16}", compiled)


if __name__ == "__main__":
    unittest.main()
