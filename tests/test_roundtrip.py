import unittest
from pathlib import Path

from mainote.maidata import compile_maidata, parse_file, parse_maidata_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset" / "xxdw" / "maidata"

TOKEN_MATRIX = """\
&title=matrix
&artist=mainote
&wholebpm=150
&first=0.25
&clock_count=4
&lv_5=12.5
&des_5=test
&inote_5=(150){4}1,2h[4:1],3-5[8:1],E1,Cf,1/8,3b/7b,1bx,4hx[2:1],5-7[4:1]b,E
"""


class TestRoundtrip(unittest.TestCase):
    def test_sample_double_roundtrip(self):
        maidata = parse_file(SAMPLE)
        text1 = compile_maidata(maidata)
        maidata2 = parse_maidata_text(text1)
        text2 = compile_maidata(maidata2)
        maidata3 = parse_maidata_text(text2)
        self.assertEqual(maidata.signature(), maidata2.signature())
        self.assertEqual(maidata2.signature(), maidata3.signature())
        self.assertEqual(text1, text2)

    def test_token_matrix(self):
        maidata = parse_maidata_text(TOKEN_MATRIX)
        text = compile_maidata(maidata)
        parsed = parse_maidata_text(text)
        self.assertEqual(maidata.signature(), parsed.signature())
        self.assertEqual(parsed.get("title"), "matrix")
        self.assertEqual(parsed.get("first"), "0.25")
        chart = parsed.chart(5)
        self.assertEqual(chart.level, "12.5")
        self.assertEqual(chart.designer, "test")
        self.assertEqual(chart.note_count, 12)

    def test_metadata_order_independent(self):
        text1 = "&title=a\n&wholebpm=120\n&inote_5=(120){4}1,E\n"
        text2 = "&inote_5=(120){4}1,E\n&wholebpm=120\n&title=a\n"
        a = parse_maidata_text(text1)
        b = parse_maidata_text(text2)
        self.assertEqual(a.signature(), b.signature())


if __name__ == "__main__":
    unittest.main()
