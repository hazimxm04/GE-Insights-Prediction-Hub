import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scrape_news_window as sw

XML = b"""<rss><channel>
<item><title>Harapan and BN clash over election seats</title><link>u1</link>
 <pubDate>Wed, 07 Oct 2026 03:00:00 GMT</pubDate><source>Malay Mail</source></item>
<item><title>Harapan and BN clash over election seats</title><link>u2</link>
 <pubDate>Thu, 08 Oct 2026 03:00:00 GMT</pubDate><source>FMT</source></item>
<item><title>Minister speaks on prices</title><link>u3</link>
 <pubDate>Mon, 02 Nov 2026 03:00:00 GMT</pubDate><source>FMT</source></item>
<item><title>Football final tonight</title><link>u4</link>
 <pubDate>Wed, 07 Oct 2026 03:00:00 GMT</pubDate><source>FMT</source></item>
<item><title>Minister on subsidies</title><link>u5</link>
 <pubDate>garbage</pubDate><source>FMT</source></item>
</channel></rss>"""


def test_week_slices_cover_window_without_gaps():
    s = sw.week_slices('2026-10-01', '2026-10-31')
    assert s[0][0] == '2026-10-01' and s[-1][1] == '2026-11-01'
    assert all(s[i][1] == s[i + 1][0] for i in range(len(s) - 1))


def test_url_has_date_operators():
    u = sw.build_url('Malaysia politics', '2026-10-01', '2026-10-08')
    assert 'after%3A2026-10-01' in u and 'before%3A2026-10-08' in u


def test_parse_keeps_only_political_items_inside_window():
    rows = sw.parse_items(XML, '2026-10-01', '2026-10-31')
    titles = [r['title'] for r in rows]
    assert len(rows) == 2 and 'Football final tonight' not in titles   # out-of-window, non-political, bad date dropped
    assert rows[0]['source'] == 'GoogleNews-Malay Mail'


def test_dedupe_ignores_case_and_punctuation():
    df = pd.DataFrame({'title': ['Anwar: BN wins!', 'anwar bn wins', 'Other news']})
    assert len(sw.dedupe(df)) == 2
