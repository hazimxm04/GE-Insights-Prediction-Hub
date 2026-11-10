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


XML_MS = b"""<rss><channel>
<item><title>PAS tolak usul kerajaan di Dewan Rakyat</title><link>m1</link>
 <pubDate>Wed, 07 Oct 2026 03:00:00 GMT</pubDate><source>Sinar Harian</source></item>
<item><title>Pasar malam dibuka semula</title><link>m2</link>
 <pubDate>Wed, 07 Oct 2026 03:00:00 GMT</pubDate><source>Harian Metro</source></item>
<item><title>Kos sara hidup naik, rakyat tertekan</title><link>m3</link>
 <pubDate>Thu, 08 Oct 2026 03:00:00 GMT</pubDate><source>Berita Harian</source></item>
</channel></rss>"""


def test_malay_url_uses_the_malay_edition():
    u = sw.build_url('politik Malaysia', '2026-10-01', '2026-10-08', 'ms')
    assert 'hl=ms-MY' in u and 'ceid=MY:ms' in u
    assert 'hl=en-MY' in sw.build_url('Malaysia politics', '2026-10-01', '2026-10-08')   # default unchanged


def test_malay_filter_uses_whole_words_for_short_names():
    assert sw.is_political_ms('PAS tolak usul')
    assert not sw.is_political_ms('Pasar malam dibuka semula')       # "pas" inside a word
    assert sw.is_political_ms('Kos sara hidup naik')


def test_malay_rows_are_tagged_with_their_language():
    rows = sw.parse_items(XML_MS, '2026-10-01', '2026-10-31', 'ms')
    assert [r['language'] for r in rows] == ['ms', 'ms'] and len(rows) == 2
    assert all(r['language'] == 'en' for r in sw.parse_items(XML, '2026-10-01', '2026-10-31'))


def test_scrape_window_runs_both_query_lists(monkeypatch):
    seen = []

    class R:
        content = b'<rss><channel></channel></rss>'

    def fake_get(url, **kw):
        seen.append(url)
        return R()
    monkeypatch.setattr(sw.requests, 'get', fake_get)
    sw.scrape_window('2026-10-01', '2026-10-07', sleep=0)
    n = len(sw.week_slices('2026-10-01', '2026-10-07'))
    assert len(seen) == (len(sw.QUERIES) + len(sw.QUERIES_MS)) * n
    assert sum('ceid=MY%3Ams' in u or 'ceid=MY:ms' in u for u in seen) == len(sw.QUERIES_MS) * n
    seen.clear()
    sw.scrape_window('2026-10-01', '2026-10-07', sleep=0, include_malay=False)
    assert len(seen) == len(sw.QUERIES) * n


def test_everyday_malay_words_are_not_treated_as_political():
    assert not sw.is_political_ms('Harapan baharu untuk pesakit kanser')      # harapan = hope
    assert not sw.is_political_ms('Pasukan bersatu menang perlawanan')        # bersatu = united
    assert not sw.is_political_ms('Pasar malam bermula 8 pm')
    assert not sw.is_political_ms('Calon SPM terima keputusan')                # exam candidates
    assert sw.is_political_ms('Pakatan Harapan umum calon DUN')
    assert sw.is_political_ms('Isu 3R: kerajaan beri amaran')


def test_party_names_count_only_as_proper_nouns():
    assert sw.is_political_ms('Ahli Bersatu sertai Amanah')              # party names, capitalised, not first word
    assert sw.is_political_ms('Rakyat sokong Harapan di Johor')
    assert sw.is_political_ms('PH dan BN bersaing di kerusi ini')
    assert not sw.is_political_ms('Harapan baharu untuk pesakit kanser')   # first word: everyday "hope"
    assert not sw.is_political_ms('Amanah diberi kepada pengurus baharu')
    assert not sw.is_political_ms('Rakyat berharap dan bersatu padu')      # lower case, everyday words
    assert not sw.is_political_ms('Nilai pH air sungai menurun')           # chemistry


def test_barisan_and_perikatan_count_as_names_not_as_plain_words():
    assert sw.is_political_ms('Rakyat percaya Barisan akan menang')
    assert sw.is_political_ms('Ahli Perikatan keluar parti')
    assert not sw.is_political_ms('Barisan hadapan pasukan bola sepak')    # first word, everyday "row"
    assert not sw.is_political_ms('Mereka duduk dalam barisan panjang')    # lower case


def test_theme_headlines_pass_without_a_party_or_minister_word():
    assert sw.is_political_en('Rising prices and inflation hit families')
    assert sw.is_political_en('MACC arrests senior officer over graft')
    assert sw.is_political_en('Racial remarks spark outrage')
    assert not sw.is_political_en('Football final tonight')
    assert sw.is_political_ms('Isu agama dan kaum dibangkitkan semula')
    assert sw.is_political_ms('Isu 3R: polis siasat kenyataan itu')
    assert sw.is_political_ms('Perpaduan Melayu diuji selepas pilihan raya')
    assert not sw.is_political_ms('Pasar malam dibuka semula')


def test_the_unity_and_islam_queries_exist():
    assert any('unity' in q.lower() for q in sw.QUERIES)
    assert 'perpaduan Melayu UMNO PAS' in sw.QUERIES_MS and 'Islam terancam' in sw.QUERIES_MS