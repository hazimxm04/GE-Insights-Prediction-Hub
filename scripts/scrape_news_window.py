"""
scripts/scrape_news_window.py
=============================
Scrape Malaysian political headlines for ONE date window (for example all of October 2026)
from Google News RSS, so two windows can be compared.

    python scripts/scrape_news_window.py --label "Oct 2026" --start 2026-10-01 --end 2026-10-31

Why a fixed query list: both windows must be collected the SAME way. If one month used different
queries or outlets, the "change" between months would reflect the collection, not the news.
The queries cover all sides on purpose (government, BN, Harapan, PN, and the economy / race /
religion / corruption themes). Each query is cut into 7-day slices, because one request returns
at most about 100 items.

Languages: every query list is run twice, once against the English Malaysian edition and once against the
Malay one (Berita Harian, Utusan, Harian Metro, Sinar Harian and others). The Malay list mirrors the English
themes one for one. Each row carries language = 'en' or 'ms', so the scorer's work on Malay headlines can be
hand-checked separately. Use the SAME lists for every window, so the change between windows is the news.

Output: data/raw/news/window_<label>.csv (the 'window_' prefix keeps the older scorer's file
patterns from picking it up). Then score it with scripts/score_news_file.py.
"""
import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentiment.scrapers.news_scraper import is_political, get_source_lean   # noqa: E402

OUT_DIR = ROOT / 'data/raw/news'

QUERIES = [
    'Malaysia politics',
    'Anwar Ibrahim government',
    'Perikatan Nasional PAS Bersatu',
    'UMNO Barisan Nasional',
    'Pakatan Harapan DAP PKR',
    'Malaysia cost of living prices',
    'Malaysia corruption MACC',
    'Malay rights Islam Malaysia',
    'Malaysia race religion',
    'Johor Selangor Perak Melaka Negeri Sembilan election',
    # added 5 Nov 2026 to widen coverage (same queries for BOTH windows, so the comparison stays fair)
    'Malaysia economy inflation ringgit',
    'Dewan Rakyat Parliament Malaysia',
    'PAS Hadi Muhyiddin Bersatu',
    'DAP MCA MIC Malaysia',
    'Malaysia election commission SPR',
    'Malaysia budget subsidies government',
    # added 7 Nov 2026: the Malay_unity label had no query at all
    'Malay unity UMNO PAS cooperation',
]


# The Malay list mirrors the English one theme by theme (politics, each coalition, cost of living,
# corruption, race and religion, economy, parliament, election commission, budget).
QUERIES_MS = [
    'politik Malaysia',
    'kerajaan Anwar Ibrahim',
    'Perikatan Nasional PAS Bersatu',
    'UMNO Barisan Nasional',
    'Pakatan Harapan DAP PKR',
    'kos sara hidup harga barang',
    'rasuah SPRM',
    'hak Melayu Islam',
    'isu 3R kaum agama raja',
    'ekonomi inflasi ringgit',
    'Dewan Rakyat Parlimen',
    'pilihan raya SPR',
    'subsidi bajet kerajaan',
    # added 7 Nov 2026: the Malay_unity and Islam_threat labels ("Islam/Melayu terancam")
    'perpaduan Melayu UMNO PAS',
    'Islam terancam',
]

# Malay headlines: long words match anywhere, short party names and abbreviations only as whole words
# (so 'pas' matches PAS but not 'pasar').
MALAY_KEYWORDS = [
    'politik', 'menteri', 'anwar', 'madani', 'muhyiddin', 'zahid', 'najib',
    'pilihan raya', 'parlimen', 'dewan rakyat', 'dewan undangan negeri', 'ahli parlimen',
    'undi', 'kerajaan', 'pembangkang', 'rasuah', 'subsidi', 'bajet', 'kos sara hidup', 'harga barang',
    'hak melayu', 'isu 3r', 'islam', 'agama', 'kaum', 'perpaduan melayu', 'inflasi', 'pakatan harapan', 'perikatan nasional', 'barisan nasional', 'parti amanah',
]
# Party and coalition names that are ALSO everyday Malay words: Harapan (hope), Amanah (trust), Bersatu (united),
# Barisan (row), Perikatan (alliance).
# They count only when written as a proper noun (capital letter) somewhere after the first word of the
# headline: "Ahli Bersatu sertai ..." counts, "Harapan baharu untuk pesakit" and "bersatu padu" do not.
# PH (Pakatan Harapan) counts only in capitals, so the chemistry word pH is ignored.
# Left out as bare words: calon (candidate; a headline about an election candidate nearly always has another
# political word too) and melayu (culture and language stories); 'hak melayu' is in the list above.
MALAY_SHORT = ['3r', 'pas', 'bn', 'pn', 'dap', 'pkr', 'umno', 'prn', 'prk', 'pru', 'spr', 'sprm', 'dun']
_MALAY_SHORT_RE = re.compile(r'\b(' + '|'.join(MALAY_SHORT) + r')\b', re.IGNORECASE)
_MALAY_NAME_RE = re.compile(r'\b(Harapan|Amanah|Bersatu|Barisan|Perikatan)\b')
_PH_RE = re.compile(r'\bPH\b')


# The English filter in news_scraper.py has no theme words, so a headline about prices or corruption was dropped
# unless it also said "government" or "minister". These words (whole words, any case) let theme headlines in.
# Used here only; the older scraper's own list is unchanged.
_THEME_EN_RE = re.compile(
    r'\b(islam|islamic|racial|religious|3r|corruption|macc|graft|bribery|inflation|subsidy|subsidies'
    r'|cost of living|malay unity|malay rights)\b', re.IGNORECASE)


def is_political_en(text: str) -> bool:
    """The older political-word filter, plus the theme words above."""
    return is_political(text) or bool(_THEME_EN_RE.search(text))


def is_political_ms(text: str) -> bool:
    """True for a Malay (or mixed) headline about politics."""
    t = text.lower()
    if any(k in t for k in MALAY_KEYWORDS) or _MALAY_SHORT_RE.search(text) or _PH_RE.search(text):
        return True
    return any(m.start() > 0 for m in _MALAY_NAME_RE.finditer(text))


def week_slices(start: str, end: str):
    """[(from, to_exclusive)] 7-day slices covering start..end inclusive, as YYYY-MM-DD strings."""
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= last:
        nxt = min(d + timedelta(days=7), last + timedelta(days=1))
        out.append((d.isoformat(), nxt.isoformat()))
        d = nxt
    return out


def build_url(query: str, after: str, before: str, lang: str = 'en') -> str:
    q = f'{query} after:{after} before:{before}'
    return f'https://news.google.com/rss/search?q={quote(q)}&hl={lang}-MY&gl=MY&ceid=MY:{lang}'


def parse_items(xml_bytes: bytes, start: str, end: str, lang: str = 'en') -> list:
    """Political items published inside [start, end]. Items with an unreadable date are dropped.
    lang is 'en' or 'ms' and picks the political-word filter; it is stored in the 'language' column."""
    political = is_political_ms if lang == 'ms' else is_political_en
    lo = pd.Timestamp(start, tz='UTC')
    hi = pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)
    rows = []
    for item in ET.fromstring(xml_bytes).findall('.//item'):
        title = (item.findtext('title') or '').strip()
        pub = (item.findtext('pubDate') or '').strip()
        try:
            when = pd.Timestamp(parsedate_to_datetime(pub)).tz_convert('UTC')
        except Exception:
            continue
        if not title or not (lo <= when < hi) or not political(title):
            continue
        source = f"GoogleNews-{item.findtext('source', 'Google News')}"
        rows.append({'source': source, 'source_lean': get_source_lean(source), 'title': title,
                     'description': '', 'url': (item.findtext('link') or '').strip(), 'published': pub,
                     'scraped_at': datetime.now().isoformat(), 'language': lang, 'full_text': title})
    return rows


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per headline (case and punctuation ignored)."""
    key = df['title'].str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
    return df.loc[~key.duplicated()].reset_index(drop=True)


def scrape_window(start: str, end: str, sleep: float = 1.0, include_malay: bool = True) -> pd.DataFrame:
    """All English queries, then (unless include_malay is False) all Malay queries, for [start, end]."""
    plan = [(q, 'en') for q in QUERIES]
    if include_malay:
        plan += [(q, 'ms') for q in QUERIES_MS]
    rows = []
    for query, lang in plan:
        for a, b in week_slices(start, end):
            try:
                r = requests.get(build_url(query, a, b, lang), headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
                got = parse_items(r.content, start, end, lang)
                print(f'  [{lang}] {query!r} {a}..{b}: {len(got)} political items')
                rows.extend(got)
            except Exception as e:
                print(f'  [{lang}] {query!r} {a}..{b} failed: {e}')
            time.sleep(sleep)
    return dedupe(pd.DataFrame(rows)) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--start', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='YYYY-MM-DD, included')
    a = ap.parse_args()
    df = scrape_window(a.start, a.end)
    if df.empty:
        sys.exit('Nothing scraped. Check the network connection and try again.')
    name = re.sub(r'[^A-Za-z0-9]+', '_', a.label).strip('_')
    path = OUT_DIR / f'window_{name}.csv'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f'\nSaved {len(df)} unique headlines to {path}')
    print(df['language'].value_counts().to_string())
    print(df['source'].value_counts().head(8).to_string())
    if len(df) < 300:
        print('\nWARNING: fewer than 300 headlines. Add queries or widen the dates before using this window.')


if __name__ == '__main__':
    main()