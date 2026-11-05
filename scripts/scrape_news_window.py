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
]


def week_slices(start: str, end: str):
    """[(from, to_exclusive)] 7-day slices covering start..end inclusive, as YYYY-MM-DD strings."""
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    out = []
    while d <= last:
        nxt = min(d + timedelta(days=7), last + timedelta(days=1))
        out.append((d.isoformat(), nxt.isoformat()))
        d = nxt
    return out


def build_url(query: str, after: str, before: str) -> str:
    q = f'{query} after:{after} before:{before}'
    return f'https://news.google.com/rss/search?q={quote(q)}&hl=en-MY&gl=MY&ceid=MY:en'


def parse_items(xml_bytes: bytes, start: str, end: str) -> list:
    """Political items published inside [start, end]. Items with an unreadable date are dropped."""
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
        if not title or not (lo <= when < hi) or not is_political(title):
            continue
        source = f"GoogleNews-{item.findtext('source', 'Google News')}"
        rows.append({'source': source, 'source_lean': get_source_lean(source), 'title': title,
                     'description': '', 'url': (item.findtext('link') or '').strip(), 'published': pub,
                     'scraped_at': datetime.now().isoformat(), 'language': 'en', 'full_text': title})
    return rows


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """One row per headline (case and punctuation ignored)."""
    key = df['title'].str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
    return df.loc[~key.duplicated()].reset_index(drop=True)


def scrape_window(start: str, end: str, sleep: float = 1.0) -> pd.DataFrame:
    rows = []
    for query in QUERIES:
        for a, b in week_slices(start, end):
            try:
                r = requests.get(build_url(query, a, b), headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
                got = parse_items(r.content, start, end)
                print(f'  {query!r} {a}..{b}: {len(got)} political items')
                rows.extend(got)
            except Exception as e:
                print(f'  {query!r} {a}..{b} failed: {e}')
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
    print(df['source'].value_counts().head(8).to_string())
    if len(df) < 300:
        print('\nWARNING: fewer than 300 headlines. Add queries or widen the dates before using this window.')


if __name__ == '__main__':
    main()
