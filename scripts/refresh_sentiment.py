"""
scripts/refresh_sentiment.py
============================
One command to bring the sentiment view up to date. Run it when YOU want (it is not scheduled):

    python scripts/refresh_sentiment.py                 # scrape, score up to 150 NEW headlines, rebuild windows
    python scripts/refresh_sentiment.py --max-new 60    # smaller run
    python scripts/refresh_sentiment.py --no-scrape     # skip the web, just score what is waiting

    # Pre-election test windows (to test the sentiment layer against a real result):
    python scripts/refresh_sentiment.py --state johor --max-new 300
    python scripts/refresh_sentiment.py --state neg_sembilan --max-new 300

What it does
  1. Scrapes the last 60 days of headlines (same fixed queries every time) into a raw pool.
  2. Scores only headlines that are NOT scored yet, newest first, at most --max-new per run
     (default 150, which fits Groq's free daily allowance). Re-running adds more.
  3. Rebuilds two rolling windows from the scored pool: the LATEST 30 days and the 30 days
     BEFORE them. The prediction page compares these two, so "today's sentiment" moves forward
     by itself as news arrives.
  4. Updates data/processed/narrative_windows.csv, which also feeds the "last updated" line.

Notes
  - The first time, it reuses any data/processed/scored_window_*.csv you already scored.
  - A window needs >= 300 successfully scored headlines; below that the page ignores it and
    this script tells you how many more are needed.
  - Rolling windows REPLACE the older national rows (Sep/Oct 2026 hand-made windows). Rows for a
    single state (pre-election test windows) are left alone.
  - With --state, the two 30-day windows END THE DAY BEFORE that state's polling day (news from
    polling day itself is never used). Only that state's two rows are written or replaced; the national
    rows and other states are left alone. Headlines are scored once even when two windows overlap
    (Johor's last window and Negeri Sembilan's earlier window share some weeks).
  - Use the same Groq model every time (set GROQ_MODEL if you change it).
"""
import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from sentiment.scoring.score_quality import usable, done_titles, window_row   # noqa: E402

RAW_POOL = ROOT / 'data/raw/news/pool_raw.csv'
SCORED_POOL = ROOT / 'data/processed/scored_pool.csv'
WINDOWS = ROOT / 'data/processed/narrative_windows.csv'
WINDOW_DAYS = 30
SCRAPE_DAYS = 2 * WINDOW_DAYS
MIN_HEADLINES = 300
CHUNK = 20
CURRENT_LABEL = 'Latest 30 days'
REFERENCE_LABEL = 'Previous 30 days'
STATE_CURRENT_LABEL = 'Last 30 days before the election'
STATE_REFERENCE_LABEL = 'Previous 30 days'
# Polling days of the 2026 state elections already held (override with --election-date)
ELECTION_DATES = {'johor': date(2026, 7, 11), 'neg_sembilan': date(2026, 8, 1)}


def state_end_date(state: str, election_date: str = None) -> date:
    """Last day of news for a pre-election window: the day BEFORE polling day."""
    poll = date.fromisoformat(election_date) if election_date else ELECTION_DATES[state]
    return poll - timedelta(days=1)


def rolling_dates(today: date):
    """(current_start, current_end, reference_start, reference_end), both windows 30 days long."""
    cur_end = today
    cur_start = today - timedelta(days=WINDOW_DAYS - 1)
    ref_end = cur_start - timedelta(days=1)
    ref_start = ref_end - timedelta(days=WINDOW_DAYS - 1)
    return cur_start, cur_end, ref_start, ref_end


def merge_raw(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add newly scraped headlines to the raw pool; one row per title."""
    both = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if both.empty:
        return both
    return both.drop_duplicates('title').reset_index(drop=True)


def pick_new(raw: pd.DataFrame, pool: pd.DataFrame, max_new: int, today: date) -> pd.DataFrame:
    """Headlines still to score: published inside the two windows, not already scored OK,
    newest first, at most max_new."""
    if raw.empty:
        return raw
    cur_start, cur_end, ref_start, ref_end = rolling_dates(today)
    when = pd.to_datetime(raw['published'], errors='coerce', utc=True)
    lo = pd.Timestamp(ref_start, tz='UTC')
    hi = pd.Timestamp(cur_end, tz='UTC') + pd.Timedelta(days=1)
    r = raw.assign(_when=when)
    r = r[(r['_when'] >= lo) & (r['_when'] < hi)]
    if pool is not None and not pool.empty:
        r = r[~r['title'].isin(done_titles(pool))]
    r = r.sort_values('_when', ascending=False).head(max_new).drop(columns='_when')
    return r.reset_index(drop=True)


def rebuild_windows(pool: pd.DataFrame, windows_path, today: date, state: str = ''):
    """Write two 30-day windows ending on `today`. With state='' they are the national rolling windows
    (older national rows are replaced, state rows kept). With a state name they are that state's
    pre-election windows (only that state's rows are replaced; national and other states are kept).
    Returns [(label, n_usable)] so the caller can report. Raises ValueError if a window is empty."""
    cur_start, cur_end, ref_start, ref_end = rolling_dates(today)
    ref_label, cur_label = (STATE_REFERENCE_LABEL, STATE_CURRENT_LABEL) if state else (REFERENCE_LABEL, CURRENT_LABEL)
    spans = [(ref_label, ref_start, ref_end), (cur_label, cur_start, cur_end)]
    rows = []
    for label, a, b in spans:
        row = window_row(pool, a.isoformat(), b.isoformat(), label)
        row['state'] = state
        rows.append(row)
    windows_path = Path(windows_path)
    keep = pd.DataFrame()
    if windows_path.exists():
        old = pd.read_csv(windows_path)
        if 'state' not in old.columns:
            old['state'] = ''
        keep = old[old['state'].fillna('').astype(str).str.strip() != state]
    out = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True) if not keep.empty else pd.DataFrame(rows)
    out.sort_values('date').to_csv(windows_path, index=False)
    return [(r['label'], int(r['n_articles'])) for r in rows]


def seed_pool(pool_path: Path, processed_dir: Path) -> pd.DataFrame:
    """First run: reuse scored_window_*.csv files so earlier scoring is not wasted."""
    if pool_path.exists():
        return pd.read_csv(pool_path)
    parts = [pd.read_csv(p) for p in sorted(processed_dir.glob('scored_window_*.csv'))]
    if not parts:
        return pd.DataFrame()
    pool = pd.concat(parts, ignore_index=True)
    return pool.drop_duplicates('title', keep='last').reset_index(drop=True)


def score_into_pool(todo: pd.DataFrame, pool: pd.DataFrame, pool_path: Path, score_fn):
    """Score in chunks, saving after each. Stops cleanly when the scorer returns a short chunk
    (daily limit). Returns (pool, stopped_early)."""
    done = usable(pool) if not pool.empty else pool
    for start in range(0, len(todo), CHUNK):
        chunk = todo.iloc[start:start + CHUNK].reset_index(drop=True)
        scored = score_fn(chunk)
        done = pd.concat([done, scored], ignore_index=True) if not done.empty else scored
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(pool_path, index=False)
        print(f'  saved progress: {len(usable(done))} scored successfully -> {pool_path}')
        if len(scored) < len(chunk):
            return done, True
    return done, False


def refresh(today: date, max_new=150, scrape=True, scrape_fn=None, score_fn=None,
            raw_path=RAW_POOL, pool_path=SCORED_POOL, windows_path=WINDOWS, processed_dir=None, state=''):
    """`today` is the LAST day of news: today's date for the national windows, or the day before
    polling day when `state` is given."""
    raw_path, pool_path, windows_path = Path(raw_path), Path(pool_path), Path(windows_path)
    processed_dir = Path(processed_dir) if processed_dir else pool_path.parent
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    if scrape:
        _, _, ref_start, _ = rolling_dates(today)
        fresh = scrape_fn(ref_start.isoformat(), today.isoformat())
        raw = merge_raw(raw, fresh)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, index=False)
        print(f'Raw pool: {len(raw)} headlines')
    pool = seed_pool(pool_path, processed_dir)
    todo = pick_new(raw, pool, max_new, today)
    print(f'{len(usable(pool)) if not pool.empty else 0} already scored; scoring up to {max_new} new '
          f'({len(todo)} waiting in the two windows).')
    stopped = False
    if not todo.empty:
        pool, stopped = score_into_pool(todo, pool, pool_path, score_fn)
    elif not pool.empty:
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(pool_path, index=False)
    if pool.empty:
        print('Nothing scored yet, so no windows were built.')
        return {'windows': None, 'stopped': stopped}
    try:
        counts = rebuild_windows(pool, windows_path, today, state)
    except ValueError as e:
        print(f'Windows not rebuilt: {e}')
        return {'windows': None, 'stopped': stopped}
    for label, n in counts:
        flag = '' if n >= MIN_HEADLINES else f'  <-- needs {MIN_HEADLINES - n} more; page ignores it until then'
        print(f'  {label}: {n} headlines scored{flag}')
    if stopped:
        print('\nStopped early (Groq daily limit). Run again later; the page keeps the windows above.')
    return {'windows': counts, 'stopped': stopped}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--max-new', type=int, default=150, help='most NEW headlines to score this run')
    ap.add_argument('--no-scrape', action='store_true', help='skip scraping, just score and rebuild')
    ap.add_argument('--today', default=None, help='YYYY-MM-DD (default: today), for testing')
    ap.add_argument('--state', choices=sorted(ELECTION_DATES), default=None,
                    help='build this state\'s PRE-ELECTION windows instead of the national rolling ones')
    ap.add_argument('--election-date', default=None, help='YYYY-MM-DD polling day (default: the known 2026 date)')
    a = ap.parse_args()
    if a.state:
        today = state_end_date(a.state, a.election_date)
        print(f'{a.state}: windows end {today} (the day before polling day)')
    else:
        today = date.fromisoformat(a.today) if a.today else date.today()

    scrape_fn = score_fn = None
    if not a.no_scrape:
        from scrape_news_window import scrape_window
        scrape_fn = scrape_window
    from sentiment.scoring.sentiment_scorer import score_dataframe      # needs groq + GROQ_API_KEY
    score_fn = score_dataframe
    refresh(today, a.max_new, not a.no_scrape, scrape_fn, score_fn, state=a.state or '')


if __name__ == '__main__':
    main()