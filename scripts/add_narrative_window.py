"""
scripts/add_narrative_window.py
===============================
Turn already-scored articles into ONE news window and save it to
data/processed/narrative_windows.csv. The sentiment page uses the two newest windows:
the earlier one is the reference, the later one is the current period.

    python scripts/add_narrative_window.py --label "Oct 2026" --start 2026-10-01 --end 2026-10-31

No API calls: score the articles first with sentiment/scoring/sentiment_scorer.py.
Adding the same label again replaces that window. Each window needs at least 300 successfully
scored headlines or the sentiment page will not apply an adjustment.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentiment.scoring.score_quality import window_row   # noqa: E402

SCORED = ROOT / 'data/processed/scored_articles.csv'
WINDOWS = ROOT / 'data/processed/narrative_windows.csv'


def add_window(scored_path, windows_path, label, start, end):
    row = window_row(pd.read_csv(scored_path), start, end, label)
    windows_path = Path(windows_path)
    if windows_path.exists():
        old = pd.read_csv(windows_path)
        old = old[old['label'] != label]
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        new = pd.DataFrame([row])
    new.sort_values('date').to_csv(windows_path, index=False)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--start', required=True, help='YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='YYYY-MM-DD, included')
    ap.add_argument('--scored', default=str(SCORED))
    ap.add_argument('--out', default=str(WINDOWS))
    a = ap.parse_args()
    row = add_window(a.scored, a.out, a.label, a.start, a.end)
    print(f"Saved window '{row['label']}' ({row['start']} to {row['date']}): {row['n_articles']} headlines")
    if row['n_articles'] < 300:
        print('WARNING: fewer than 300 headlines. The sentiment page will not use this window.')


if __name__ == '__main__':
    main()
