"""
scripts/score_news_file.py
==========================
Score ONE raw news file with the Groq scorer and write a separate scored file. It does not touch
data/processed/scored_articles.csv or the other scorer outputs.

    python scripts/score_news_file.py data/raw/news/window_Oct_2026.csv data/processed/scored_window_Oct_2026.csv

It can be stopped and run again: articles already scored successfully in the output file are kept
and skipped, and the file is saved every 20 articles. If Groq's daily token allowance runs out, it
stops cleanly; run the same command again later (the allowance refills over time) to finish.
Failed articles are kept with scoring_ok = False and are left out of every average, and are retried
on the next run.

Use the SAME Groq model for every window you compare (set GROQ_MODEL if you change it).

Then:
    python scripts/add_narrative_window.py --label "Oct 2026" --start 2026-10-01 --end 2026-10-31 \
        --scored data/processed/scored_window_Oct_2026.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentiment.scoring.score_quality import usable, done_titles   # noqa: E402

CHUNK = 20


def pending(raw: pd.DataFrame, scored: pd.DataFrame = None) -> pd.DataFrame:
    """Articles still to score: everything not already scored successfully."""
    if scored is None or scored.empty:
        return raw.reset_index(drop=True)
    return raw[~raw['title'].isin(done_titles(scored))].reset_index(drop=True)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    raw = pd.read_csv(src).drop_duplicates('title').reset_index(drop=True)
    previous = pd.read_csv(dst) if dst.exists() else pd.DataFrame()
    kept = usable(previous) if not previous.empty else previous
    todo = pending(raw, previous)
    print(f'{len(raw)} articles in the file; {len(kept)} already scored; {len(todo)} to score now.')
    if todo.empty:
        print('Nothing to do.')
        return

    from sentiment.scoring.sentiment_scorer import score_dataframe    # needs groq + GROQ_API_KEY
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = kept
    for start in range(0, len(todo), CHUNK):
        chunk = todo.iloc[start:start + CHUNK].reset_index(drop=True)
        scored = score_dataframe(chunk)
        done = pd.concat([done, scored], ignore_index=True)
        done.to_csv(dst, index=False)
        print(f'  saved progress: {len(usable(done))} scored successfully so far -> {dst}')
        if len(scored) < len(chunk):
            print('\nStopped early (daily limit). Run the same command again later to finish.')
            break
    ok = len(usable(done))
    print(f'\nDone for now: {ok} scored, {len(done) - ok} failed (retried next run), {len(raw) - ok} still missing.')


if __name__ == '__main__':
    main()