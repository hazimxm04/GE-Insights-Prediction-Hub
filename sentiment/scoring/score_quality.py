"""
sentiment/scoring/score_quality.py
==================================
Failed scoring must never look like "no news about this theme".

Before: when the Groq call failed, timed out, or the reply could not be read, the scorer wrote all-zero
scores. A zero is a real value (the theme is absent), so failures silently pulled the national averages
down and would have biased any shift built on them.

Now: a failed article gets NaN scores and scoring_ok = False. Averages skip it, and the failure count is
reported. No groq or network code here, so it is tested in CI.
"""
import re

import numpy as np
import pandas as pd

# (field in the scored table, label in the model's reply, lowest value, highest value)
FIELDS = [
    ('bn_sentiment',     'BN',               -1.0, 1.0),
    ('harapan_sentiment', 'Harapan',          -1.0, 1.0),
    ('pn_sentiment',     'PN',               -1.0, 1.0),
    ('racial_tension',   'Tension',           0.0, 1.0),
    ('islam_threat',     'Islam_threat',      0.0, 1.0),
    ('malay_unity',      'Malay_unity',       0.0, 1.0),
    ('cost_living',      'Cost_living',       0.0, 1.0),
    ('corruption',       'Corruption',        0.0, 1.0),
    ('govt_performance', 'Govt_performance',  0.0, 1.0),
]
SCORE_FIELDS = [f[0] for f in FIELDS]
THEME_FIELDS = ['islam_threat', 'malay_unity', 'cost_living', 'corruption', 'govt_performance']


def failed_scores() -> dict:
    """What a failed article gets: missing values, not zeros."""
    return {k: float('nan') for k in SCORE_FIELDS}


def parse_scores(reply: str):
    """Read the nine scores from the model's reply. Returns None if any label is missing,
    so a half-readable reply counts as failed instead of filling the gap with 0."""
    out = {}
    for field, label, lo, hi in FIELDS:
        m = re.search(rf'{label}:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))', reply or '', re.IGNORECASE)   # the model often writes +0.8
        if not m:
            return None
        out[field] = max(lo, min(hi, float(m.group(1))))
    return out


def is_ok(scores: dict) -> bool:
    return not any(pd.isna(v) for v in scores.values())


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that were scored successfully. Tables written before this change have no
    scoring_ok column and are used as they are."""
    if 'scoring_ok' not in df.columns:
        return df
    return df[df['scoring_ok'].astype(str).str.lower().isin(['true', '1'])]


def failure_counts(df: pd.DataFrame):
    """(failed, total)."""
    total = len(df)
    return total - len(usable(df)), total


def theme_means(df: pd.DataFrame) -> dict:
    """Mean theme score over the successfully scored articles. Refuses to return numbers
    when nothing was scored, instead of returning zeros."""
    ok = usable(df)
    if ok.empty:
        raise ValueError('No article was scored successfully, so there are no theme scores. '
                         'Check the API key, the model name and the rate limit, then score again.')
    return {t: float(np.nanmean(ok[t].astype(float))) for t in THEME_FIELDS}


COALITION_FIELDS = ['bn_sentiment', 'harapan_sentiment', 'pn_sentiment']


def window_row(df: pd.DataFrame, start: str, end: str, label: str) -> dict:
    """One row for data/processed/narrative_windows.csv: theme means, coalition means and the
    headline count, over successfully scored articles published in [start, end] (inclusive).
    The count is of articles actually used, so the 300-headline rule sees the true number."""
    published = pd.to_datetime(df['published'], errors='coerce', utc=True)
    lo = pd.Timestamp(start, tz='UTC')
    hi = pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)          # end date included
    in_window = df[(published >= lo) & (published < hi)]
    ok = usable(in_window)
    if ok.empty:
        raise ValueError(f'No successfully scored article published between {start} and {end}.')
    row = {'date': end, 'label': label, 'start': start}
    row.update(theme_means(in_window))
    row.update({c: float(np.nanmean(ok[c].astype(float))) for c in COALITION_FIELDS})
    row['n_articles'] = len(ok)
    return row


def classify_api_error(message: str) -> str:
    """'daily' = out of tokens for today (stop and resume later), 'minute' = short per-minute
    limit (wait a few seconds and retry once), 'other' = anything else (access denied, timeout...)."""
    m = (message or '').lower()
    if 'rate_limit_exceeded' in m or 'rate limit' in m or '429' in m:
        if 'per day' in m or '(tpd)' in m or 'requests per day' in m or '(rpd)' in m:
            return 'daily'
        return 'minute'
    return 'other'


def done_titles(scored: pd.DataFrame) -> set:
    """Titles already scored successfully (used to resume an interrupted scoring run)."""
    return set(usable(scored)['title'])