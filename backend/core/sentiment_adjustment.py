"""
backend/core/sentiment_adjustment.py
====================================
EXPERIMENTAL. A sentiment-informed prediction that sits BESIDE the frozen forecast.

What it is
  News themes (islam_threat, malay_unity, cost_living, corruption) are scored for a current
  period. Each theme is assumed to land differently on different voter groups (older Malay,
  younger Malay, Chinese, Indian, other). A seat's shift is the sum over its voter groups of
  (group share x group reaction x change in the theme). The shift moves the seat's win
  probability on the logit scale.

What it is NOT
  - Not trained. Every number in REACTIONS and the strengths below is an ASSUMPTION written by
    a person. There is no past news paired with past results to fit them on, so no accuracy can
    be claimed for the adjusted numbers. The headline accuracy belongs to the frozen model.
  - Not applied to levels. Only the CHANGE in a theme between two windows counts, because
    the level of a theme depends on which outlets were scraped.
  - Never written over the frozen forecast. Output has new columns; the frozen file is unchanged.

Direction convention
  Probabilities are P(opposition wins), the same as the frozen forecast. A reaction > 0 moves
  a group toward the opposition (Harapan), < 0 toward BN/PN. Anti-incumbent themes (cost of
  living, corruption) take their sign from who held the seat last time, so both sides can lose.

Safety rules (each is tested)
  - Needs MIN_HEADLINES in both windows, otherwise the shift is 0 and the status says why.
  - govt_performance has zero weight (too few labelled examples).
  - islam_threat is down-weighted (low-reliability label).
  - The total shift is capped at MAX_ABS_SHIFT on the logit scale.
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEGMENTS = ['older_malay', 'young_malay', 'chinese', 'indian', 'other']
THEMES = ['islam_threat', 'malay_unity', 'cost_living', 'corruption', 'govt_performance']

MIN_HEADLINES = 300          # per window; below this the theme means are too noisy
STRENGTH = 4.0               # logit change per +1.0 change in a theme share, at full reaction (ASSUMED)
LEAN_STRENGTH = 1.5          # logit per +1.0 change in coalition lean (ASSUMED)
LEAN_RELIABILITY = 0.5       # the lean comes from the same unchecked LLM scorer, so it is halved
# How strongly each group follows a change in coalition coverage (ASSUMED). Older Malay voters are
# assumed least moved by news, younger and non-Malay voters most.
LEAN_GROUP_WEIGHT = {'older_malay': 0.6, 'young_malay': 0.9, 'chinese': 1.0, 'indian': 1.0, 'other': 1.0}
MAX_ABS_SHIFT = 0.5          # logit; about 12 points near 50 %, much less near 0 or 1
PROB_FLOOR, PROB_CEIL = 0.01, 0.99   # frozen probabilities of exactly 0 or 1 cannot move otherwise

# Reliability multiplier per theme. 0 = unused.
RELIABILITY = {
    'islam_threat': 0.5,       # subjective label, flagged low-reliability
    'malay_unity': 1.0,
    'cost_living': 1.0,
    'corruption': 1.0,
    'govt_performance': 0.0,   # about 14 labelled examples: not used
}

# Identity themes: fixed direction per group (+ toward opposition/Harapan, - toward BN/PN). ASSUMED.
REACTIONS = {
    'islam_threat': {'older_malay': -1.0, 'young_malay': -0.6, 'chinese': 0.5, 'indian': 0.5, 'other': 0.3},
    'malay_unity':  {'older_malay': -0.8, 'young_malay': -0.5, 'chinese': 0.4, 'indian': 0.4, 'other': 0.2},
}
# Anti-incumbent themes: size per group (ASSUMED); the sign comes from the seat holder.
ANTI_INCUMBENT = {
    'cost_living': {'older_malay': 0.4, 'young_malay': 0.8, 'chinese': 0.6, 'indian': 0.6, 'other': 0.6},
    'corruption':  {'older_malay': 0.4, 'young_malay': 0.8, 'chinese': 0.6, 'indian': 0.6, 'other': 0.6},
}


def _logit(p):
    p = min(max(p, PROB_FLOOR), PROB_CEIL)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def segment_shares(row) -> dict:
    """Voter-group shares for a seat from the ethnicity file columns. They sum to 1.
    Malay share is split young/older by the age columns; 'other' is what is left."""
    malay = float(row['malay_pct'])
    young, older = float(row['young_malay_pct']), float(row['older_malay_pct'])
    young_frac = young / (young + older) if (young + older) > 0 else 0.5
    chinese, indian = float(row['chinese_pct']), float(row['indian_pct'])
    other = max(0.0, 1.0 - malay - chinese - indian)
    shares = {'older_malay': malay * (1 - young_frac), 'young_malay': malay * young_frac,
              'chinese': chinese, 'indian': indian, 'other': other}
    total = sum(shares.values())
    return {k: v / total for k, v in shares.items()} if total > 0 else shares


def coalition_lean(window: dict):
    """Net lean toward Harapan in a window, like a 'points toward one party' measure:
    Harapan sentiment minus the average of BN and PN sentiment (each -1..+1).
    Positive = coverage is more favourable to Harapan than to BN/PN. None if the window lacks the columns."""
    try:
        return float(window['harapan_sentiment']) - (float(window['bn_sentiment']) + float(window['pn_sentiment'])) / 2
    except (KeyError, TypeError, ValueError):
        return None


def lean_change(current: dict, reference: dict) -> float:
    """Change in coalition lean between windows; 0 if either window lacks coalition sentiment."""
    a, b = coalition_lean(current), coalition_lean(reference)
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return 0.0
    return a - b


def theme_change(current: dict, reference: dict) -> dict:
    """Change in each theme share between two windows. Only change is used, never level."""
    return {t: float(current[t]) - float(reference[t]) for t in THEMES}


def data_status(current: dict, reference: dict, min_headlines: int = MIN_HEADLINES):
    """(ok, message). Both windows need enough headlines."""
    n_cur, n_ref = int(current.get('n_articles', 0)), int(reference.get('n_articles', 0))
    if n_cur < min_headlines or n_ref < min_headlines:
        return False, (f'Not enough headlines for a stable theme average '
                       f'(current {n_cur}, reference {n_ref}, need {min_headlines} in each). No adjustment applied.')
    return True, f'OK (current {n_cur}, reference {n_ref} headlines)'


def seat_shift(shares: dict, change: dict, held_by_opposition: bool, lean_delta: float = 0.0) -> float:
    """Logit shift for one seat. Positive = toward opposition (Harapan)."""
    holder_sign = -1.0 if held_by_opposition else 1.0   # anti-incumbent pushes AWAY from the holder
    total = 0.0
    for theme, d in change.items():
        rel = RELIABILITY.get(theme, 0.0)
        if rel == 0.0:
            continue
        for seg, share in shares.items():
            if theme in REACTIONS:
                reaction = REACTIONS[theme][seg]
            elif theme in ANTI_INCUMBENT:
                reaction = holder_sign * ANTI_INCUMBENT[theme][seg]
            else:
                continue
            total += share * reaction * d * rel * STRENGTH
    for seg, share in shares.items():          # coalition lean: same direction, group-specific size
        total += share * LEAN_GROUP_WEIGHT[seg] * lean_delta * LEAN_STRENGTH * LEAN_RELIABILITY
    return max(-MAX_ABS_SHIFT, min(MAX_ABS_SHIFT, total))


def adjust_forecast(forecast: pd.DataFrame, demographics: pd.DataFrame,
                    current: dict, reference: dict, min_headlines: int = MIN_HEADLINES):
    """Add sentiment-informed columns to a frozen forecast (the input is not modified).

    forecast      : frozen forecast (needs seat_no, probability, baseline_same_as_last)
    demographics  : ethnicity file rows (needs seat plus the *_pct columns)
    current/reference : theme means + n_articles for the two windows
    Returns (DataFrame, status message). With too little data the shift column is 0.
    """
    ok, message = data_status(current, reference, min_headlines)
    change = theme_change(current, reference) if ok else {t: 0.0 for t in THEMES}
    lean = lean_change(current, reference) if ok else 0.0
    if ok and lean == 0.0 and coalition_lean(current) is None:
        message += '; coalition sentiment missing, lean channel off'
    demo = demographics.copy()
    demo['seat_no'] = demo['seat'].astype(str).str.split().str[0]
    demo = demo.set_index('seat_no')

    out = forecast.copy()
    shifts, adjusted = [], []
    for _, r in out.iterrows():
        p = float(r['probability'])
        if not ok or r['seat_no'] not in demo.index:
            shift = 0.0
        else:
            shift = seat_shift(segment_shares(demo.loc[r['seat_no']]), change,
                               held_by_opposition=bool(r['baseline_same_as_last']), lean_delta=lean)
        shifts.append(round(shift, 4))
        adjusted.append(round(_sigmoid(_logit(p) + shift), 4) if shift else p)
    out['sentiment_shift_logit'] = shifts
    out['sentiment_probability'] = adjusted
    out['sentiment_flips_call'] = (out['sentiment_probability'] >= 0.5) != (out['probability'] >= 0.5)
    return out, message


def load_demographics(state: str) -> pd.DataFrame:
    from backend.core.states import STATES
    year = STATES[state]['ethnicity_year']
    return pd.read_csv(ROOT / 'data/raw/ethnicity' / f'ethnicity_{state}_{year}.csv')


def assumption_table() -> pd.DataFrame:
    """The reaction table as shown on the dashboard (positive = toward Harapan)."""
    rows = []
    for t, m in REACTIONS.items():
        rows.append({'theme': t, 'type': 'fixed direction', 'weight': RELIABILITY[t], **m})
    for t, m in ANTI_INCUMBENT.items():
        rows.append({'theme': t, 'type': 'against the seat holder', 'weight': RELIABILITY[t], **m})
    rows.append({'theme': 'coalition_lean (Harapan minus BN/PN sentiment)', 'type': 'same direction, size by group',
                 'weight': LEAN_RELIABILITY, **LEAN_GROUP_WEIGHT})
    rows.append({'theme': 'govt_performance', 'type': 'not used', 'weight': 0.0,
                 **{s: 0.0 for s in SEGMENTS}})
    return pd.DataFrame(rows)


WINDOWS_FILE = ROOT / 'data/processed/narrative_windows.csv'


def _rows_for(df: pd.DataFrame, state=None) -> pd.DataFrame:
    """Window rows for one state. A row with an empty 'state' column is a national, current window
    (used for the states that have not voted yet). A row with a state name is news collected
    BEFORE that state's election, used to test the layer against the real result."""
    if 'state' not in df.columns:
        df = df.assign(state='')
    key = df['state'].fillna('').astype(str).str.strip()
    return df[key == (state or '')].sort_values('date')


def load_windows(path=None, state=None):
    """(current, reference) theme dicts from the two newest rows of narrative_windows.csv for this
    state (state=None: the national windows for upcoming elections). None if fewer than two exist.
    Columns: date, label, [state], the five themes, the coalition sentiments, n_articles."""
    path = Path(path) if path else WINDOWS_FILE
    if not path.exists():
        return None
    df = _rows_for(pd.read_csv(path), state)
    if len(df) < 2:
        return None
    return df.iloc[-1].to_dict(), df.iloc[-2].to_dict()


def update_summary(path=None, state=None):
    """One line saying which news the sentiment layer last used, or None if no window exists."""
    path = Path(path) if path else WINDOWS_FILE
    if not path.exists():
        return None
    df = _rows_for(pd.read_csv(path), state)
    if df.empty:
        return None
    r = df.iloc[-1]
    start = f"{r['start']} to " if 'start' in df.columns and pd.notna(r.get('start')) else 'up to '
    return f"{r.get('label', r['date'])}: news {start}{r['date']}, {int(r['n_articles'])} headlines scored"


STALE_AFTER_DAYS = 14


def freshness(current: dict, reference: dict, today=None) -> dict:
    """How old the news behind the sentiment view is. Returns
    {'updated': 'YYYY-MM-DD', 'days_old': int, 'stale': bool, 'n_current': int, 'n_reference': int,
     'text': one line for the page}. days_old counts from the END date of the current window."""
    import datetime as _dt
    updated = pd.Timestamp(current['date']).date()
    today = pd.Timestamp(today).date() if today is not None else _dt.date.today()
    days = max((today - updated).days, 0)
    n_cur, n_ref = int(current['n_articles']), int(reference['n_articles'])
    stale = days > STALE_AFTER_DAYS
    first = pd.Timestamp(reference['start']).date() if pd.notna(reference.get('start')) else None
    span = f"{first:%d %b %Y} to {updated:%d %b %Y}" if first else f"up to {updated:%d %b %Y}"
    text = f"News scored: {span} ({days} day{'s' if days != 1 else ''} since the latest headline); " \
           f"{n_cur} headlines in the current window, {n_ref} in the one before."
    if stale:
        text += f" This is more than {STALE_AFTER_DAYS} days old, so treat it as out of date."
    return {'updated': str(updated), 'days_old': days, 'stale': stale,
            'n_current': n_cur, 'n_reference': n_ref, 'text': text}


def adjust_validation(val: pd.DataFrame, demographics: pd.DataFrame, current: dict, reference: dict,
                      min_headlines: int = MIN_HEADLINES):
    """Sentiment-informed probabilities for a state that has ALREADY voted (validation_2026.csv),
    so they can be compared with the real result. Use windows of news published BEFORE polling day.
    Adds sentiment_probability, sentiment_predicted, sentiment_correct. Returns (DataFrame, message)."""
    from backend.core.features import OPPOSITION
    f = val.copy()
    f['seat_no'] = f['seat'].astype(str).str.split().str[0]
    f['baseline_same_as_last'] = f['winner_coalition_a'].isin(OPPOSITION).astype(int)
    out, message = adjust_forecast(f, demographics, current, reference, min_headlines)
    out['sentiment_predicted'] = (out['sentiment_probability'] >= 0.5).astype(int)
    out['sentiment_correct'] = out['sentiment_predicted'] == out['actual_non_bn']
    return out, message