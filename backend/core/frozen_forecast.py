"""
backend/core/frozen_forecast.py
===============================
Read the newest FROZEN forecast for a state (written by scripts/freeze_forecasts.py).

The dashboard, the map builder and the tests all use this one function, so the numbers
people see for Melaka, Selangor and Perak are the same numbers that were saved before
the elections, not the output of an older script.

    df, info = latest_frozen('selangor')
    # df: one row per seat (seat_no, seat_name, probability, ...); info: dates and commit
    # (None, None) if the state has no frozen forecast
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FORECAST_DIR = ROOT / 'forecasts'


def frozen_dates(state: str, base: Path = None) -> list:
    """Dates (YYYY-MM-DD) of the frozen folders for a state, oldest first."""
    base = Path(base) if base else FORECAST_DIR
    folder = base / state
    if not folder.exists():
        return []
    return sorted(p.name for p in folder.iterdir() if (p / 'forecast.csv').exists() and (p / 'manifest.json').exists())


def latest_frozen(state: str, base: Path = None):
    """(forecast DataFrame, info dict) for the newest frozen folder, or (None, None)."""
    dates = frozen_dates(state, base)
    if not dates:
        return None, None
    base = Path(base) if base else FORECAST_DIR
    folder = base / state / dates[-1]
    df = pd.read_csv(folder / 'forecast.csv')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    info = {
        'state': state,
        'folder_date': dates[-1],
        'frozen_at_utc': manifest.get('frozen_at_utc'),
        'code_commit': (manifest.get('code_commit') or '')[:7],
        'forecasting_election_after': manifest.get('forecasting_election_after'),
        'n_seats': manifest.get('n_seats'),
    }
    return df, info


def last_side(df: pd.DataFrame) -> pd.Series:
    """'Harapan' or 'BN/PN' for who held each seat at the last election (from the frozen baseline)."""
    return df['baseline_same_as_last'].map({1: 'Harapan', 0: 'BN/PN'})


# ── Side changes: who gained or lost seats ───────────────────────────────────────
# Two sides only: 'Harapan' and 'BN/PN'. BN and PN are summed into ONE group, so an election with three
# coalitions (BN vs PN vs PH) still compares as Harapan against the rest. This is the same definition
# as the model's target, so every page that shows "gains and losses" uses this one function.

def seat_changes(last, now, df: pd.DataFrame) -> dict:
    """Compare two Series of 'Harapan' / 'BN/PN' (same index as df): who held each seat before, and who
    holds (or is predicted to hold) it now. Returns counts, deltas and the seats that change side:
      to_harapan : seats BN/PN -> Harapan  (Harapan gains)
      to_bnpn    : seats Harapan -> BN/PN  (BN/PN gains)"""
    n = len(df)
    last_h, now_h = int((last == 'Harapan').sum()), int((now == 'Harapan').sum())
    return {
        'total': n,
        'last_harapan': last_h, 'last_bnpn': n - last_h,
        'now_harapan': now_h, 'now_bnpn': n - now_h,
        'delta_harapan': now_h - last_h, 'delta_bnpn': (n - now_h) - (n - last_h),
        'to_harapan': df[(last == 'BN/PN') & (now == 'Harapan')],
        'to_bnpn': df[(last == 'Harapan') & (now == 'BN/PN')],
    }


def side_changes(df: pd.DataFrame, prob_col: str = 'probability') -> dict:
    """Forecast view: who held each seat at the last election vs the side the forecast gives it (P >= 0.5 = Harapan)."""
    now = df[prob_col].map(lambda p: 'Harapan' if p >= 0.5 else 'BN/PN')
    return seat_changes(last_side(df), now, df)


def result_changes(df: pd.DataFrame) -> dict:
    """Validated view (rows of backend/models/<state>/validation_2026.csv): the previous election's winner
    (winner_coalition_a) vs the REAL 2026 result (actual_non_bn). PH = Harapan; BN, PN and independents = BN/PN."""
    last = df['winner_coalition_a'].map(lambda c: 'Harapan' if c == 'PH' else 'BN/PN')
    now = df['actual_non_bn'].map({1: 'Harapan', 0: 'BN/PN'})
    return seat_changes(last, now, df)