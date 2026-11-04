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
