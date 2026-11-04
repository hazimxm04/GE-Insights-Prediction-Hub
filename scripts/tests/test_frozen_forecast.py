"""The dashboard reads frozen forecasts through backend/core/frozen_forecast.py."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from backend.core import frozen_forecast as fz


def make(base, state, date, probs):
    folder = base / state / date
    folder.mkdir(parents=True)
    pd.DataFrame({'seat_no': [f'N.0{i}' for i in range(len(probs))],
                  'seat_name': [f'N.0{i} X' for i in range(len(probs))],
                  'probability': probs,
                  'baseline_same_as_last': [1, 0, 0][:len(probs)]}).to_csv(folder / 'forecast.csv', index=False)
    (folder / 'manifest.json').write_text(json.dumps(
        {'frozen_at_utc': f'{date}T00:00:00Z', 'code_commit': 'abcdef1234', 'forecasting_election_after': 2023,
         'n_seats': len(probs)}), encoding='utf-8')


def test_newest_folder_is_used(tmp_path):
    make(tmp_path, 'selangor', '2026-10-30', [0.1, 0.2, 0.3])
    make(tmp_path, 'selangor', '2026-12-01', [0.9, 0.8, 0.7])
    df, info = fz.latest_frozen('selangor', tmp_path)
    assert info['folder_date'] == '2026-12-01' and info['code_commit'] == 'abcdef1'
    assert df['probability'].tolist() == [0.9, 0.8, 0.7]


def test_state_without_a_freeze_returns_none(tmp_path):
    assert fz.latest_frozen('johor', tmp_path) == (None, None)


def test_last_side_maps_baseline(tmp_path):
    make(tmp_path, 'perak', '2026-10-30', [0.1, 0.2, 0.3])
    df, _ = fz.latest_frozen('perak', tmp_path)
    assert fz.last_side(df).tolist() == ['Harapan', 'BN/PN', 'BN/PN']


def test_real_frozen_forecasts_load():
    for state in ('selangor', 'melaka', 'perak'):
        df, info = fz.latest_frozen(state)
        assert df is not None and len(df) == info['n_seats'] > 0
        assert df['probability'].between(0, 1).all()
