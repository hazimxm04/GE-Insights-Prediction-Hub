"""
Scoring a frozen forecast: counts must match a hand-made example, and a frozen folder
that was edited, or an election that has not happened yet, must be refused.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import freeze_forecasts as ff
import score_forecast as sf


def make_frozen(tmp_path, year_a=2023):
    """A tiny hand-made frozen forecast: 4 seats."""
    probs = [0.9, 0.2, 0.7, 0.1]
    df = pd.DataFrame({
        'seat_no': ['N.01', 'N.02', 'N.03', 'N.04'],
        'seat_name': ['N.01 A', 'N.02 B', 'N.03 C', 'N.04 D'],
        'predicted_opposition': [int(p >= 0.5) for p in probs],
        'prediction': ['non-BN' if p >= 0.5 else 'BN' for p in probs],
        'probability': probs,
        'rf_prob': probs, 'xgb_prob': probs, 'calibrated_prob': probs,
        'confidence': 'HIGH', 'is_ood': False, 'ood_score': 0.5,
        'model_used': 'calibrated_rf', 'base_rate': 0.5,
        'baseline_never_opposition': 0,
        'baseline_same_as_last': [1, 0, 0, 0],
    })[ff.FORECAST_COLUMNS]
    folder = tmp_path / 'selangor' / '2026-10-30'
    folder.mkdir(parents=True)
    df.to_csv(folder / 'forecast.csv', index=False, lineterminator='\n')
    manifest = {'state': 'selangor', 'frozen_at_utc': '2026-10-30T00:00:00Z',
                'forecasting_election_after': year_a, 'n_seats': 4,
                'forecast_csv_sha256': ff.sha256_file(folder / 'forecast.csv', text=True)}
    (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return folder


def make_ballots(year_a=2023, year_b=2028, with_result=True):
    rows = [{'date': f'{year_a}-08-12', 'seat': f'N.0{i} X', 'coalition': 'BN', 'result': 'won'} for i in range(1, 5)]
    if with_result:
        winners = {1: 'PH', 2: 'BN', 3: 'PERIKATAN', 4: 'PH'}     # opposition won seats 1 and 4
        rows += [{'date': f'{year_b}-05-01', 'seat': f'N.0{i} Renamed', 'coalition': c, 'result': 'won'}
                 for i, c in winners.items()]
    return pd.DataFrame(rows)


def test_counts_match_hand_calculation(tmp_path):
    r = sf.score_frozen(make_frozen(tmp_path), make_ballots())
    assert r['result_year'] == 2028
    assert r['seats_scored'] == 4 and r['opposition_won'] == 2
    assert r['model_correct'] == 2            # right on seats 1 and 2
    assert r['never_opposition_correct'] == 2  # right on seats 2 and 3
    assert r['same_as_last_correct'] == 3      # right on seats 1, 2 and 3
    assert r['false_alarms'] == 1 and r['missed_wins'] == 1
    assert r['brier'] == pytest.approx(0.3375, abs=1e-4)
    assert r['auc'] == pytest.approx(0.5)
    assert r['forecast_seats_without_result'] == [] and r['result_seats_without_forecast'] == []


def test_seats_are_matched_on_number_not_name(tmp_path):
    # the result ballots use different seat names ("Renamed") but the same numbers
    r = sf.score_frozen(make_frozen(tmp_path), make_ballots())
    assert r['seats_scored'] == 4


def test_no_result_yet_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        sf.score_frozen(make_frozen(tmp_path), make_ballots(with_result=False))


def test_edited_frozen_forecast_is_refused(tmp_path):
    folder = make_frozen(tmp_path)
    text = (folder / 'forecast.csv').read_text(encoding='utf-8')
    (folder / 'forecast.csv').write_text(text.replace('0.9', '0.4', 1), encoding='utf-8', newline='\n')
    with pytest.raises(SystemExit):
        sf.score_frozen(folder, make_ballots())


def test_unmatched_seats_are_reported(tmp_path):
    ballots = make_ballots()
    ballots = ballots[~ballots['seat'].str.startswith('N.04 Renamed')]      # no result for seat 4
    extra = pd.DataFrame([{'date': '2028-05-01', 'seat': 'N.05 New', 'coalition': 'BN', 'result': 'won'}])
    r = sf.score_frozen(make_frozen(tmp_path), pd.concat([ballots, extra], ignore_index=True))
    assert r['seats_scored'] == 3
    assert r['forecast_seats_without_result'] == ['N.04']
    assert r['result_seats_without_forecast'] == ['N.05']
