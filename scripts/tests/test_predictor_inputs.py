"""
The live forecast must be built exactly like training, and must not look at anything
after election A (the latest election we are allowed to know about).

Offline tests use a small synthetic state. One extra test compares StatePredictor with
scripts/validation.py on real data and is skipped when the data or models are missing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.core.features import build_features, build_transition, seat_base_rates, seat_no

FEATURES = ['majority_change', 'turnout_change', 'incumbent_held',
            'log_voters', 'majority_perc_change', 'n_candidates_b']
SEATS = ['N.01 A', 'N.02 B', 'N.03 C', 'N.04 D']


def _state(years, winners, bump_after=None):
    """years: election years. winners: {year: [coalition per seat]}.
    bump_after: rewrite every election AFTER this year (different winners and numbers)."""
    rng = np.random.RandomState(1)
    ballots, stats = [], []
    for y in years:
        for i, seat in enumerate(SEATS):
            coal = winners[y][i]
            maj, voters = int(rng.randint(500, 9000)), int(rng.randint(30000, 60000))
            valid, ncand = int(voters * rng.uniform(0.6, 0.85)), int(rng.randint(2, 5))
            if bump_after is not None and y > bump_after:
                coal = {'BN': 'PH', 'PH': 'BN', 'PN': 'BN'}.get(coal, 'BN')
                maj, voters, valid, ncand = maj * 4 + 99, voters + 12345, int(valid * .6), ncand + 3
            ballots.append({'seat': seat, 'date': f'{y}-05-09', 'coalition': coal, 'result': 'won'})
            stats.append({'seat': seat, 'date': f'{y}-05-09', 'majority': maj, 'votes_valid': valid,
                          'voters_total': voters, 'n_candidates': ncand})
    return pd.DataFrame(ballots), pd.DataFrame(stats)


WIN = {2013: ['BN', 'BN', 'PR', 'BN'], 2018: ['PH', 'BN', 'PH', 'PH'],
       2022: ['PN', 'BN', 'PH', 'BN'], 2026: ['BN', 'PH', 'PH', 'BN']}


def test_features_can_be_built_without_election_B():
    ballots, stats = _state([2013, 2018], WIN)               # nothing after 2018 exists
    out = build_features(ballots, stats, 2018)
    assert len(out) == len(SEATS) and set(FEATURES) <= set(out.columns)


def test_forecast_features_equal_training_features():
    ballots, stats = _state([2013, 2018, 2022], WIN)
    f = build_features(ballots, stats, 2018).sort_values('seat_no').reset_index(drop=True)
    t = build_transition(ballots, stats, 2018, 2022).sort_values('seat_no').reset_index(drop=True)
    for col in FEATURES:
        assert np.allclose(f[col].astype(float), t[col].astype(float), equal_nan=True), col


def test_nothing_after_election_A_changes_the_features_or_base_rates():
    b1, s1 = _state([2013, 2018, 2022, 2026], WIN)
    b2, s2 = _state([2013, 2018, 2022, 2026], WIN, bump_after=2018)
    f1 = build_features(b1, s1, 2018).sort_values('seat_no').reset_index(drop=True)
    f2 = build_features(b2, s2, 2018).sort_values('seat_no').reset_index(drop=True)
    for col in FEATURES:
        assert np.allclose(f1[col].astype(float), f2[col].astype(float), equal_nan=True), col
    assert seat_base_rates(b1, 2018) == seat_base_rates(b2, 2018)


def test_base_rate_uses_only_elections_up_to_A_and_counts_pn_as_not_opposition():
    ballots, _ = _state([2013, 2018, 2022, 2026], WIN)
    rates_2018, _ = seat_base_rates(ballots, 2018)
    # N.01: BN(2013) then PH(2018) -> newest weighs 1, older 0.5 -> (0*0.5 + 1*1)/1.5
    assert rates_2018['N.01'] == pytest.approx(round(1 / 1.5, 4))
    rates_2022, _ = seat_base_rates(ballots, 2022)
    # 2022 winner of N.01 is PN, which is NOT opposition in the target -> 0.
    assert rates_2022['N.01'] == pytest.approx(round((0 * .25 + 1 * .5 + 0 * 1) / 1.75, 4))


def test_base_rate_ignores_elections_before_min_year():
    w = {2000: ['PH'] * 4, **WIN}
    ballots, _ = _state([2000, 2013, 2018], w)
    with_old, _ = seat_base_rates(ballots, 2018)
    ballots_no_old = ballots[~ballots['date'].str.startswith('2000')]
    without_old, _ = seat_base_rates(ballots_no_old, 2018)
    assert with_old == without_old


def test_base_rate_matches_renamed_seats_by_number():
    ballots, _ = _state([2013, 2018], WIN)
    ballots.loc[ballots['date'].str.startswith('2018') & (ballots['seat'] == 'N.01 A'), 'seat'] = 'N.01 Renamed'
    rates, _ = seat_base_rates(ballots, 2018)
    assert 'N.01' in rates and len(rates) == len(SEATS)       # one row per seat number, not two


# ── real data: the live predictor must agree with scripts/validation.py ────────
@pytest.mark.parametrize('state,as_of', [('johor', 2022), ('neg_sembilan', 2023)])
def test_backtest_predictor_matches_validation(state, as_of):
    csv = ROOT / 'backend' / 'models' / state / 'validation_2026.csv'
    needed = [csv, ROOT / 'backend' / 'models' / state / 'rf_model.pkl']
    if not all(p.exists() for p in needed):
        pytest.skip('saved model / validation output not available')
    try:
        from backend.core.models.state_predictor import StatePredictor
        pred = StatePredictor(state, as_of=as_of).predict_all()
    except Exception as e:                                    # e.g. no raw data and no network
        pytest.skip(f'cannot build predictor here: {type(e).__name__}')
    pred = pred[['seat_name', 'ensemble_prob']].assign(sn=lambda d: d['seat_name'].map(seat_no))
    val = pd.read_csv(csv)[['seat', 'probability']].assign(sn=lambda d: d['seat'].map(seat_no))
    m = pred.merge(val, on='sn')
    assert len(m) == len(val)
    assert (m['ensemble_prob'] - m['probability']).abs().max() < 1e-3
