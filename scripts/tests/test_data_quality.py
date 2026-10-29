"""
Data-quality checks on the local election data (data/raw).
A state is skipped if its files are not downloaded yet
(run: python backend/scripts/download_data.py).

These catch the problems we actually hit: renamed seats, missing winners,
duplicate rows, a different number of seats between elections.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.core.features import MIN_YEAR, build_transition, seat_no
from backend.core.states import STATES
from state_data import _paths
from training_config import FEATURE_NAMES_TRAIN

STATS_COLS = {'date', 'seat', 'majority', 'votes_valid', 'voters_total', 'n_candidates'}
BALLOT_COLS = {'date', 'seat', 'coalition', 'result'}


def _load(state):
    b, s = _paths(state)
    if not (b.exists() and s.exists()):
        pytest.skip(f'{state}: local data not downloaded')
    return pd.read_parquet(b), pd.read_parquet(s)


@pytest.mark.parametrize('state', list(STATES))
def test_required_columns(state):
    ballots, stats = _load(state)
    assert STATS_COLS <= set(stats.columns), STATS_COLS - set(stats.columns)
    assert BALLOT_COLS <= set(ballots.columns), BALLOT_COLS - set(ballots.columns)


@pytest.mark.parametrize('state', list(STATES))
def test_no_duplicate_seats_within_an_election(state):
    _, stats = _load(state)
    stats = stats.assign(year=pd.to_datetime(stats['date']).dt.year, no=stats['seat'].map(seat_no))
    stats = stats[stats.year >= MIN_YEAR]
    dup = stats[stats.duplicated(['year', 'no'], keep=False)]
    assert dup.empty, f'duplicate seat numbers in {sorted(dup.year.unique())}'


@pytest.mark.parametrize('state', list(STATES))
def test_same_number_of_seats_in_every_election_since_min_year(state):
    _, stats = _load(state)
    counts = (stats.assign(year=pd.to_datetime(stats['date']).dt.year)
              .query('year >= @MIN_YEAR').groupby('year')['seat'].nunique())
    assert counts.nunique() == 1, f'seat count changes between elections: {counts.to_dict()}'


@pytest.mark.parametrize('state', list(STATES))
def test_registry_transitions_build_cleanly(state):
    """Every transition in the registry must be usable: most seats kept,
    no missing feature values."""
    ballots, stats = _load(state)
    n_seats = stats[pd.to_datetime(stats['date']).dt.year == STATES[state]['transitions'][0][1]]['seat'].nunique()
    steps = list(STATES[state]['transitions'])
    if STATES[state]['final_test']:
        steps.append(STATES[state]['final_test'])
    for a, b in steps:
        df = build_transition(ballots, stats, a, b)
        assert len(df) >= 0.9 * n_seats, f'{state} {a}->{b}: only {len(df)} of {n_seats} seats kept'
        feats = [c for c in FEATURE_NAMES_TRAIN if c in df.columns]
        assert not df[feats].isna().any().any(), f'{state} {a}->{b}: missing feature values'


def test_every_state_has_a_registry_entry_for_each_setting():
    for state, s in STATES.items():
        assert {'url_prefix', 'file_stem', 'ethnicity_year', 'transitions', 'final_test'} <= set(s), state
        for a, b in s['transitions']:
            assert a < b and a >= MIN_YEAR, f'{state}: bad transition {(a, b)}'

# ── ethnicity data: the file training uses must exist and line up with the seats ──
ETH_COLS = ['malay_pct', 'chinese_pct', 'indian_pct', 'young_malay_pct', 'young_chinese_pct',
            'older_malay_pct', 'youth_pct', 'median_age']


@pytest.mark.parametrize('state', list(STATES))
def test_ethnicity_year_is_a_registered_voter_roll(state):
    from backend.core.states import voter_roll_url
    info = STATES[state]
    assert info['ethnicity_year'] in info['voter_rolls']
    url = voter_roll_url(state, info['ethnicity_year'])
    assert url.startswith('https://lake.electiondata.my/voter_rolls/')
    assert f"{info['url_prefix']}_" in url and url.endswith(f"_{info['ethnicity_year']}.parquet")


@pytest.mark.parametrize('state', list(STATES))
def test_training_ethnicity_file_exists_and_matches_the_seats(state):
    """Without this file the models silently train on zeros for 8 features."""
    year = STATES[state]['ethnicity_year']
    path = ROOT / 'data' / 'raw' / 'ethnicity' / f'ethnicity_{state}_{year}.csv'
    assert path.exists(), f'missing {path.name} - run backend/scripts/download_ethnicity.py'
    eth = pd.read_csv(path)
    assert set(ETH_COLS) <= set(eth.columns)
    assert eth[ETH_COLS].notna().all().all()
    _, stats = _load(state)
    seats = stats[pd.to_datetime(stats['date']).dt.year == year]['seat'].map(seat_no)
    assert set(eth['seat'].map(seat_no)) == set(seats), 'ethnicity seats differ from the election seats'