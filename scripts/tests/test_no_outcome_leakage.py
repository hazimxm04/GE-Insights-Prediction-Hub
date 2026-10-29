"""
Outcome-leakage test.

RULE: when predicting election B, every feature may use only what was known
BEFORE B (election A and earlier). So if we change anything about election B --
who won, the majority, the turnout, the electorate, the number of candidates --
the feature values for that row must NOT change. Only the target may change.

This test builds a small synthetic state (no network needed), computes the
features, then rewrites election B's results and checks that the features are
identical. A feature that moves is reading the outcome year = leakage.

It is parametrised per feature so a failure names exactly which feature leaks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from train_models import build_transition
from training_config import FEATURE_NAMES_TRAIN

YEAR_PRE, YEAR_A, YEAR_B = 2013, 2018, 2022
SEATS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']


def _synthetic_state(perturb_b: bool):
    """Three elections (2013, 2018, 2022) for six seats.
    With perturb_b=True, election B (2022) is rewritten: different winners,
    majorities, turnout, electorate and number of candidates."""
    ballots, stats = [], []
    rng = np.random.RandomState(0)
    winners = {
        YEAR_PRE: ['BN', 'BN', 'BN', 'PR', 'BN', 'PR'],
        YEAR_A:   ['PH', 'BN', 'PH', 'PH', 'BN', 'PH'],
        YEAR_B:   ['BN', 'BN', 'PH', 'PH', 'PH', 'BN'],
    }
    flipped = {'BN': 'PH', 'PH': 'BN', 'PR': 'BN'}
    for year in (YEAR_PRE, YEAR_A, YEAR_B):
        for i, seat in enumerate(SEATS):
            coal = winners[year][i]
            majority = int(rng.randint(500, 9000))
            voters = int(rng.randint(30000, 60000))
            valid = int(voters * rng.uniform(0.6, 0.85))
            n_cand = int(rng.randint(2, 5))
            if perturb_b and year == YEAR_B:
                coal = flipped[coal]
                majority = majority * 5 + 777
                voters = int(voters * 1.35) + 1234
                valid = int(valid * 0.7) + 321
                n_cand = n_cand + 4
            date = f'{year}-05-09'
            ballots.append({'seat': seat, 'date': date, 'coalition': coal, 'result': 'won'})
            stats.append({'seat': seat, 'date': date, 'majority': majority,
                          'votes_valid': valid, 'voters_total': voters,
                          'n_candidates': n_cand})
    return pd.DataFrame(ballots), pd.DataFrame(stats)


def _features(perturb_b: bool):
    ballots, stats = _synthetic_state(perturb_b)
    df = build_transition(ballots, stats, YEAR_A, YEAR_B)
    return df.sort_values('seat').reset_index(drop=True)


BASE = None
PERTURBED = None


def _frames():
    global BASE, PERTURBED
    if BASE is None:
        BASE, PERTURBED = _features(False), _features(True)
    return BASE, PERTURBED


def test_perturbation_actually_changes_the_target():
    """Sanity check: the rewrite of election B really changes the answer."""
    base, pert = _frames()
    assert not base['target_non_bn_won'].equals(pert['target_non_bn_won'])


# Only the features that build_transition itself produces (ethnicity and the
# constant-0 sentiment placeholders are added elsewhere).
def _computed_features():
    base, _ = _frames()
    return [f for f in FEATURE_NAMES_TRAIN if f in base.columns]


@pytest.mark.parametrize('feature', ['majority_change', 'turnout_change', 'incumbent_held',
                                     'log_voters', 'majority_perc_change', 'n_candidates_b'])
def test_feature_does_not_depend_on_election_B(feature):
    base, pert = _frames()
    assert feature in base.columns, f'{feature} is no longer produced by build_transition'
    a = base[feature].astype(float).values
    b = pert[feature].astype(float).values
    same = np.allclose(a, b, equal_nan=True)
    assert same, (
        f"'{feature}' changed when only election {YEAR_B} (the election being predicted) "
        f"was rewritten -> it reads outcome-year data (leakage). "
        f"Rebuild it from {YEAR_A} and earlier only."
    )


# ── The features must still carry real information from A and earlier ──────────
def test_features_do_respond_to_election_A():
    """Guards against 'passing' by making every feature constant: rewriting
    election A (the last known result) MUST change the features."""
    ballots, stats = _synthetic_state(False)
    base = build_transition(ballots, stats, YEAR_A, YEAR_B).sort_values('seat').reset_index(drop=True)
    stats2 = stats.copy()
    m = pd.to_datetime(stats2['date']).dt.year == YEAR_A
    stats2.loc[m, 'majority'] = stats2.loc[m, 'majority'] * 3 + 100
    changed = build_transition(ballots, stats2, YEAR_A, YEAR_B).sort_values('seat').reset_index(drop=True)
    assert not np.allclose(base['majority_change'], changed['majority_change'])


def test_elections_before_min_year_are_refused():
    """The change features need an election in MIN_YEAR (2004) or later before A."""
    ballots, stats = _synthetic_state(False)
    with pytest.raises(ValueError):
        build_transition(ballots, stats, YEAR_PRE, YEAR_A)   # nothing before 2013 in the data


def test_seats_are_matched_on_number_not_name():
    """A seat renamed between elections (same N.xx number) must keep its row."""
    ballots, stats = _synthetic_state(False)
    for df in (ballots, stats):
        df['seat'] = df['seat'].str.replace('S', 'N.0', regex=False)
        later = pd.to_datetime(df['date']).dt.year >= YEAR_B
        df.loc[later & (df['seat'] == 'N.01'), 'seat'] = 'N.01 Renamed Seat'
        df.loc[~later, 'seat'] = df.loc[~later, 'seat'].replace({'N.01': 'N.01 Old Name'})
    out = build_transition(ballots, stats, YEAR_A, YEAR_B)
    assert len(out) == len(SEATS), 'renamed seat was dropped'
