"""Tuning must be time-ordered: train on earlier elections, test on a later one,
and never touch the real final-test election."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from tune_hyperparams import make_folds


def _frame(year_b, n=3):
    return pd.DataFrame({'year_b': [year_b] * n, 'seat_no': [f'N.0{i}' for i in range(n)]})


def test_every_fold_trains_on_earlier_elections_only():
    frames = {'x': [((2008, 2013), _frame(2013)), ((2013, 2018), _frame(2018)), ((2018, 2022), _frame(2022))]}
    folds = make_folds(frames)
    assert [f[1] for f in folds] == [(2013, 2018), (2018, 2022)]
    for _, step, train, test in folds:
        assert train['year_b'].max() < test['year_b'].min()
        assert (test['year_b'] == step[1]).all()


def test_states_do_not_mix_and_single_transition_states_give_no_fold():
    frames = {'a': [((2008, 2013), _frame(2013)), ((2013, 2018), _frame(2018))],
              'b': [((2013, 2018), _frame(2018))]}
    folds = make_folds(frames)
    assert [f[0] for f in folds] == ['a']


def test_the_real_2026_elections_are_not_in_the_tuning_data():
    from backend.core.states import STATES
    for state, info in STATES.items():
        assert all(b != 2026 for _, b in info['transitions']), state
