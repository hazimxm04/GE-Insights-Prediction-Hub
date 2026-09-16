import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd
from train_models import build_transition, train_state
from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN


# ── Regression tests: bugs actually found during development ──────

def test_no_temporal_leakage():
    """
    Sentiment/economic data (only available as of Aug 2026) must
    never appear in training features -- it would leak future
    information into historical training rows (2008-2023).
    """
    leaky_features = {'bn_sentiment', 'harapan_sentiment', 'pn_sentiment',
                       'economic_pressure', 'racial_tension_index'}
    assert not leaky_features & set(FEATURE_NAMES_TRAIN), \
        "Sentiment/economic features leaked into training set!"


def test_coalition_standardization_applied_to_both_columns():
    """
    Regression test for a real bug: winner_coalition_a was not being
    standardized the same way as winner_coalition_b. Any feature
    derived from _a (e.g. incumbent-history features) would silently
    compute wrong values for raw labels like 'Harapan' instead of 'PH'.
    """
    fake_ballots = pd.DataFrame({
        'seat': ['A', 'A'],
        'date': ['2013-01-01', '2018-01-01'],
        'coalition': ['Harapan', 'BN'],
        'result': ['won', 'won'],
    })
    fake_stats = pd.DataFrame({
        'seat': ['A', 'A'],
        'date': ['2013-01-01', '2018-01-01'],
        'majority': [1000, 2000],
        'votes_valid': [10000, 11000],
        'voters_total': [15000, 16000],
        'n_candidates': [3, 3],
    })
    df = build_transition(fake_ballots, fake_stats, 2013, 2018)
    assert df['winner_coalition_a'].iloc[0] == 'PH', \
        f"winner_coalition_a not standardized: got {df['winner_coalition_a'].iloc[0]!r}"


def test_target_non_bn_won_is_binary():
    """Confirms target stays strictly 0/1."""
    fake_ballots = pd.DataFrame({
        'seat': ['A', 'A'], 'date': ['2013-01-01', '2018-01-01'],
        'coalition': ['BN', 'PH'], 'result': ['won', 'won'],
    })
    fake_stats = pd.DataFrame({
        'seat': ['A', 'A'], 'date': ['2013-01-01', '2018-01-01'],
        'majority': [1000, 2000], 'votes_valid': [10000, 11000],
        'voters_total': [15000, 16000], 'n_candidates': [3, 3],
    })
    df = build_transition(fake_ballots, fake_stats, 2013, 2018)
    assert set(df['target_non_bn_won'].unique()) <= {0, 1}


def test_all_states_have_required_config_keys():
    """
    Regression test for a real bug: a state config was missing
    'ethnicity_year', causing a KeyError deep inside train_state()
    instead of a clear config-validation error.
    """
    required_keys = {'transitions', 'rf_params', 'xgb_params',
                      'description', 'ethnicity_year'}
    for state, config in TRAINING_CONFIGS.items():
        missing = required_keys - set(config.keys())
        assert not missing, f"{state} config missing keys: {missing}"


def test_feature_names_train_has_no_duplicates():
    assert len(FEATURE_NAMES_TRAIN) == len(set(FEATURE_NAMES_TRAIN))


# ── Smoke test: does the full pipeline actually run? ───────────────

def test_train_state_end_to_end_johor():
    """
    Integration smoke test: runs the full train_state() pipeline for
    Johor (fetches real data, builds transitions, trains RF+XGB,
    calibrates, runs OOD detection, saves models) and checks it
    completes without error and produces a sane result.

    This is slower than the other tests (hits network + trains real
    models) but catches an entire CLASS of bugs the unit tests above
    can't: NameError, UnboundLocalError, KeyError, and any other
    exception that only shows up when the full path actually runs.
    """
    result = train_state('johor', TRAINING_CONFIGS['johor'])

    assert result is not None, "train_state returned None"
    assert 'models' in result
    assert 'ensemble' in result['models']

    accuracy = result['models']['ensemble']['accuracy']
    assert accuracy > 0.5, \
        f"Ensemble accuracy ({accuracy:.2%}) below sanity floor -- something is broken"

    assert result['seats'] > 0, "No training rows produced"


# ── Runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_no_temporal_leakage,
        test_coalition_standardization_applied_to_both_columns,
        test_target_non_bn_won_is_binary,
        test_all_states_have_required_config_keys,
        test_feature_names_train_has_no_duplicates,
        test_train_state_end_to_end_johor,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test_fn.__name__} -- {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test_fn.__name__} -- {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)