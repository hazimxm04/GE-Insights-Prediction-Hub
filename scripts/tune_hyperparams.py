"""
scripts/tune_hyperparams.py
============================
Compare model settings (Random Forest / XGBoost) with a TIME-ORDERED test.

Why time-ordered: a random k-fold would put seats from the same election in both
the training and the test part, and seats in one election swing together, so the
score would be too optimistic. Here every fold trains on earlier elections and
tests on a later one, exactly like scripts/evaluate_holdout.py:

    train on transitions 1..j-1  ->  test on transition j      (for every state)

The real 2026 elections (the 'final_test' of each state) are NEVER used here, so
the settings are not chosen by looking at the answers.

Score: Brier score over all test seats (lower is better; it rewards sensible
probabilities, not just the yes/no call). Accuracy is shown too.

Honest limit: there are only a few folds (about two per state) and 28-59 seats
each, so small differences are noise. Switch to other settings only if they beat
the current ones clearly (a Brier gap of 0.01 or more).

Usage:
    python scripts/tune_hyperparams.py                 # all states
    python scripts/tune_hyperparams.py johor melaka
Prints the best settings; copy them into scripts/training_config.py by hand,
then retrain with scripts/train_models.py.
"""
import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mlops'))
sys.path.insert(0, str(ROOT / 'scripts'))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import evaluate_holdout as eh
from backend.core.states import STATES
from training_config import RF_PARAMS, XGB_PARAMS, FEATURE_NAMES_TRAIN

RF_GRID = {'max_depth': [3, 4, 5, 6], 'min_samples_leaf': [3, 5, 8], 'n_estimators': [100, 150, 200]}
XGB_GRID = {'max_depth': [2, 3, 4], 'learning_rate': [0.01, 0.03, 0.05, 0.1], 'n_estimators': [100, 150, 200]}
NOISE = 0.01   # Brier differences smaller than this are treated as noise


def make_folds(frames_by_state):
    """frames_by_state: {state: [(step, dataframe), ...]} in time order.
    Returns time-ordered folds (state, test_step, train_df, test_df): train on every
    earlier transition of that state, test on the next one. Needs >= 2 transitions."""
    folds = []
    for state, frames in frames_by_state.items():
        for j in range(1, len(frames)):
            train = pd.concat([df for _, df in frames[:j]], ignore_index=True)
            folds.append((state, frames[j][0], train, frames[j][1]))
    return folds


def build_folds(states):
    frames = {}
    for st in states:
        try:
            ballots, stats = eh.load_state(st)
        except Exception as e:
            print(f'  (skipping {st}: no data - {type(e).__name__})')
            continue
        # only the training transitions: the real 2026 test is never used for tuning
        frames[st] = [((a, b), eh.features(ballots, stats, st, a, b)) for a, b in STATES[st]['transitions']]
    return make_folds(frames)


def make_model(kind, params):
    if kind == 'rf':
        return RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
    return XGBClassifier(**params, eval_metric='logloss', random_state=42)


def score_params(kind, params, folds):
    """Returns (brier, accuracy) over all test seats of all folds."""
    sq, correct, n = 0.0, 0, 0
    for _, _, tr, te in folds:
        model = make_model(kind, params).fit(tr[FEATURE_NAMES_TRAIN].fillna(0), tr['target_non_bn_won'])
        p = model.predict_proba(te[FEATURE_NAMES_TRAIN].fillna(0))[:, 1]
        y = te['target_non_bn_won'].values
        sq += float(((p - y) ** 2).sum())
        correct += int(((p >= 0.5) == y).sum())
        n += len(y)
    return sq / n, correct / n


def search(kind, grid, current, folds):
    keys = list(grid)
    rows = []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        brier, acc = score_params(kind, params, folds)
        rows.append((brier, acc, params))
    rows.sort(key=lambda r: (r[0], -r[1]))
    cur_brier, cur_acc = score_params(kind, current, folds)
    return rows, cur_brier, cur_acc


def report(name, rows, cur_brier, cur_acc, current):
    print(f"\n  [{name}]  top 5 of {len(rows)} settings (lower Brier is better)")
    for brier, acc, params in rows[:5]:
        print(f"    Brier {brier:.4f}  accuracy {acc:.1%}  {params}")
    print(f"    current   Brier {cur_brier:.4f}  accuracy {cur_acc:.1%}  {current}")
    gap = cur_brier - rows[0][0]
    if gap >= NOISE:
        print(f"    -> best beats current by {gap:.4f} Brier: worth considering.")
    else:
        print(f"    -> best beats current by only {gap:.4f} Brier (< {NOISE}): within noise, keep current settings.")


def main():
    states = sys.argv[1:] or list(STATES)
    print('Time-ordered tuning (train on earlier elections, test on a later one; the real 2026 tests are not used)')
    folds = build_folds(states)
    if not folds:
        print('No folds: need at least two training transitions with data.')
        return
    n_seats = sum(len(te) for _, _, _, te in folds)
    print(f"{len(folds)} folds, {n_seats} test seats in total")
    for state, step, tr, te in folds:
        print(f"  {state:<13} test {step[0]}->{step[1]}  train rows {len(tr)}  test seats {len(te)}")

    rows, cb, ca = search('rf', RF_GRID, RF_PARAMS, folds)
    report('Random Forest', rows, cb, ca, RF_PARAMS)
    rows, cb, ca = search('xgb', XGB_GRID, XGB_PARAMS, folds)
    report('XGBoost', rows, cb, ca, XGB_PARAMS)

    print("\nIf you change settings: edit RF_PARAMS / XGB_PARAMS in scripts/training_config.py, "
          "then run scripts/train_models.py.")


if __name__ == '__main__':
    main()