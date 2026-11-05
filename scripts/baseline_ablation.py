"""
scripts/baseline_ablation.py
============================
Does the Random Forest + XGBoost ensemble earn its complexity?  What does each feature group add?

Same time-ordered splits as scripts/evaluate_holdout.py (train on earlier elections, test on the next
one), for every test election of every state, including the real 2026 results. Nothing here touches the
saved models or the frozen forecasts.

Variants compared (all use the same training data and the same test seats):
  current           RF + XGB ensemble, all 15 features (what the project uses)
  logistic_all      logistic regression (standardised, balanced classes), all 15 features
  logistic_2feat    logistic regression on just majority_perc_change and incumbent_held
  drop_ethnicity    current model without the 8 ethnicity / age features
  drop_turnout      current model without turnout_change
  drop_majority     current model without majority_change and majority_perc_change

Also reported for the current model: "ranking accuracy", the accuracy if the cut-off were set so the
number of seats called opposition equals the true number. It is an ORACLE (it uses the answer to pick
the cut-off), so it is not a forecast; it measures how well the model ORDERS seats.

    python scripts/baseline_ablation.py              # all states
    python scripts/baseline_ablation.py johor        # one state
    python scripts/baseline_ablation.py --save       # also writes docs/baseline_ablation.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'mlops'))

import evaluate_holdout as eh
from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN

ETH = list(eh.ETH_COLS) + ['tension_x_mixed']
VARIANTS = {
    'current':         ('ensemble', list(FEATURE_NAMES_TRAIN)),
    'logistic_all':    ('logistic', list(FEATURE_NAMES_TRAIN)),
    'logistic_2feat':  ('logistic', ['majority_perc_change', 'incumbent_held']),
    'drop_ethnicity':  ('ensemble', [c for c in FEATURE_NAMES_TRAIN if c not in ETH]),
    'drop_turnout':    ('ensemble', [c for c in FEATURE_NAMES_TRAIN if c != 'turnout_change']),
    'drop_majority':   ('ensemble', [c for c in FEATURE_NAMES_TRAIN if c not in ('majority_change', 'majority_perc_change')]),
}


def fit_predict(kind, cols, cfg, train, test):
    X_tr, y_tr = train[cols].fillna(0), train['target_non_bn_won']
    X_te = test[cols].fillna(0)
    if kind == 'logistic':
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000))
        return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    rf = RandomForestClassifier(**cfg['rf_params'], class_weight='balanced', random_state=42).fit(X_tr, y_tr)
    xgb = XGBClassifier(**cfg['xgb_params'], eval_metric='logloss', random_state=42).fit(X_tr, y_tr)
    return (rf.predict_proba(X_te)[:, 1] + xgb.predict_proba(X_te)[:, 1]) / 2


def metrics(y, p):
    y = np.asarray(y)
    return {'correct': int(((p >= 0.5).astype(int) == y).sum()),
            'auc': round(float(roc_auc_score(y, p)), 3) if len(set(y)) == 2 else None,
            'brier': round(float(brier_score_loss(y, p)), 3)}


def ranking_accuracy(y, p):
    """Accuracy if exactly as many seats were called opposition as there really were (oracle cut-off)."""
    y = np.asarray(y)
    k = int(y.sum())
    pred = np.zeros(len(y), dtype=int)
    pred[np.argsort(-np.asarray(p))[:k]] = 1
    return int((pred == y).sum())


def run_state(state):
    cfg = TRAINING_CONFIGS[state]
    ballots, stats = eh.load_state(state)
    steps = list(cfg['transitions']) + ([eh.FINAL_TEST[state]] if state in eh.FINAL_TEST else [])
    rows = []
    for j in range(1, len(steps)):
        train = pd.concat([eh.features(ballots, stats, state, a, b) for a, b in steps[:j]], ignore_index=True)
        a, b = steps[j]
        test = eh.features(ballots, stats, state, a, b)
        y = test['target_non_bn_won']
        row = {'state': state, 'test': f'{a}->{b}', 'is_real_2026': b == 2026, 'seats': int(len(y)),
               'opposition_wins': int(y.sum()), 'variants': {}}
        for name, (kind, cols) in VARIANTS.items():
            p = fit_predict(kind, cols, cfg, train, test)
            row['variants'][name] = metrics(y, p)
            if name == 'current':
                row['ranking_accuracy_oracle'] = ranking_accuracy(y, p)
        rows.append(row)
    return rows


def summarise(rows):
    names = list(VARIANTS)
    out = {}
    for label, sel in (('all_tests', rows), ('real_2026_only', [r for r in rows if r['is_real_2026']])):
        n_seats = sum(r['seats'] for r in sel)
        out[label] = {
            'tests': len(sel), 'seats': n_seats,
            'accuracy': {v: round(sum(r['variants'][v]['correct'] for r in sel) / n_seats, 3) for v in names},
            'mean_auc': {v: round(float(np.mean([r['variants'][v]['auc'] for r in sel])), 3) for v in names},
            'mean_brier': {v: round(float(np.mean([r['variants'][v]['brier'] for r in sel])), 3) for v in names},
            'oracle_ranking_accuracy': round(sum(r['ranking_accuracy_oracle'] for r in sel) / n_seats, 3),
        }
    return out


def print_report(rows, summary):
    names = list(VARIANTS)
    print(f"{'state':<13}{'test':<11}{'n':>4}{'opp':>5}  " + ''.join(f'{v:>16}' for v in names) + f"{'oracle-rank':>13}")
    for r in rows:
        star = ' *' if r['is_real_2026'] else ''
        cells = ''.join(f"{r['variants'][v]['correct']:>11}/{r['seats']:<4}" for v in names)
        print(f"{r['state']:<13}{r['test']:<11}{r['seats']:>4}{r['opposition_wins']:>5}  {cells}{r['ranking_accuracy_oracle']:>9}/{r['seats']}{star}")
    for label, s in summary.items():
        print(f"\n{label}: {s['tests']} test elections, {s['seats']} seats")
        print(f"  {'':<16}" + ''.join(f'{v:>16}' for v in names))
        print(f"  {'accuracy':<16}" + ''.join(f"{s['accuracy'][v]:>16.3f}" for v in names))
        print(f"  {'mean AUC':<16}" + ''.join(f"{s['mean_auc'][v]:>16.3f}" for v in names))
        print(f"  {'mean Brier':<16}" + ''.join(f"{s['mean_brier'][v]:>16.3f}" for v in names))
        print(f"  oracle ranking accuracy of the current model: {s['oracle_ranking_accuracy']:.3f}")
    print("\n* = real 2026 election.  Cells are seats correct at the 0.5 cut-off.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('states', nargs='*')
    ap.add_argument('--save', action='store_true')
    args = ap.parse_args()
    rows = []
    for s in args.states or list(TRAINING_CONFIGS):
        try:
            rows += run_state(s)
        except Exception as e:      # a state without data is skipped, not fatal
            print(f'{s}: skipped ({type(e).__name__}: {e})')
    summary = summarise(rows)
    print_report(rows, summary)
    if args.save:
        path = ROOT / 'docs' / 'baseline_ablation.json'
        path.write_text(json.dumps({'tests': rows, 'summary': summary}, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(f'saved {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
