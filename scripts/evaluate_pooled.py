"""
Pooled model vs per-state models, with a time-ordered (no future information) test.

POOLED  = one model trained on the transitions of ALL states together.
FAIR-TEST RULE: to test an election with B-year Y, the pooled model trains only on
transitions (any state) whose own B-year is strictly earlier than Y. Nothing from
the test year or later is ever used, in any state.

Columns in the report (seats correct out of n):
  per-state    the model trained on that state's own earlier transitions only
               (same as evaluate_holdout.py)
  pooled       one model, all states, no state indicator
  pooled+state one model, all states, plus a state indicator column
  never-opp    baseline: always predict "opposition does not win"
  same-last    baseline: predict the seat goes the same way as last time

Run from the project root:
    python scripts/evaluate_pooled.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mlops'))
sys.path.insert(0, str(ROOT / 'scripts'))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss

import evaluate_holdout as eh
from backend.core.features import OPPOSITION
from backend.core.states import STATES
from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN

SHARED = TRAINING_CONFIGS[next(iter(TRAINING_CONFIGS))]   # model settings are shared by all states


def fit_predict(train_df, test_df, cols):
    X_tr, y_tr = train_df[cols].fillna(0), train_df['target_non_bn_won']
    X_te = test_df[cols].fillna(0)
    rf = RandomForestClassifier(**SHARED['rf_params'], class_weight='balanced', random_state=42).fit(X_tr, y_tr)
    xgb = XGBClassifier(**SHARED['xgb_params'], eval_metric='logloss', random_state=42).fit(X_tr, y_tr)
    return (rf.predict_proba(X_te)[:, 1] + xgb.predict_proba(X_te)[:, 1]) / 2


def build_all(states):
    """Every transition of every state (training transitions + the final real election)."""
    rows, skipped = [], []
    for st in states:
        try:
            ballots, stats = eh.load_state(st)
        except Exception as e:
            skipped.append((st, f'{type(e).__name__}'))
            continue
        steps = list(STATES[st]['transitions']) + ([STATES[st]['final_test']] if STATES[st]['final_test'] else [])
        for a, b in steps:
            df = eh.features(ballots, stats, st, a, b)
            df['state'] = st
            rows.append({'state': st, 'a': a, 'b': b, 'df': df})
    return rows, skipped


def add_state_indicator(df, states):
    out = df.copy()
    for s in states:
        out[f'state_{s}'] = (out['state'] == s).astype(int)
    return out


def main():
    states = sys.argv[1:] or list(STATES)
    trans, skipped = build_all(states)
    for st, why in skipped:
        print(f'(skipped {st}: no data - {why})')
    have = sorted({t['state'] for t in trans}, key=list(STATES).index)
    ind_cols = [f'state_{s}' for s in have]
    cols_plain = list(FEATURE_NAMES_TRAIN)
    cols_state = cols_plain + ind_cols

    results = []
    hdr = (f"{'state':<13}{'test':<12}{'n':>4}{'opp':>5}{'per-state':>11}{'pooled':>9}{'pooled+state':>14}"
           f"{'never-opp':>11}{'same-last':>11}{'train rows':>12}{'FP':>4}{'FN':>4}")
    print(hdr)
    print('-' * len(hdr))
    for st in have:
        own = [t for t in trans if t['state'] == st]
        for j in range(1, len(own)):
            test = own[j]
            df_te = test['df']
            y = df_te['target_non_bn_won']
            # per-state: this state's own earlier transitions
            tr_own = pd.concat([t['df'] for t in own[:j]], ignore_index=True)
            p_state = fit_predict(tr_own, df_te, cols_plain)
            # pooled: every state's transitions with a strictly earlier B-year
            pool = [t for t in trans if t['b'] < test['b']]
            tr_pool = pd.concat([t['df'] for t in pool], ignore_index=True)
            p_pool = fit_predict(tr_pool, df_te, cols_plain)
            tr_pool_s = add_state_indicator(tr_pool, have)
            te_s = add_state_indicator(df_te, have)
            p_pool_s = fit_predict(tr_pool_s, te_s, cols_state)

            pred_ps = (p_state >= .5).astype(int)
            pred_p = (p_pool >= .5).astype(int)
            pred_pss = (p_pool_s >= .5).astype(int)
            never = int((y == 0).sum())
            last = df_te['winner_coalition_a'].isin(OPPOSITION).astype(int)
            same = int((last == y).sum())
            fp = int(((pred_pss == 1) & (y.values == 0)).sum())
            fn = int(((pred_pss == 0) & (y.values == 1)).sum())
            r = {
                'state': st, 'test': f"{test['a']}->{test['b']}", 'n': int(len(y)), 'opposition_wins': int(y.sum()),
                'per_state': int((pred_ps == y.values).sum()), 'pooled': int((pred_p == y.values).sum()),
                'pooled_state': int((pred_pss == y.values).sum()), 'never_opposition': never, 'same_as_last': same,
                'pooled_train_rows': int(len(tr_pool)), 'pooled_state_fp': fp, 'pooled_state_fn': fn,
                'is_latest': j == len(own) - 1,
            }
            results.append(r)
            star = ' *' if r['is_latest'] else ''
            print(f"{st:<13}{r['test']:<12}{r['n']:>4}{r['opposition_wins']:>5}{r['per_state']:>11}{r['pooled']:>9}"
                  f"{r['pooled_state']:>14}{never:>11}{same:>11}{r['pooled_train_rows']:>12}{fp:>4}{fn:>4}{star}")

    def total(rows, key):
        return sum(r[key] for r in rows)

    for label, rows in (('LATEST test per state (*)', [r for r in results if r['is_latest']]),
                        ('ALL tests above', results)):
        n = total(rows, 'n')
        print(f"\n{label}: {len(rows)} tests, {n} seats")
        for key, name in (('per_state', 'per-state'), ('pooled', 'pooled'), ('pooled_state', 'pooled+state'),
                          ('never_opposition', 'never-opp'), ('same_as_last', 'same-as-last')):
            print(f"  {name:<14}{total(rows, key):>5}/{n}  ({total(rows, key) / n:.1%})")

    print("\n* = the latest test for each state. Test rule: train only on transitions (any state) with an earlier B-year.")
    out = ROOT / 'backend' / 'models' / 'pooled_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()