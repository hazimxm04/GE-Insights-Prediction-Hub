"""
Honest held-out evaluation (time-ordered), compared with naive baselines.

For each state we only ever test on an election that comes AFTER everything the
model was trained on:

    train on transitions 1..j-1   ->   test on transition j

The last test for Johor and Negeri Sembilan is the real 2026 result, with the
model trained on the config transitions only (2026 is never in training and the
features use only election A and the election before it).

Baselines (the model has to beat these to be worth anything):
  never_opposition : always predict "Pakatan/opposition does not win"
  same_as_last     : predict that each seat is won by the same side as last time

Run from the project root:
    python scripts/evaluate_holdout.py                 # all five states
    python scripts/evaluate_holdout.py johor melaka
Uses data/raw/*.parquet when present, otherwise the online data source.
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
from sklearn.metrics import brier_score_loss, roc_auc_score

from train_models import build_transition, seat_no, OPPOSITION
from backend.core.states import STATES
from state_data import load_state_data
from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN

# The latest real election per state (ground truth), from the shared registry.
FINAL_TEST = {k: v['final_test'] for k, v in STATES.items() if v['final_test']}

ETH_COLS = ['malay_pct', 'chinese_pct', 'indian_pct', 'young_malay_pct',
            'young_chinese_pct', 'older_malay_pct', 'youth_pct', 'median_age']
PLACEHOLDERS = ['bn_sent_x_malay', 'harapan_sent_x_chinese', 'pn_sent_x_young_malay',
                'tension_x_mixed', 'economic_x_youth', 'narrative_pressure']


load_state = load_state_data   # one shared loader (local copy first)


def features(ballots, stats, state, a, b, build=build_transition):
    df = build(ballots, stats, a, b)
    eth_path = ROOT / 'data/raw/ethnicity' / f'ethnicity_{state}_{TRAINING_CONFIGS[state]["ethnicity_year"]}.csv'
    if eth_path.exists():
        eth = pd.read_csv(eth_path)
        eth['seat_no'] = eth['seat'].map(seat_no)
        df = df.merge(eth[['seat_no'] + ETH_COLS], on='seat_no', how='left')
    for c in ETH_COLS + PLACEHOLDERS:
        if c not in df.columns:
            df[c] = 0
    for c in PLACEHOLDERS:
        df[c] = 0
    return df


def fit_predict(cfg, train_df, test_df):
    X_tr = train_df[FEATURE_NAMES_TRAIN].fillna(0)
    y_tr = train_df['target_non_bn_won']
    X_te = test_df[FEATURE_NAMES_TRAIN].fillna(0)
    rf = RandomForestClassifier(**cfg['rf_params'], class_weight='balanced', random_state=42).fit(X_tr, y_tr)
    xgb = XGBClassifier(**cfg['xgb_params'], eval_metric='logloss', random_state=42).fit(X_tr, y_tr)
    return (rf.predict_proba(X_te)[:, 1] + xgb.predict_proba(X_te)[:, 1]) / 2


def score(y, proba):
    pred = (proba >= 0.5).astype(int)
    out = {'correct': int((pred == y).sum()), 'n': int(len(y)),
           'brier': round(float(brier_score_loss(y, proba)), 4)}
    out['auc'] = round(float(roc_auc_score(y, proba)), 3) if y.nunique() == 2 else None
    return out, pred


def evaluate_state(state):
    cfg = TRAINING_CONFIGS[state]
    ballots, stats = load_state(state)
    steps = list(cfg['transitions']) + ([FINAL_TEST[state]] if state in FINAL_TEST else [])
    rows = []
    for j in range(1, len(steps)):
        train = pd.concat([features(ballots, stats, state, a, b) for a, b in steps[:j]], ignore_index=True)
        a, b = steps[j]
        test = features(ballots, stats, state, a, b)
        y = test['target_non_bn_won']
        proba = fit_predict(cfg, train, test)
        model, pred = score(y, proba)
        never = (y == 0).sum()
        last = test['winner_coalition_a'].isin(OPPOSITION).astype(int)
        same = (last == y).sum()
        rows.append({
            'state': state, 'train': [f'{x}->{z}' for x, z in steps[:j]], 'test': f'{a}->{b}',
            'is_real_2026': b == 2026,
            'seats': int(len(y)), 'opposition_wins': int(y.sum()),
            'model': model,
            'never_opposition_correct': int(never),
            'same_as_last_correct': int(same),
            'model_vs_same_as_last_differ_on': int((pred != last.values).sum()),
            'predicted_opposition': int(pred.sum()),
            'false_positive': int(((pred == 1) & (y.values == 0)).sum()),   # said opposition, was not
            'false_negative': int(((pred == 0) & (y.values == 1)).sum()),   # missed an opposition win
        })
    return rows


def main():
    states = sys.argv[1:] or list(TRAINING_CONFIGS.keys())
    results = []
    print(f"{'state':<13}{'train on':<28}{'test':<11}{'n':>4}{'opp':>5}"
          f"{'MODEL':>9}{'never-opp':>11}{'same-as-last':>14}{'Brier':>8}{'AUC':>7}{'pred-opp':>9}{'FP':>4}{'FN':>4}")
    print('-' * 125)
    for st in states:
        try:
            rows = evaluate_state(st)
        except Exception as e:
            print(f'{st}: skipped ({type(e).__name__}: {e})')
            continue
        for r in rows:
            m = r['model']
            star = ' *' if r['is_real_2026'] else ''
            print(f"{st:<13}{'+'.join(r['train']):<28}{r['test']:<11}{r['seats']:>4}{r['opposition_wins']:>5}"
                  f"{m['correct']:>5}/{m['n']:<3}{r['never_opposition_correct']:>8}/{m['n']:<3}"
                  f"{r['same_as_last_correct']:>10}/{m['n']:<3}{m['brier']:>8.3f}"
                  f"{(m['auc'] if m['auc'] is not None else float('nan')):>7.3f}"
                  f"{r['predicted_opposition']:>9}{r['false_positive']:>4}{r['false_negative']:>4}{star}")
            results.append(r)
    print("\n* = the real 2026 election. 'never-opp' and 'same-as-last' are the naive baselines (seats correct).")
    print("pred-opp = seats the model called opposition (compare with 'opp'); FP = said opposition but was not; FN = missed an opposition win.")
    out = ROOT / 'backend' / 'models' / 'holdout_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f'Saved {out}')


if __name__ == '__main__':
    main()