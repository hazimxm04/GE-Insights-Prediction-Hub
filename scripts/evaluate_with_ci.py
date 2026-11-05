"""
scripts/evaluate_with_ci.py
===========================
How sure are we about the 2026 results? Adds uncertainty to the headline numbers.

For Johor and Negeri Sembilan (the states that have voted) it reports, with 95% bootstrap
confidence intervals (seats resampled with replacement):

  * model accuracy, AUC and Brier score
  * accuracy of the two simple rules ("never opposition", "same as last time")
  * the paired difference (model minus rule) in correct seats, with its own interval and an
    exact McNemar test (do the model and the rule disagree in a lopsided way?)

Why: with 36 to 56 seats, a gap of one or two seats is inside the noise. This script puts a
number on that, so the README can say what the data can and cannot show.

    python scripts/evaluate_with_ci.py            # prints a table
    python scripts/evaluate_with_ci.py --save     # also writes docs/evaluation_with_ci.json

Input: backend/models/<state>/validation_2026.csv (already in the repo). Nothing is retrained.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.features import OPPOSITION, COALITION_RENAME

STATES = {'johor': 'Johor', 'neg_sembilan': 'Negeri Sembilan'}
N_BOOT = 10000
SEED = 20261104


def load(state: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / 'backend' / 'models' / state / 'validation_2026.csv')
    last = df['winner_coalition_a'].replace(COALITION_RENAME)
    return pd.DataFrame({
        'y': df['actual_non_bn'].astype(int),
        'model': df['predicted_non_bn'].astype(int),
        'prob': df['probability'].astype(float),
        'never_opp': 0,
        'same_as_last': last.isin(OPPOSITION).astype(int),
    })


def ci(values, lo=2.5, hi=97.5):
    return [round(float(np.percentile(values, lo)), 4), round(float(np.percentile(values, hi)), 4)]


def mcnemar_exact(model_correct, rule_correct) -> float:
    """Exact two-sided McNemar test on the seats where exactly one of the two is right."""
    only_model = int(((model_correct == 1) & (rule_correct == 0)).sum())
    only_rule = int(((model_correct == 0) & (rule_correct == 1)).sum())
    n = only_model + only_rule
    return 1.0 if n == 0 else float(binomtest(only_model, n, 0.5).pvalue)


def evaluate(df: pd.DataFrame, n_boot=N_BOOT, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(df)
    y = df['y'].to_numpy()
    correct = {k: (df[k].to_numpy() == y).astype(int) for k in ('model', 'never_opp', 'same_as_last')}

    idx = rng.integers(0, n, size=(n_boot, n))
    acc_b = {k: correct[k][idx].mean(axis=1) for k in correct}
    auc_b, brier_b = [], []
    for row in idx:
        yb, pb = y[row], df['prob'].to_numpy()[row]
        brier_b.append(brier_score_loss(yb, pb))
        if len(set(yb)) == 2:
            auc_b.append(roc_auc_score(yb, pb))

    out = {
        'seats': int(n),
        'opposition_won': int(y.sum()),
        'accuracy': {k: {'correct': int(correct[k].sum()), 'value': round(float(correct[k].mean()), 4),
                         'ci95': ci(acc_b[k])} for k in correct},
        'auc': {'value': round(float(roc_auc_score(y, df['prob'])), 4), 'ci95': ci(auc_b)},
        'brier': {'value': round(float(brier_score_loss(y, df['prob'])), 4), 'ci95': ci(brier_b)},
        'paired_vs_model': {},
    }
    for rule in ('never_opp', 'same_as_last'):
        diff_b = (acc_b['model'] - acc_b[rule]) * n          # in seats
        out['paired_vs_model'][rule] = {
            'model_minus_rule_seats': int(correct['model'].sum() - correct[rule].sum()),
            'ci95_seats': [round(float(np.percentile(diff_b, 2.5)), 1), round(float(np.percentile(diff_b, 97.5)), 1)],
            'mcnemar_exact_p': round(mcnemar_exact(correct['model'], correct[rule]), 4),
        }
    return out


def print_report(name: str, r: dict):
    n = r['seats']
    print(f"\n{name}: {n} seats, opposition won {r['opposition_won']}")
    for key, label in (('model', 'model'), ('never_opp', 'never opposition'), ('same_as_last', 'same as last time')):
        a = r['accuracy'][key]
        print(f"  {label:<18}{a['correct']:>3}/{n}  {a['value']:.1%}  95% CI [{a['ci95'][0]:.1%}, {a['ci95'][1]:.1%}]")
    print(f"  AUC   {r['auc']['value']:.3f}  95% CI [{r['auc']['ci95'][0]:.3f}, {r['auc']['ci95'][1]:.3f}]")
    print(f"  Brier {r['brier']['value']:.3f}  95% CI [{r['brier']['ci95'][0]:.3f}, {r['brier']['ci95'][1]:.3f}]")
    for rule, label in (('never_opp', 'never opposition'), ('same_as_last', 'same as last time')):
        p = r['paired_vs_model'][rule]
        print(f"  model minus {label}: {p['model_minus_rule_seats']:+d} seats, 95% CI "
              f"[{p['ci95_seats'][0]:+.1f}, {p['ci95_seats'][1]:+.1f}], McNemar exact p = {p['mcnemar_exact_p']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--save', action='store_true', help='write docs/evaluation_with_ci.json')
    args = ap.parse_args()
    results = {}
    for key, name in STATES.items():
        results[key] = evaluate(load(key))
        print_report(name, results[key])
    print(f"\n(bootstrap: {N_BOOT} resamples of seats, seed {SEED})")
    if args.save:
        path = ROOT / 'docs' / 'evaluation_with_ci.json'
        path.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(f'saved {path.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
