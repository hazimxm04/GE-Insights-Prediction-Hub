"""
eval/baseline_eval.py
======================
Measures Groq's performance against your human labels.

WHY THIS EXISTS:
  After labeling, you know what the right answers are.
  This script measures how well Groq's scores match your labels.
  The output is your BASELINE — the number your fine-tuned model
  needs to beat.

METRICS:
  F1 score (per theme + macro average)
    — balances precision and recall, right for imbalanced labels
    — this is the standard metric for multi-label classification

  Precision: of articles Groq flagged as having a theme,
             how many actually did?
  Recall:    of articles that actually had a theme,
             how many did Groq catch?

  Groq threshold: 0.3 — Groq score >= 0.3 → "theme present"
  You can tune this to see how it affects F1.

OUTPUT:
  eval/baseline_results.json   ← saved so compare_models.py can use it
  printed table in terminal

Usage:
    python eval/baseline_eval.py
    python eval/baseline_eval.py --threshold 0.4
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────

THEME_COLS = [
    'islam_threat',
    'malay_unity',
    'cost_living',
    'corruption',
    'govt_performance',
]

DEFAULT_INPUT     = Path("eval/eval_set.csv")
DEFAULT_OUTPUT    = Path("eval/baseline_results.json")
DEFAULT_THRESHOLD = 0.3


# ── Metrics ───────────────────────────────────────────────────────────

def precision_recall_f1(y_true, y_pred):
    """Manual P/R/F1 — no sklearn needed."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return round(precision, 3), round(recall, 3), round(f1, 3)


# ── Main ──────────────────────────────────────────────────────────────

def run_baseline_eval(
    input_path: Path,
    output_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
):
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)

    # only evaluate labeled articles
    # fill NaN with empty string first so comparison works
    human_col    = f'human_{THEME_COLS[0]}'
    df[human_col] = df[human_col].fillna("").astype(str)
    labeled_mask = df[human_col].str.strip() != ""
    df_labeled   = df[labeled_mask].copy()

    # fill all human columns to be safe
    for theme in THEME_COLS:
        col = f'human_{theme}'
        df_labeled[col] = df_labeled[col].fillna(0).astype(str).str.strip()
        df_labeled[col] = pd.to_numeric(df_labeled[col], errors='coerce').fillna(0)

    print(f"  Total articles:  {len(df)}")
    print(f"  Labeled:         {len(df_labeled)}")
    print(f"  Groq threshold:  {threshold}  (score >= threshold → theme present)")

    if len(df_labeled) == 0:
        print("\n  No labeled articles yet. Run label_articles.py first.")
        return

    if len(df_labeled) < 30:
        print(f"\n  WARNING: only {len(df_labeled)} labeled articles. "
              f"Results may not be reliable. Aim for 80+.")

    results = {}
    f1_scores = []

    print(f"\n{'='*65}")
    print(f"  {'Theme':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'Support':>9}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")

    for theme in THEME_COLS:
        # ground truth: your human labels (already cleaned above)
        y_true = df_labeled[f'human_{theme}'].astype(int).tolist()

        # groq prediction: threshold the continuous score
        y_pred = (df_labeled[theme] >= threshold).astype(int).tolist()

        p, r, f1 = precision_recall_f1(y_true, y_pred)
        support   = sum(y_true)   # how many articles actually had this theme
        f1_scores.append(f1)

        results[theme] = {
            'precision': p,
            'recall':    r,
            'f1':        f1,
            'support':   support,
            'threshold': threshold,
        }

        print(f"  {theme:<20} {p:>10.3f} {r:>8.3f} {f1:>8.3f} "
              f"{support:>9}")

    macro_f1 = round(float(np.mean(f1_scores)), 3)
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    print(f"  {'MACRO AVERAGE':<20} {'':>10} {'':>8} {macro_f1:>8.3f}")

    print(f"\n  Labeled articles: {len(df_labeled)}")
    print(f"  Groq macro F1:    {macro_f1}  ← this is your baseline to beat")

    # save results
    output = {
        'model':          'groq_baseline',
        'model_name':     'qwen/qwen3.8-27b via Groq API',
        'threshold':      threshold,
        'n_labeled':      len(df_labeled),
        'macro_f1':       macro_f1,
        'per_theme':      results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to: {output_path}")
    print(f"\n  Next step: finetune/prepare_dataset.py")
    print(f"  (after that: finetune/train.py  →  eval/compare_models.py)")

    return output


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Groq baseline against human labels"
    )
    parser.add_argument(
        "--input",     type=Path,  default=DEFAULT_INPUT,
        help=f"Path to labeled eval_set.csv (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output",    type=Path,  default=DEFAULT_OUTPUT,
        help=f"Where to save results JSON (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Groq score threshold for 'theme present' (default: {DEFAULT_THRESHOLD})"
    )
    args = parser.parse_args()

    run_baseline_eval(args.input, args.output, args.threshold)