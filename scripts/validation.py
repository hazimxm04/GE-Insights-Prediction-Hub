"""
scripts/validation.py
======================
Validates trained models against actual election results (binary target).

Only Johor and Neg Sembilan have held their 2026 elections so far.
Selangor, Melaka, and Perak elections are pending -- no ground truth
to score against yet for those states.

Usage:
    python scripts/validation.py
    python scripts/validation.py johor
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN
from train_models import build_transition, STATE_URL_PREFIX

VALIDATABLE_STATES = {
    'johor':        (2022, 2026),
    'neg_sembilan': (2023, 2026),
}


def load_models(state: str):
    models_dir = ROOT / 'backend/models' / state
    rf  = pickle.load(open(models_dir / 'rf_model.pkl', 'rb'))
    xgb = pickle.load(open(models_dir / 'xgb_model.pkl', 'rb'))
    ood = pickle.load(open(models_dir / 'ood_detector.pkl', 'rb'))
    return rf, xgb, ood


def build_eval_set(state: str, year_from: int, year_to: int):
    prefix = STATE_URL_PREFIX[state]
    url_ballots = f'https://lake.electiondata.my/results_headline/headline_ballots_state_{prefix}.parquet'
    url_stats   = f'https://lake.electiondata.my/results_headline/headline_stats_state_{prefix}.parquet'

    ballots = pd.read_parquet(url_ballots)
    stats   = pd.read_parquet(url_stats)

    df = build_transition(ballots, stats, year_from, year_to)

    config = TRAINING_CONFIGS[state]
    eth_path = ROOT / 'data/raw/ethnicity' / f'ethnicity_{state}_{config["ethnicity_year"]}.csv'
    eth_cols = ['malay_pct', 'chinese_pct', 'indian_pct', 'young_malay_pct',
                'young_chinese_pct', 'older_malay_pct', 'youth_pct', 'median_age']

    if eth_path.exists():
        eth_df = pd.read_csv(eth_path)
        df = df.merge(eth_df[['seat'] + eth_cols], on='seat', how='left')
    else:
        for col in eth_cols:
            df[col] = 0

    for col in eth_cols:
        if col not in df.columns:
            df[col] = 0

    for col in ['bn_sent_x_malay', 'harapan_sent_x_chinese', 'pn_sent_x_young_malay',
                'tension_x_mixed', 'economic_x_youth', 'narrative_pressure']:
        df[col] = 0

    return df


def validate_state(state: str):
    print(f"\n{'='*70}")
    print(f"  VALIDATION: {state.upper()}")
    year_from, year_to = VALIDATABLE_STATES[state]
    print(f"  (Train transition features: {year_from}->{year_to}, ground truth: {year_to} actual)")
    print(f"{'='*70}")

    df = build_eval_set(state, year_from, year_to)
    print(f"\nSeats: {len(df)}")

    X = df[FEATURE_NAMES_TRAIN].fillna(0)
    y_actual = df['target_non_bn_won']

    rf, xgb, ood = load_models(state)

    rf_proba  = rf.predict_proba(X)[:, 1]
    xgb_proba = xgb.predict_proba(X)[:, 1]
    ens_proba = (rf_proba + xgb_proba) / 2
    y_pred    = (ens_proba >= 0.5).astype(int)

    ood_pred = ood.predict(X.values)
    ood_mask = (ood_pred == -1)

    accuracy = accuracy_score(y_actual, y_pred)
    correct  = (y_pred == y_actual.values).sum()
    total    = len(df)

    print(f"\nAccuracy: {accuracy:.2%} ({correct}/{total} seats)")

    cm = confusion_matrix(y_actual, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  BN correct: {tn}, Predicted non-BN (wrong): {fp}")
    print(f"  Non-BN missed: {fn}, Non-BN correct: {tp}")

    print(f"\nOOD Detection: {ood_mask.sum()}/{total} flagged ({ood_mask.mean():.1%})")

    df_results = df[['seat', 'winner_coalition_a', 'winner_coalition_b']].copy()
    df_results['predicted_non_bn'] = y_pred
    df_results['actual_non_bn']    = y_actual.values
    df_results['probability']      = ens_proba.round(4)
    df_results['correct']          = (y_pred == y_actual.values)
    df_results['is_ood']           = ood_mask
    df_results['ood_score']        = (-ood.score_samples(X.values)).round(3)

    out_path = ROOT / 'backend/models' / state / 'validation_2026.csv'
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    wrong = df_results[~df_results['correct']]
    if len(wrong) > 0:
        print(f"\nWrong predictions ({len(wrong)} seats):")
        for _, row in wrong.iterrows():
            pred_label = "non-BN" if row['predicted_non_bn'] else "BN"
            actual_label = "non-BN" if row['actual_non_bn'] else "BN"
            ood_flag = "  OOD" if row['is_ood'] else ""
            print(f"    {row['seat']:<35} Pred: {pred_label:<8} Actual: {actual_label:<8}"
                  f" P={row['probability']:.2f}{ood_flag}")
    else:
        print(f"\nAll seats predicted correctly!")

    return {
        'state': state,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'ood_pct': round(ood_mask.mean() * 100, 1),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate election predictions')
    parser.add_argument('states', nargs='*',
                         help='States to validate (default: all states with ground truth)')
    args = parser.parse_args()

    states_to_validate = args.states if args.states else list(VALIDATABLE_STATES.keys())

    results = []
    for state in states_to_validate:
        if state not in TRAINING_CONFIGS:
            print(f"Unknown state: {state}")
            continue
        if state not in VALIDATABLE_STATES:
            print(f"\n{state.upper()}: 2026 election not yet held -- no ground truth available")
            print(f"  Model can still generate live forecasts via StatePredictor.")
            continue

        result = validate_state(state)
        if result:
            results.append(result)

    print(f"\n\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'State':<15} {'Accuracy':<12} {'Correct':<10} {'OOD%':<8}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['state']:<15} {r['accuracy']:>10.2%}  {r['correct']:>3}/{r['total']:<3}    {r['ood_pct']:>5.1f}%")

    print("\nValidation complete")
    print("  (Selangor/Melaka/Perak: pending election results)")


if __name__ == "__main__":
    main()