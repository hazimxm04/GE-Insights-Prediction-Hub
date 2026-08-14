#!/usr/bin/env python
"""
Validate trained models against 2026 actual election results.
Usage: python backend/scripts/validate_2026.py
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    accuracy_score, brier_score_loss,
    roc_auc_score, confusion_matrix, classification_report
)
from backend.core.pipelines.state_pipeline import StateElectionPipeline

MODELS_DIR = ROOT / "backend" / "models"
FEATURES   = [
    'majority_change',
    'turnout_change',
    'incumbent_held',
    'log_voters',
    'majority_perc_change',
    'n_candidates_b',
]

def load_models(state: str):
    d = MODELS_DIR / state
    return {
        'rf':     pickle.load(open(d / "rf_model.pkl",     "rb")),
        'xgb':    pickle.load(open(d / "xgb_model.pkl",    "rb")),
        'rf_cal': pickle.load(open(d / "rf_cal_model.pkl", "rb")),
        'ood':    pickle.load(open(d / "ood_detector.pkl",  "rb")),
    }

def validate_state(state: str):
    print(f"\n{'='*60}")
    print(f"  VALIDATING: {state.upper()} 2026")
    print(f"{'='*60}")

    # Load pipeline + get validation data
    pipeline = StateElectionPipeline(state)
    df_val   = pipeline.get_validation()

    if df_val.empty:
        print(f"  ⚠️  No validation data for {state}")
        return None

    sentiment = pipeline.load_sentiment_features()
    df_val['bn_sentiment']         = sentiment['bn_sentiment']
    df_val['harapan_sentiment']    = sentiment['harapan_sentiment']
    df_val['pn_sentiment']         = sentiment['pn_sentiment']
    df_val['racial_tension_index'] = sentiment['racial_tension_index']

    # Add economic pressure feature
    economic_pressure = pipeline.load_economic_features()
    df_val['economic_pressure'] = economic_pressure

    from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features
    val_year = pipeline.config['val_year']
    df_val = merge_ethnicity_into_features(
        df_features=df_val,
        state=state,
        year_b=val_year,
        sentiment=sentiment,
        economic_pressure=economic_pressure
    )

    FEATURE_COLS = [
        'majority_change', 'turnout_change', 'incumbent_held',
        'log_voters', 'majority_perc_change', 'n_candidates_b',
        'bn_sentiment', 'harapan_sentiment', 'pn_sentiment',
        'racial_tension_index', 'economic_pressure',
        'malay_pct', 'chinese_pct', 'indian_pct',
        'young_malay_pct', 'young_chinese_pct',
        'older_malay_pct', 'youth_pct', 'median_age',
        'bn_sent_x_malay', 'harapan_sent_x_chinese',
        'pn_sent_x_young_malay', 'tension_x_mixed',
        'economic_x_youth',
        'narrative_pressure',
    ]
    
    X_val = df_val[FEATURE_COLS].fillna(0)
    y_val = df_val['target_non_bn_won']

    print(f"\n  Validation seats: {len(X_val)}")
    print(f"  Actual 2026 results:")
    print(f"    BN won:     {(y_val==0).sum()} seats")
    print(f"    Non-BN won: {(y_val==1).sum()} seats")

    # Load models
    models = load_models(state)

    # Get probabilities
    rf_proba  = models['rf'].predict_proba(X_val)[:, 1]
    xgb_proba = models['xgb'].predict_proba(X_val)[:, 1]
    ens_proba = (rf_proba + xgb_proba) / 2
    ens_pred  = (ens_proba >= 0.5).astype(int)

    # OOD detection
    ood_pred   = models['ood'].predict(X_val.values)
    ood_scores = models['ood'].score_samples(X_val.values) # negative so higher = more OOD
    ood_mask   = (ood_pred == -1)

    # Overall metrics
    acc   = accuracy_score(y_val, ens_pred)
    brier = brier_score_loss(y_val, ens_proba)
    try:
        auc = roc_auc_score(y_val, ens_proba)
    except:
        auc = float('nan')

    print(f"\n  2026 Validation Results (Ensemble):")
    print(f"    Accuracy:  {acc:.2%}  ← REAL accuracy on unseen data")
    print(f"    Brier:     {brier:.4f}")
    print(f"    AUC-ROC:   {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, ens_pred)
    print(f"\n  Confusion Matrix:")
    print(f"             Pred BN  Pred non-BN")
    print(f"  Actual BN    {cm[0,0]:>4}       {cm[0,1]:>4}")
    print(f"  Actual nBN   {cm[1,0]:>4}       {cm[1,1]:>4}")

    # OOD breakdown
    print(f"\n  OOD Detection on 2026 seats:")
    print(f"    In-distribution: {(~ood_mask).sum()} seats")
    print(f"    OOD flagged:     {ood_mask.sum()} seats ({ood_mask.mean():.1%})")

    if (~ood_mask).any():
        in_acc = accuracy_score(y_val[~ood_mask], ens_pred[~ood_mask])
        print(f"    In-dist accuracy:  {in_acc:.2%}")

    if ood_mask.any():
        ood_acc = accuracy_score(y_val[ood_mask], ens_pred[ood_mask])
        print(f"    OOD accuracy:      {ood_acc:.2%}  ← harder seats")

    # Per-seat breakdown
    df_results = df_val[['seat', 'winner_coalition_b']].copy()
    df_results['winner_coalition_prev'] = df_val['winner_coalition_a'].values
    df_results['predicted_non_bn']      = ens_pred
    df_results['actual_non_bn']         = y_val.values
    df_results['probability']           = ens_proba.round(3)
    df_results['correct']               = (ens_pred == y_val.values)
    df_results['is_ood']                = ood_mask
    df_results['ood_score']             = ood_scores.round(3)

    # Show wrong predictions
    wrong = df_results[~df_results['correct']]
    if len(wrong) > 0:
        print(f"\n  ❌ Wrong predictions ({len(wrong)} seats):")
        for _, row in wrong.iterrows():
            pred_label   = "non-BN" if row['predicted_non_bn'] else "BN"
            actual_label = row['winner_coalition_b']
            ood_flag     = "⚠️ OOD" if row['is_ood'] else ""
            print(f"    {row['seat']:<35} Pred: {pred_label:<8} Actual: {actual_label:<10} P={row['probability']:.2f} {ood_flag}")
    else:
        print(f"\n  ✅ All seats predicted correctly!")

    # Save results
    out_path = MODELS_DIR / state / "validation_2026.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\n  ✅ Saved: backend/models/{state}/validation_2026.csv")

    return {
        'state':    state,
        'seats':    len(X_val),
        'accuracy': round(acc, 4),
        'brier':    round(brier, 4),
        'auc':      round(auc, 4),
        'ood_pct':  round(ood_mask.mean() * 100, 1),
        'wrong':    len(wrong),
    }


def main():
    all_results = {}

    for state in ['johor', 'neg_sembilan', 'melaka']:
        result = validate_state(state)
        if result:
            all_results[state] = result

    # Final summary
    print(f"\n\n{'='*60}")
    print(f"  2026 VALIDATION SUMMARY  (REAL accuracy on unseen data)")
    print(f"{'='*60}")
    print(f"  {'State':<18} {'Seats':>6} {'Accuracy':>10} {'Brier':>8} {'OOD%':>7} {'Wrong':>6}")
    print(f"  {'-'*18} {'-'*6} {'-'*10} {'-'*8} {'-'*7} {'-'*6}")

    for state, r in all_results.items():
        print(
            f"  {state:<18} {r['seats']:>6} "
            f"{r['accuracy']:>10.2%} "
            f"{r['brier']:>8.4f} "
            f"{r['ood_pct']:>6.1f}% "
            f"{r['wrong']:>6}"
        )

    print(f"\n  Note: Training accuracy was 100% (memorized).")
    print(f"  These are the HONEST numbers on actual 2026 elections.")
    print(f"\n✅ Validation complete!")
    print(f"   Next: python backend/scripts/build_predictor.py")


if __name__ == "__main__":
    main()