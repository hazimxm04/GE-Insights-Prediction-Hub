#!/usr/bin/env python
"""
Train state election models.
Usage: python backend/scripts/train_models.py
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)

from backend.core.pipelines.state_pipeline import StateElectionPipeline

# ── Config ────────────────────────────────────────────────────────

STATES   = ['johor', 'neg_sembilan', 'melaka']
FEATURES = [
    'majority_change',
    'turnout_change',
    'incumbent_held',
    'log_voters',
    'majority_perc_change',
    'n_candidates_b',
]
TARGET   = 'target_non_bn_won'
MODELS_DIR = ROOT / "backend" / "models"

# ── Evaluation helper ─────────────────────────────────────────────

def evaluate(name, model, X_test, y_test):
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, pred)
    prec  = precision_score(y_test, pred, zero_division=0)
    rec   = recall_score(y_test, pred, zero_division=0)
    f1    = f1_score(y_test, pred, zero_division=0)
    auc   = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)

    print(f"\n  [{name}]")
    print(f"    Accuracy:  {acc:.2%}")
    print(f"    Precision: {prec:.2%}")
    print(f"    Recall:    {rec:.2%}")
    print(f"    F1:        {f1:.2%}")
    print(f"    AUC-ROC:   {auc:.4f}")
    print(f"    Brier:     {brier:.4f}  ← lower is better")

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'brier': round(brier, 4),
    }, proba

def ablation_study(model, X_test, y_test, feature_names):
    """Drop each feature, measure accuracy drop"""
    baseline = accuracy_score(y_test, model.predict(X_test))
    results  = {}

    print(f"\n  Ablation Study (baseline accuracy: {baseline:.2%}):")
    for feat in feature_names:
        X_ablated = X_test.copy()
        X_ablated[feat] = 0
        acc_drop = baseline - accuracy_score(y_test, model.predict(X_ablated))
        results[feat] = round(acc_drop, 4)
        bar = '█' * max(0, int(acc_drop * 200))
        print(f"    {feat:25s}: {bar} {acc_drop:+.4f}")

    return results

# ── Main training ─────────────────────────────────────────────────

def train_state(state: str):
    print(f"\n{'='*60}")
    print(f"  TRAINING: {state.upper()}")
    print(f"{'='*60}")

    # 1. Load features from pipeline
    pipeline = StateElectionPipeline(state)
    X, y, df = pipeline.get_train_test()

    if X.empty:
        print(f"  ❌ No data for {state}")
        return None

    print(f"\n  Seats: {len(X)}")
    print(f"  Class balance: BN={( y==0).sum()}, non-BN={(y==1).sum()}")
    print(f"  Features: {FEATURES}")

    # 2. Use ALL data for training (small dataset — no random split)
    # Validation is done on actual 2026 elections later
    X_train = X.copy()
    y_train = y.copy()

    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_train_sc = pd.DataFrame(X_train_sc, columns=FEATURES)

    # 3. Train models (class_weight='balanced' handles imbalance)
    print(f"\n  Training models...")

    # Logistic Regression (baseline)
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_sc, y_train)
    lr_metrics, lr_proba = evaluate("Logistic Regression", lr, X_train_sc, y_train)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_proba = evaluate("Random Forest", rf, X_train, y_train)

    # XGBoost (GradientBoosting)
    xgb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=3,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_metrics, xgb_proba = evaluate("XGBoost", xgb, X_train, y_train)

    # Calibrated RF (honest probabilities)
    rf_cal = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42
        ),
        cv=3,
        method='isotonic'
    )
    rf_cal.fit(X_train, y_train)
    rf_cal_metrics, rf_cal_proba = evaluate("RF Calibrated", rf_cal, X_train, y_train)

    # Ensemble: average RF + XGB probabilities
    ens_proba = (rf_proba + xgb_proba) / 2
    ens_pred  = (ens_proba >= 0.5).astype(int)
    ens_acc   = accuracy_score(y_train, ens_pred)
    ens_auc   = roc_auc_score(y_train, ens_proba)
    ens_brier = brier_score_loss(y_train, ens_proba)

    print(f"\n  [Ensemble RF+XGB]")
    print(f"    Accuracy: {ens_acc:.2%}")
    print(f"    AUC-ROC:  {ens_auc:.4f}")
    print(f"    Brier:    {ens_brier:.4f}")

    # 4. Confusion matrix (ensemble)
    cm = confusion_matrix(y_train, ens_pred)
    print(f"\n  Confusion Matrix (Ensemble):")
    print(f"             Pred BN  Pred non-BN")
    print(f"  Actual BN    {cm[0,0]:>4}       {cm[0,1]:>4}")
    print(f"  Actual nBN   {cm[1,0]:>4}       {cm[1,1]:>4}")

    # 5. Feature importance + ablation
    rf_importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n  RF Feature Importance:")
    for _, row in rf_importance.iterrows():
        bar = '█' * int(row['importance'] * 60)
        print(f"    {row['feature']:25s}: {bar} {row['importance']:.4f}")

    ablation = ablation_study(rf, X_train, y_train, FEATURES)

    # 6. Train OOD detector
    print(f"\n  Training OOD Detector...")
    ood = EllipticEnvelope(contamination=0.1, random_state=42)
    ood.fit(X_train.values)
    ood_pred   = ood.predict(X_train.values)
    ood_count  = (ood_pred == -1).sum()
    print(f"    OOD seats detected: {ood_count}/{len(X_train)} ({ood_count/len(X_train):.1%})")

    # 7. Save everything
    out_dir = MODELS_DIR / state
    out_dir.mkdir(parents=True, exist_ok=True)

    pickle.dump(rf,     open(out_dir / "rf_model.pkl",      "wb"))
    pickle.dump(xgb,    open(out_dir / "xgb_model.pkl",     "wb"))
    pickle.dump(rf_cal, open(out_dir / "rf_cal_model.pkl",  "wb"))
    pickle.dump(scaler, open(out_dir / "scaler.pkl",         "wb"))
    pickle.dump(ood,    open(out_dir / "ood_detector.pkl",   "wb"))

    print(f"\n  ✅ Saved models to backend/models/{state}/")

    # 8. Save metadata
    metadata = {
        'state': state,
        'seats': len(X),
        'features': FEATURES,
        'target': TARGET,
        'class_balance': {'BN': int((y==0).sum()), 'non_BN': int((y==1).sum())},
        'models': {
            'logistic_regression': lr_metrics,
            'random_forest':       rf_metrics,
            'xgboost':             xgb_metrics,
            'rf_calibrated':       rf_cal_metrics,
            'ensemble': {
                'accuracy': round(ens_acc, 4),
                'auc':      round(ens_auc, 4),
                'brier':    round(ens_brier, 4),
            }
        },
        'feature_importance': rf_importance.set_index('feature')['importance'].round(4).to_dict(),
        'ablation': ablation,
        'ood': {
            'n_flagged': int(ood_count),
            'pct_flagged': round(ood_count/len(X_train)*100, 1)
        }
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✅ Saved metadata.json")
    return metadata


def main():
    all_results = {}

    for state in STATES:
        result = train_state(state)
        if result:
            all_results[state] = result

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'State':<18} {'LR':>7} {'RF':>7} {'XGB':>7} {'Ens':>7} {'Brier':>8}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for state, r in all_results.items():
        m = r['models']
        print(
            f"  {state:<18} "
            f"{m['logistic_regression']['accuracy']:>7.2%} "
            f"{m['random_forest']['accuracy']:>7.2%} "
            f"{m['xgboost']['accuracy']:>7.2%} "
            f"{m['ensemble']['accuracy']:>7.2%} "
            f"{m['ensemble']['brier']:>8.4f}"
        )

    print(f"\n✅ All models trained and saved!")
    print(f"   Next: python backend/scripts/validate_2026.py")


if __name__ == "__main__":
    main()