"""
model_training.py
=================
Trains a Random Forest classifier for each GE year and saves:
  - model_{year}.joblib      → the trained pipeline
  - model_report_{year}.txt  → full evaluation metrics (accuracy, precision,
                               recall, F1, confusion matrix, cross-validation)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from analysis import get_database, get_year_analysis
import gedataset_pipeline as gep

YEARS        = ["GE12", "GE13", "GE14"]
FEATURES     = ['turnout_rate', 'relative_vote_margin', 'log_total_voters',
                'state', 'region', 'standardized_coalition']
REPORTS_DIR  = os.path.dirname(os.path.abspath(__file__))


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, X, y, year: str) -> dict:
    """
    Run a full suite of evaluation metrics and return them as a dict.
    Also prints a human-readable summary.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Core metrics
    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred).tolist()
    report    = classification_report(y_test, y_pred, target_names=["Loss", "Win"])

    # Cross-validation (5-fold stratified) on the full dataset
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    metrics = {
        "year":              year,
        "timestamp":         datetime.now().isoformat(),
        "test_size":         len(y_test),
        "train_size":        len(y) - len(y_test),
        "accuracy":          round(acc, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "confusion_matrix":  cm,
        "cv_mean_accuracy":  round(float(np.mean(cv_scores)), 4),
        "cv_std_accuracy":   round(float(np.std(cv_scores)), 4),
        "cv_scores":         [round(s, 4) for s in cv_scores],
    }

    print(f"\n{'='*52}")
    print(f"  MODEL EVALUATION REPORT — {year}")
    print(f"{'='*52}")
    print(f"  Train / Test split : {metrics['train_size']} / {metrics['test_size']} rows")
    print(f"  Accuracy           : {acc:.2%}")
    print(f"  Precision          : {precision:.2%}")
    print(f"  Recall             : {recall:.2%}")
    print(f"  F1 Score           : {f1:.2%}")
    print(f"\n  Cross-Validation (5-fold):")
    print(f"    Scores : {[f'{s:.2%}' for s in cv_scores]}")
    print(f"    Mean   : {np.mean(cv_scores):.2%}  ±  {np.std(cv_scores):.2%}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred Loss  Pred Win")
    print(f"  Actual Loss    {cm[0][0]:>5}     {cm[0][1]:>5}")
    print(f"  Actual Win     {cm[1][0]:>5}     {cm[1][1]:>5}")
    print(f"\n  Classification Report:")
    for line in report.split('\n'):
        print(f"    {line}")
    print(f"{'='*52}\n")

    return metrics


def save_report(metrics: dict, year: str):
    """Save metrics as both a human-readable .txt and a machine-readable .json."""
    base = os.path.join(REPORTS_DIR, f"model_report_{year}")

    # ── TXT ──
    with open(f"{base}.txt", "w") as f:
        f.write(f"MODEL EVALUATION REPORT — {year}\n")
        f.write(f"Generated: {metrics['timestamp']}\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"Train rows     : {metrics['train_size']}\n")
        f.write(f"Test rows      : {metrics['test_size']}\n\n")
        f.write(f"Accuracy       : {metrics['accuracy']:.2%}\n")
        f.write(f"Precision      : {metrics['precision']:.2%}\n")
        f.write(f"Recall         : {metrics['recall']:.2%}\n")
        f.write(f"F1 Score       : {metrics['f1_score']:.2%}\n\n")
        f.write(f"CV Mean Acc    : {metrics['cv_mean_accuracy']:.2%}\n")
        f.write(f"CV Std         : ±{metrics['cv_std_accuracy']:.2%}\n")
        f.write(f"CV Scores      : {metrics['cv_scores']}\n\n")
        cm = metrics['confusion_matrix']
        f.write("Confusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"             Pred Loss  Pred Win\n")
        f.write(f"Actual Loss    {cm[0][0]:>5}     {cm[0][1]:>5}\n")
        f.write(f"Actual Win     {cm[1][0]:>5}     {cm[1][1]:>5}\n")
    print(f"  📄 Saved: model_report_{year}.txt")

    # ── JSON (for API endpoint) ──
    with open(f"{base}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄 Saved: model_report_{year}.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all_models():
    all_metrics = {}

    for year in YEARS:
        print(f"\n⏳ Training model for {year}...")

        # 1. Fetch + clean data for this year
        df_raw  = get_database(year=year)
        if df_raw.empty:
            print(f"  ❌ Skipping {year}: No data found in Supabase.")
            continue

        df = gep.clean_dataset(df_raw)

        # 2. Verify required features exist
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            print(f"  ⚠️  Skipping {year}: Missing columns: {missing}")
            continue

        # 3. Prepare X / y
        X = df[FEATURES]
        y = df['is_winner']

        if y.nunique() < 2:
            print(f"  ⚠️  Skipping {year}: Only one class in target — cannot train.")
            continue

        # 4. Train via analysis.py pipeline (keeps encoding consistent with predictor)
        model_pipeline = get_year_analysis(df, year)
        if model_pipeline is None:
            print(f"  ❌ Training failed for {year}.")
            continue

        # 5. Evaluate — split must match what get_year_analysis uses (test_size=0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        metrics = evaluate_model(model_pipeline, X_test, y_test, X, y, year)
        all_metrics[year] = metrics

        # 6. Save model + report
        model_path = os.path.join(REPORTS_DIR, f"model_{year}.joblib")
        joblib.dump(model_pipeline, model_path)
        print(f"  ✅ Saved: model_{year}.joblib")
        save_report(metrics, year)

    # ── Summary table ──
    if all_metrics:
        print("\n" + "=" * 52)
        print("  SUMMARY — ALL YEARS")
        print("=" * 52)
        print(f"  {'Year':<6} {'Accuracy':>10} {'F1':>8} {'CV Mean':>10} {'CV ±':>8}")
        print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
        for yr, m in all_metrics.items():
            print(f"  {yr:<6} {m['accuracy']:>10.2%} {m['f1_score']:>8.2%} "
                  f"{m['cv_mean_accuracy']:>10.2%} {m['cv_std_accuracy']:>7.2%}")
        print("=" * 52)

    return all_metrics


if __name__ == "__main__":
    generate_all_models()