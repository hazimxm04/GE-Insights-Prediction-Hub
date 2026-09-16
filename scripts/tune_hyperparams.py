"""
scripts/tune_hyperparams.py
============================
Systematic hyperparameter search for the 3-class coalition model
(BN=0, Pakatan=1, PN=2), designed around severe class imbalance.

WHY THIS EXISTS:
    Manually-guessed hyperparameters (in training_config.py) optimize
    nothing in particular. This script searches a grid of hyperparameter
    combinations and picks the one that scores best on f1_macro under
    StratifiedKFold cross-validation.

WHY StratifiedKFold (not regular KFold):
    PN is extremely rare (as few as 3 examples in some states). Regular
    random KFold could put ALL PN examples in one fold, leaving other
    folds unable to evaluate PN performance at all. StratifiedKFold
    preserves class proportions across folds so every fold gets a
    chance to score PN.

WHY f1_macro (not accuracy):
    Accuracy rewards a model that always predicts the majority class
    (BN) and ignores PN entirely, since PN is rare. f1_macro computes
    F1 separately per class then averages them EQUALLY (not weighted
    by frequency), so a model that ignores PN gets punished even if
    overall accuracy looks fine.

LIMITATION TO KNOW:
    With PN counts as low as 3 in some states, 3-fold CV puts ~1 PN
    example per fold. This is close to the mathematical floor for
    what cross-validation can meaningfully evaluate — the search will
    still run, but treat PN-specific scores with appropriate caution
    for the smallest states.

Usage:
    python scripts/tune_hyperparams.py johor
    python scripts/tune_hyperparams.py            # all states
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN
from train_models import build_transition

STATE_URL_PREFIX = {
    'johor': 'jhr', 'neg_sembilan': 'nsn', 'selangor': 'sgr',
    'melaka': 'mlk', 'perak': 'prk',
}


def load_training_data(state: str, config: dict):
    """Rebuild the same training set train_models.py uses (all transitions combined)."""
    prefix = STATE_URL_PREFIX[state]
    ballots = pd.read_parquet(f'https://lake.electiondata.my/results_headline/headline_ballots_state_{prefix}.parquet')
    stats   = pd.read_parquet(f'https://lake.electiondata.my/results_headline/headline_stats_state_{prefix}.parquet')

    transitions = [build_transition(ballots, stats, a, b) for a, b in config['transitions']]
    df = pd.concat(transitions, ignore_index=True)

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

    X = df[FEATURE_NAMES_TRAIN].fillna(0)
    y = df['target_coalition']
    return X, y


def tune_state(state: str):
    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SEARCH: {state.upper()}")
    print(f"{'='*70}")

    config = TRAINING_CONFIGS[state]
    X, y = load_training_data(state, config)

    class_counts = y.value_counts().sort_index()
    print(f"\nClass distribution: {dict(class_counts)}")

    min_class_count = class_counts.min()
    n_splits = min(3, min_class_count)

    if n_splits < 2:
        print(f"  ⚠️  Smallest class has {min_class_count} example(s) — "
              f"cannot run cross-validated search. Skipping {state}.")
        return None

    if n_splits < 3:
        print(f"  ⚠️  Using {n_splits}-fold CV instead of 3-fold "
              f"(smallest class has only {min_class_count} examples)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ── Random Forest search ────────────────────────────────────
    rf_grid = {
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [3, 5, 8],
        'n_estimators': [100, 150, 200],
    }

    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        rf_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
    )
    rf_search.fit(X, y)

    print(f"\n  [Random Forest]")
    print(f"    Best params: {rf_search.best_params_}")
    print(f"    Best f1_macro: {rf_search.best_score_:.4f}")

    # ── XGBoost search ───────────────────────────────────────────
    xgb_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'n_estimators': [100, 150, 200],
    }

    xgb_search = GridSearchCV(
        XGBClassifier(eval_metric='mlogloss', random_state=42),
        xgb_grid,
        cv=skf,
        scoring='f1_macro',
        n_jobs=-1,
    )
    xgb_search.fit(X, y)

    print(f"\n  [XGBoost]")
    print(f"    Best params: {xgb_search.best_params_}")
    print(f"    Best f1_macro: {xgb_search.best_score_:.4f}")

    print(f"\n  Compare to current hardcoded config:")
    print(f"    rf_params:  {config['rf_params']}")
    print(f"    xgb_params: {config['xgb_params']}")

    return {
        'state': state,
        'rf_best_params': rf_search.best_params_,
        'rf_best_score': rf_search.best_score_,
        'xgb_best_params': xgb_search.best_params_,
        'xgb_best_score': xgb_search.best_score_,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Tune hyperparameters per state')
    parser.add_argument('states', nargs='*', help='States to tune (default: all)')
    args = parser.parse_args()

    states = args.states if args.states else list(TRAINING_CONFIGS.keys())

    results = []
    for state in states:
        if state not in TRAINING_CONFIGS:
            print(f"❌ Unknown state: {state}")
            continue
        result = tune_state(state)
        if result:
            results.append(result)

    print(f"\n\n{'='*70}")
    print(f"  TUNING SUMMARY")
    print(f"{'='*70}")
    for r in results:
        print(f"\n{r['state'].upper()}:")
        print(f"  rf_params  = {r['rf_best_params']}  (f1_macro={r['rf_best_score']:.4f})")
        print(f"  xgb_params = {r['xgb_best_params']}  (f1_macro={r['xgb_best_score']:.4f})")

    print(f"\n✅ Copy the best params above into training_config.py manually,")
    print(f"   then retrain with: python scripts/train_models.py")


if __name__ == "__main__":
    main()