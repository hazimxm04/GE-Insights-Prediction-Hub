import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mlops'))
from model_versioning import version_and_promote
from alerts import alert_on_success, alert_on_failure

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)

from backend.core.features import (  # noqa: F401 (re-exported for other scripts/tests)
    build_transition, seat_no, previous_election_year, OPPOSITION, COALITION_RENAME, MIN_YEAR, LOOKBACK,
    FEATURE_VERSION,
)
from state_data import load_state_data, STATE_URL_PREFIX  # noqa: F401
from training_config import TRAINING_CONFIGS, FEATURE_NAMES_TRAIN, FEATURE_PREDICTORS

MODELS_DIR = ROOT / "backend" / "models"


# ── Evaluation helper ─────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test):
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
    print(f"    Brier:     {brier:.4f}  <- lower is better")

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'brier': round(brier, 4),
    }, proba


def compute_holdout(df_train, config):
    """Honest, time-ordered score: train on every transition except the latest one,
    test on the latest one. (The saved model is then trained on ALL transitions.)
    Returns None when there is only one transition."""
    last_b = int(df_train['year_b'].max())
    tr = df_train[df_train['year_b'] < last_b]
    te = df_train[df_train['year_b'] == last_b]
    if tr.empty or te.empty:
        return None
    X_tr, y_tr = tr[FEATURE_NAMES_TRAIN].fillna(0), tr['target_non_bn_won']
    X_te, y_te = te[FEATURE_NAMES_TRAIN].fillna(0), te['target_non_bn_won']
    rf = RandomForestClassifier(**config['rf_params'], class_weight='balanced', random_state=42).fit(X_tr, y_tr)
    xgb = XGBClassifier(**config['xgb_params'], eval_metric='logloss', random_state=42).fit(X_tr, y_tr)
    proba = (rf.predict_proba(X_te)[:, 1] + xgb.predict_proba(X_te)[:, 1]) / 2
    pred = (proba >= 0.5).astype(int)
    n = int(len(y_te))
    correct = int((pred == y_te.values).sum())
    last_same = te['winner_coalition_a'].isin(OPPOSITION).astype(int)
    return {
        'train_transitions': sorted({f"{int(a)}->{int(b)}" for a, b in zip(tr['year_a'], tr['year_b'])}),
        'test_transition': f"{int(te['year_a'].iloc[0])}->{last_b}",
        'seats': n,
        'correct': correct,
        'accuracy': round(correct / n, 4),
        'brier': round(float(brier_score_loss(y_te, proba)), 4),
        'auc': round(float(roc_auc_score(y_te, proba)), 3) if y_te.nunique() == 2 else None,
        'baseline_never_opposition': int((y_te == 0).sum()),
        'baseline_same_as_last': int((last_same == y_te).sum()),
    }


# ── Main training ─────────────────────────────────────────────────

def train_state(state: str, config: dict):
    print(f"\n{'='*70}")
    print(f"  TRAINING: {state.upper()}")
    print(f"  {config['description']}")
    print(f"{'='*70}")

    ballots, stats = load_state_data(state)
    print(f"  Ballots: {len(ballots)} rows, Stats: {len(stats)} rows")

    print(f"\nBuilding {len(config['transitions'])} training transitions...")
    transitions = []

    for year_a, year_b in config['transitions']:
        df = build_transition(ballots, stats, year_a, year_b)
        non_bn_count = df['target_non_bn_won'].sum()
        print(f"  {year_a}->{year_b}: {len(df)} seats, Non-BN wins={non_bn_count}")
        transitions.append(df)

    df_train = pd.concat(transitions, ignore_index=True)
    print(f"\nCombined training set: {len(df_train)} rows")

    # ── Ethnicity features ──────────────────────────────────────
    print(f"\nAdding ethnicity features...")

    eth_path = ROOT / 'data/raw/ethnicity' / f'ethnicity_{state}_{config["ethnicity_year"]}.csv'
    eth_cols = ['malay_pct', 'chinese_pct', 'indian_pct', 'young_malay_pct',
                'young_chinese_pct', 'older_malay_pct', 'youth_pct', 'median_age']

    if eth_path.exists():
        eth_df = pd.read_csv(eth_path)
        eth_df['seat_no'] = eth_df['seat'].map(seat_no)
        df_train = df_train.merge(eth_df[['seat_no'] + eth_cols], on='seat_no', how='left')
        print(f"  Loaded ethnicity: {state} {config['ethnicity_year']}")
    else:
        print(f"  Ethnicity file missing: {eth_path}")
        print(f"     Training without demographics (degraded)")
        for col in eth_cols:
            df_train[col] = 0

    for col in eth_cols:
        if col not in df_train.columns:
            df_train[col] = 0

    # Interaction feature placeholders (sentiment/economic applied at prediction only)
    df_train['bn_sent_x_malay'] = 0
    df_train['harapan_sent_x_chinese'] = 0
    df_train['pn_sent_x_young_malay'] = 0
    df_train['tension_x_mixed'] = 0
    df_train['economic_x_youth'] = 0
    df_train['narrative_pressure'] = 0

    X_train = df_train[FEATURE_NAMES_TRAIN].fillna(0)
    y_train = df_train['target_non_bn_won']

    print(f"  X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"  Class balance: Non-BN={y_train.sum()}, BN={len(y_train)-y_train.sum()}")

    # ── Honest held-out score (time-ordered) ─────────────────────
    holdout = compute_holdout(df_train, config)
    if holdout:
        print(f"\n  HELD-OUT ({holdout['test_transition']}, trained on {holdout['train_transitions']}): "
              f"{holdout['correct']}/{holdout['seats']} = {holdout['accuracy']:.2%}  "
              f"(never-opposition {holdout['baseline_never_opposition']}, "
              f"same-as-last {holdout['baseline_same_as_last']})")

    # ── Train models ─────────────────────────────────────────────
    print(f"\nTraining models phase...")

    rf = RandomForestClassifier(
        **config['rf_params'],
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_proba = evaluate_model("Random Forest", rf, X_train, y_train)

    xgb = XGBClassifier(
        **config['xgb_params'],
        eval_metric='logloss',
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_metrics, xgb_proba = evaluate_model("XGBoost", xgb, X_train, y_train)

    # Calibration -- dynamic cv_folds to avoid crashing on small classes
    min_class_count = y_train.value_counts().min()
    cv_folds = min(3, min_class_count)

    if cv_folds < 2:
        print(f"  Smallest class has {min_class_count} sample(s) -- skipping calibration")
        rf_cal = rf
    else:
        rf_cal = CalibratedClassifierCV(
            RandomForestClassifier(
                **config['rf_params'],
                class_weight='balanced',
                random_state=42
            ),
            cv=cv_folds,
            method='isotonic'
        )
        rf_cal.fit(X_train, y_train)

    # ── Ensemble ─────────────────────────────────────────────────
    ens_proba = (rf_proba + xgb_proba) / 2
    ens_pred  = (ens_proba >= 0.5).astype(int)
    ens_acc   = accuracy_score(y_train, ens_pred)
    ens_auc   = roc_auc_score(y_train, ens_proba)
    ens_brier = brier_score_loss(y_train, ens_proba)

    print(f"\n  [Ensemble RF+XGB]")
    print(f"    Accuracy: {ens_acc:.2%}")
    print(f"    AUC-ROC:  {ens_auc:.4f}")
    print(f"    Brier:    {ens_brier:.4f}")

    # ── OOD detection ────────────────────────────────────────────
    print(f"\nOOD Detection Phase...")

    ood = EllipticEnvelope(contamination=0.1, random_state=42)
    ood.fit(X_train.values)
    ood_pred  = ood.predict(X_train.values)
    ood_count = (ood_pred == -1).sum()
    print(f"  OOD seats detected: {ood_count}/{len(X_train)} ({ood_count/len(X_train):.1%})")

    # ── Feature importance ───────────────────────────────────────
    print(f"\n  Top 10 feature importance (Random Forest):")

    rf_importance = pd.DataFrame({
        'feature': FEATURE_NAMES_TRAIN,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in rf_importance.head(10).iterrows():
        bar = '#' * int(row['importance'] * 60)
        print(f"    {row['feature']:25s}: {bar} {row['importance']:.4f}")

    # ── Save models ──────────────────────────────────────────────
    # Stage the new model OUTSIDE 'latest'; version_and_promote decides whether it replaces
    # the prior model (so the gate compares new vs prior, not new vs itself).
    out_dir = MODELS_DIR / state / '_candidate'
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'rf_model.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open(out_dir / 'xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb, f)
    with open(out_dir / 'rf_calibrated.pkl', 'wb') as f:
        pickle.dump(rf_cal, f)
    with open(out_dir / 'ood_detector.pkl', 'wb') as f:
        pickle.dump(ood, f)

    print(f"\n  Saved models to {out_dir}")

    metadata = {
        'state': state,
        'description': config['description'],
        'transitions': config['transitions'],
        'seats': len(X_train),
        'feature_version': FEATURE_VERSION,
        'training_cutoff_year': int(df_train['year_b'].max()),
        'holdout': holdout,
        'holdout_accuracy': holdout['accuracy'] if holdout else None,
        'in_sample_note': "'models' below are IN-SAMPLE (training-data) metrics; use 'holdout' for real performance",
        'features': FEATURE_NAMES_TRAIN,
        'class_balance': {
            'non_bn': int(y_train.sum()),
            'bn': int(len(y_train) - y_train.sum())
        },
        'models': {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'ensemble': {
                'accuracy': round(ens_acc, 4),
                'auc': round(ens_auc, 4),
                'brier': round(ens_brier, 4),
            }
        },
        'feature_importance': rf_importance.set_index('feature')['importance'].round(4).to_dict(),
        'ood': {
            'n_flagged': int(ood_count),
            'pct_flagged': round(ood_count / len(X_train) * 100, 1)
        }
    }

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata.json")

    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train election prediction models')
    parser.add_argument('states', nargs='*', help='States to train (default: all)')
    args = parser.parse_args()

    states_to_train = args.states if args.states else list(TRAINING_CONFIGS.keys())

    all_results = {}
    for state in states_to_train:
        if state not in TRAINING_CONFIGS:
            print(f"Unknown state: {state}")
            continue

        config = TRAINING_CONFIGS[state]
        try:
            result = train_state(state, config)
            if result:
                all_results[state] = result
                promoted = version_and_promote(state, result, candidate_dir=MODELS_DIR / state / '_candidate')
                alert_on_success(state, result['holdout_accuracy'] if result['holdout_accuracy'] is not None
                                 else result['models']['ensemble']['accuracy'], promoted)
        except Exception as e:
            alert_on_failure(state, str(e))
            print(f"Training failed for {state}: {e}")

    print(f"\n\n{'='*70}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"  {'State':<18} {'Transitions':<15} {'Seats':<8} {'Held-out':<10} {'(in-sample)':<12}")
    print(f"  {'-'*70}")

    for state, r in all_results.items():
        m = r['models']
        first_yr = TRAINING_CONFIGS[state]['transitions'][0][0]
        last_yr = TRAINING_CONFIGS[state]['transitions'][-1][1]
        trans = f'{first_yr}-{last_yr}'
        held = f"{r['holdout_accuracy']:.2%}" if r['holdout_accuracy'] is not None else 'n/a'
        print(
            f"  {state:<18} {trans:<15} {r['seats']:<8} "
            f"{held:<10} {m['ensemble']['accuracy']:>7.2%}"
        )

    print(f"\n  All models trained and saved to backend/models/")
    print(f"  Ready for deployment via StatePredictor.")


if __name__ == "__main__":
    main()