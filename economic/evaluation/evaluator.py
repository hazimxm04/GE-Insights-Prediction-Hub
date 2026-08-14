"""
evaluator.py
============
Evaluates trained LSTM models on test data (2025-2026).

Metrics:
  RMSE     - root mean squared error (in real price units)
  MAE      - mean absolute error (in real price units)
  MAPE     - mean absolute percentage error
  Dir acc  - directional accuracy (up/down correct %)

Also produces:
  economic_pressure_score  <- the Phase 1 feature

Usage:
    python economic/evaluation/evaluator.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from economic.models.lstm_model import EconomicLSTM, DEVICE

PROC_DIR   = Path("economic/data/processed")
MODELS_DIR = Path("economic/models/saved")
RAW_DIR    = Path("economic/data/raw")
OUT_DIR    = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────

def load_model(target: str) -> EconomicLSTM:
    model = EconomicLSTM().to(DEVICE)
    model.load_state_dict(
        torch.load(MODELS_DIR / f"lstm_{target}_best.pt",
                   map_location=DEVICE)
    )
    model.eval()
    return model

# ── Evaluate ──────────────────────────────────────────────────────

def evaluate_model(target: str) -> dict:
    print(f"\n{'='*55}")
    print(f"Evaluating: {target.upper()}")
    print(f"{'='*55}")

    # Load test data
    X_test = np.load(PROC_DIR / f"X_test_{target}.npy")
    y_test = np.load(PROC_DIR / f"y_test_{target}.npy")

    # Load scaler (to inverse transform back to real prices)
    with open(PROC_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Get predictions
    model = load_model(target)
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_pred_scaled = model(X_tensor).cpu().numpy()

    # Determine which column this target is
    target_idx = 0 if target == "klci" else 1
    n_features  = scaler.n_features_in_

    # Inverse transform: reconstruct full array, inverse, extract column
    def inverse_transform_col(values_1d, col_idx):
        dummy = np.zeros((len(values_1d), n_features))
        dummy[:, col_idx] = values_1d
        return scaler.inverse_transform(dummy)[:, col_idx]

    y_pred_real = inverse_transform_col(y_pred_scaled, target_idx)
    y_true_real = inverse_transform_col(y_test,        target_idx)

    # ── Metrics ───────────────────────────────────────────────────

    rmse = np.sqrt(np.mean((y_pred_real - y_true_real) ** 2))
    mae  = np.mean(np.abs(y_pred_real - y_true_real))
    mape = np.mean(np.abs((y_pred_real - y_true_real) / y_true_real)) * 100

    # Directional accuracy: did we predict UP or DOWN correctly?
    actual_dir    = np.sign(np.diff(y_true_real))
    predicted_dir = np.sign(np.diff(y_pred_real))
    dir_acc = np.mean(actual_dir == predicted_dir) * 100

    # Naive baseline: "tomorrow = today" (simplest possible forecast)
    naive_rmse = np.sqrt(np.mean(np.diff(y_true_real) ** 2))

    print(f"\n  Test period: 2025-2026 ({len(y_test)} samples)")
    print(f"\n  LSTM performance:")
    print(f"    RMSE:              {rmse:.4f} {get_unit(target)}")
    print(f"    MAE:               {mae:.4f} {get_unit(target)}")
    print(f"    MAPE:              {mape:.2f}%")
    print(f"    Directional acc:   {dir_acc:.1f}%")
    print(f"\n  Baseline (naive 'yesterday=today'):")
    print(f"    RMSE:              {naive_rmse:.4f} {get_unit(target)}")
    print(f"    Beats baseline:    {'YES' if rmse < naive_rmse else 'NO'}")

    return {
        'target':     target,
        'rmse':       round(rmse, 4),
        'mae':        round(mae, 4),
        'mape':       round(mape, 4),
        'dir_acc':    round(dir_acc, 2),
        'naive_rmse': round(naive_rmse, 4),
        'beats_naive': rmse < naive_rmse,
        'n_test':     len(y_test),
        'y_pred':     y_pred_real,
        'y_true':     y_true_real,
    }


def get_unit(target: str) -> str:
    return "pts" if target == "klci" else "MYR"


# ── Economic pressure score ───────────────────────────────────────

def compute_pressure_score(klci_results: dict,
                            usd_myr_results: dict) -> pd.DataFrame:
    """
    Derive economic_pressure_score from LSTM forecasts.

    Logic:
      KLCI falling   = negative economic signal (weight: 0.6)
      Ringgit weak   = negative economic signal (weight: 0.4)

      Score range: -1.0 (high stress) to +1.0 (high confidence)
      Negative score = economy under pressure = incumbents at risk
    """
    print(f"\n{'='*55}")
    print(f"Computing economic pressure score")
    print(f"{'='*55}")

    klci_pred    = klci_results['y_pred']
    klci_true    = klci_results['y_true']
    usdmyr_pred  = usd_myr_results['y_pred']
    usdmyr_true  = usd_myr_results['y_true']

    n = min(len(klci_pred), len(usdmyr_pred))

    # Percentage changes (predicted vs actual previous day)
    klci_pct    = (klci_pred[:n] - klci_true[:n]) / klci_true[:n]
    usdmyr_pct  = (usdmyr_pred[:n] - usdmyr_true[:n]) / usdmyr_true[:n]

    # Pressure score:
    #   KLCI falling (-) = pressure
    #   USD/MYR rising (+, Ringgit weakening) = pressure
    raw_score = -0.6 * klci_pct + (-0.4 * usdmyr_pct)

    # Normalize to [-1, 1] range
    max_abs = np.abs(raw_score).max()
    if max_abs > 0:
        pressure_score = raw_score / max_abs
    else:
        pressure_score = raw_score

    # Load raw data to get dates for test period
    klci_df = pd.read_csv(RAW_DIR / "klci.csv",
                          index_col="date", parse_dates=True)
    test_dates = klci_df.index[klci_df.index > "2024-12-31"][:n]

    df_pressure = pd.DataFrame({
        'date':                   test_dates,
        'klci_actual':             klci_true[:n],
        'klci_predicted':          klci_pred[:n],
        'usd_myr_actual':          usdmyr_true[:n],
        'usd_myr_predicted':       usdmyr_pred[:n],
        'klci_pct_change':         klci_pct,
        'usdmyr_pct_change':       usdmyr_pct,
        'economic_pressure_score': pressure_score,
    })

    # Summary statistics
    mean_score = pressure_score.mean()
    level = "HIGH STRESS" if mean_score < -0.3 else \
            "MODERATE"    if mean_score < 0    else \
            "POSITIVE"

    print(f"\n  Mean pressure score (2025-2026): {mean_score:.4f}")
    print(f"  Economic interpretation: {level}")
    print(f"\n  Score distribution:")
    print(f"    Negative (stress):  {(pressure_score < 0).mean():.1%} of days")
    print(f"    Positive (growth):  {(pressure_score > 0).mean():.1%} of days")

    # Save
    save_path = OUT_DIR / "economic_pressure_scores.csv"
    df_pressure.to_csv(save_path, index=False)
    print(f"\n  Saved: {save_path}")
    print(f"  This file feeds into Phase 1 as feature #11")

    return df_pressure

# Add this to evaluator.py after pressure score is computed

def compute_election_period_scores(df_pressure: pd.DataFrame) -> dict:
    """
    Aggregate pressure score for each election period.
    Phase 1 needs ONE number per election, not daily scores.
    """
    df = df_pressure.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Election periods (90 days before each election)
    election_periods = {
        'johor_2026':        ('2026-04-12', '2026-07-11'),
        'neg_sembilan_2026': ('2026-05-03', '2026-08-01'),
        'melaka_2026':       ('2026-06-01', '2026-08-31'),
    }

    scores = {}
    print(f"\n{'='*55}")
    print(f"ELECTION PERIOD ECONOMIC PRESSURE")
    print(f"{'='*55}")

    for election, (start, end) in election_periods.items():
        period = df[start:end]['economic_pressure_score']

        if len(period) == 0:
            print(f"  {election}: no data in range")
            scores[election] = 0.0
            continue

        score = period.mean()
        level = "HIGH STRESS" if score < -0.3 else \
                "MODERATE"    if score < 0    else \
                "POSITIVE"

        print(f"  {election}:")
        print(f"    Period: {start} to {end}")
        print(f"    Days:   {len(period)}")
        print(f"    Score:  {score:.4f} ({level})")
        scores[election] = round(score, 4)

    # Save election period scores
    scores_df = pd.DataFrame([
        {'state': k, 'economic_pressure_score': v}
        for k, v in scores.items()
    ])
    scores_df.to_csv(
        Path("data/processed") / "election_economic_pressure.csv",
        index=False
    )
    print(f"\n  Saved: data/processed/election_economic_pressure.csv")
    print(f"  This feeds into Phase 1 state_pipeline.py")

    return scores

# Add this to evaluator.py after pressure score is computed

def compute_election_period_scores(df_pressure: pd.DataFrame) -> dict:
    """
    Aggregate pressure score for each election period.
    Phase 1 needs ONE number per election, not daily scores.
    """
    df = df_pressure.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Election periods (90 days before each election)
    election_periods = {
        'johor_2026':        ('2026-04-12', '2026-07-11'),
        'neg_sembilan_2026': ('2026-05-03', '2026-08-01'),
        'melaka_2026':       ('2026-06-01', '2026-08-31'),
    }

    scores = {}
    print(f"\n{'='*55}")
    print(f"ELECTION PERIOD ECONOMIC PRESSURE")
    print(f"{'='*55}")

    for election, (start, end) in election_periods.items():
        period = df[start:end]['economic_pressure_score']

        if len(period) == 0:
            print(f"  {election}: no data in range")
            scores[election] = 0.0
            continue

        score = period.mean()
        level = "HIGH STRESS" if score < -0.3 else \
                "MODERATE"    if score < 0    else \
                "POSITIVE"

        print(f"  {election}:")
        print(f"    Period: {start} to {end}")
        print(f"    Days:   {len(period)}")
        print(f"    Score:  {score:.4f} ({level})")
        scores[election] = round(score, 4)

    # Save election period scores
    scores_df = pd.DataFrame([
        {'state': k, 'economic_pressure_score': v}
        for k, v in scores.items()
    ])
    scores_df.to_csv(
        Path("data/processed") / "election_economic_pressure.csv",
        index=False
    )
    print(f"\n  Saved: data/processed/election_economic_pressure.csv")
    print(f"  This feeds into Phase 1 state_pipeline.py")

    return scores

# ── Save evaluation results ───────────────────────────────────────

def save_results(results: dict):
    summary = {}
    for target, r in results.items():
        summary[target] = {k: v for k, v in r.items()
                           if k not in ('y_pred', 'y_true')}
        # Fix: convert numpy bool to Python bool
        summary[target]['beats_naive'] = bool(summary[target]['beats_naive'])

    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Evaluation results saved")
# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    results = {}
    for target in ["klci", "usd_myr"]:
        results[target] = evaluate_model(target)

    # Compute economic pressure score
    df_pressure = compute_pressure_score(
        results["klci"],
        results["usd_myr"]
    )

    # After compute_pressure_score():
    election_scores = compute_election_period_scores(df_pressure)

    save_results(results)

    # Final summary
    print(f"\n{'='*55}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*55}")
    for target, r in results.items():
        print(f"\n  {target.upper()}:")
        print(f"    RMSE:          {r['rmse']:.4f} {get_unit(target)}")
        print(f"    Directional:   {r['dir_acc']:.1f}%")
        print(f"    Beats naive:   {r['beats_naive']}")

    print(f"\n  Economic pressure score:")
    print(f"    Mean (2025-26):  "
          f"{df_pressure['economic_pressure_score'].mean():.4f}")
    print(f"    Saved to:        data/processed/economic_pressure_scores.csv")

    print(f"\nPhase 4 evaluation complete!")
    print(f"Next: connect economic_pressure_score to Phase 1")