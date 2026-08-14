"""
preprocessor.py
===============
Prepares economic data for LSTM training.

Key steps:
  1. Load KLCI + USD/MYR CSVs
  2. Align on common dates (inner join)
  3. Normalize using MinMaxScaler
  4. Create sliding window sequences:
     X: last SEQ_LEN days of features
     y: next day's value (regression target)
  5. Split: train (2010-2022) / val (2022-2024) / test (2024-2026)
  6. Save processed arrays as .npy files

Usage:
    python economic/data/preprocessor.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

RAW_DIR  = Path("economic/data/raw")
PROC_DIR = Path("economic/data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────
SEQ_LEN    = 60    # look back 60 trading days (~3 months)
PRED_LEN   = 1     # predict 1 day ahead
TRAIN_END  = "2022-12-31"
VAL_END    = "2024-12-31"
# test = 2025-2026 (most recent, closest to election period)

# ── Load data ─────────────────────────────────────────────────────

def load_and_align() -> pd.DataFrame:
    """Load KLCI + USD/MYR and align on common trading dates"""

    klci    = pd.read_csv(RAW_DIR / "klci.csv",    index_col="date", parse_dates=True)
    usd_myr = pd.read_csv(RAW_DIR / "usd_myr.csv", index_col="date", parse_dates=True)

    klci.columns    = ["klci"]
    usd_myr.columns = ["usd_myr"]

    # Inner join: only keep dates where BOTH have data
    df = klci.join(usd_myr, how="inner")
    df = df.sort_index()
    df = df.dropna()

    print(f"Combined dataset:")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Features: {df.columns.tolist()}")
    print(f"  KLCI range: {df['klci'].min():.1f} to {df['klci'].max():.1f}")
    print(f"  USD/MYR range: {df['usd_myr'].min():.4f} to {df['usd_myr'].max():.4f}")

    return df


# ── Normalize ─────────────────────────────────────────────────────

def normalize(df: pd.DataFrame) -> tuple:
    """
    MinMax scale each feature independently to [0, 1].

    WHY normalize for LSTM?
      LSTM uses sigmoid/tanh activations that saturate outside [0,1].
      Raw KLCI values (~1000-1800) would cause vanishing/exploding gradients.
      Scaling ensures all features contribute equally regardless of magnitude.

    IMPORTANT: fit scaler on TRAINING DATA ONLY.
      Fitting on all data = data leakage (test data influences scaling).
    """
    train_mask = df.index <= TRAIN_END

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[train_mask])  # ← fit on TRAINING only

    df_scaled = pd.DataFrame(
        scaler.transform(df),
        index=df.index,
        columns=df.columns
    )

    # Save scaler for inverse transform later (get real prices back)
    with open(PROC_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nNormalization:")
    print(f"  Scaler fit on training data only (up to {TRAIN_END})")
    print(f"  Scaled range: [{df_scaled.min().min():.3f}, {df_scaled.max().max():.3f}]")

    return df_scaled, scaler


# ── Create sequences ──────────────────────────────────────────────

def create_sequences(df_scaled: pd.DataFrame, target_col: str = "klci"):
    """
    Convert time series into (X, y) pairs for LSTM.

    WHAT IS A SLIDING WINDOW?
      Instead of treating each day independently,
      LSTM needs SEQUENCES — windows of past data.

      Example with SEQ_LEN=5 (simplified):
        Day 1-5:  [1.0, 1.1, 0.9, 1.2, 1.0] → predict day 6
        Day 2-6:  [1.1, 0.9, 1.2, 1.0, 1.1] → predict day 7
        Day 3-7:  [0.9, 1.2, 1.0, 1.1, 0.8] → predict day 8
        ...

      X shape: (n_samples, SEQ_LEN, n_features)
               e.g. (3800, 60, 2) for KLCI+USDMYR with 60-day windows
      y shape: (n_samples,)
               e.g. (3800,) — next day's KLCI value

    WHY SEQ_LEN=60?
      ~3 months of trading history. Long enough to capture:
        - Monthly seasonal patterns
        - Short-term trend momentum
        - Recent political/economic events
      But not so long the model focuses on ancient history.
    """
    values = df_scaled.values
    target_idx = df_scaled.columns.tolist().index(target_col)

    X, y, dates = [], [], []

    for i in range(SEQ_LEN, len(values)):
        X.append(values[i - SEQ_LEN:i])       # 60 days of all features
        y.append(values[i, target_idx])         # next day's target value
        dates.append(df_scaled.index[i])

    X = np.array(X)   # shape: (n_samples, 60, 2)
    y = np.array(y)   # shape: (n_samples,)

    print(f"\nSequences created (target: {target_col}):")
    print(f"  X shape: {X.shape}  (samples, seq_len, features)")
    print(f"  y shape: {y.shape}  (samples,)")

    return X, y, dates


# ── Train / val / test split ──────────────────────────────────────

def split_sequences(X, y, dates):
    """
    Split sequences by DATE, not randomly.

    WHY date-based split?
      Random splitting would leak future information into training.
      "Day 500 features" might include data points whose y-values
      were already seen in training — this would be cheating.
      Date-based split ensures the model NEVER sees future data.

      This is the same principle as temporal validation in Phase 1.
    """
    dates = pd.DatetimeIndex(dates)

    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask  = dates > VAL_END

    splits = {
        "train": (X[train_mask], y[train_mask]),
        "val":   (X[val_mask],   y[val_mask]),
        "test":  (X[test_mask],  y[test_mask]),
    }

    print(f"\nTrain/val/test split:")
    for name, (Xs, ys) in splits.items():
        print(f"  {name:<6}: {len(Xs):>5} samples")

    return splits


# ── Save ──────────────────────────────────────────────────────────

def save_splits(splits: dict, target_col: str):
    """Save numpy arrays for fast loading during training"""
    for split_name, (X, y) in splits.items():
        np.save(PROC_DIR / f"X_{split_name}_{target_col}.npy", X)
        np.save(PROC_DIR / f"y_{split_name}_{target_col}.npy", y)

    print(f"\nSaved to: {PROC_DIR}")
    print(f"  Files: X_train, y_train, X_val, y_val, X_test, y_test")


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Preprocessing economic data for LSTM...\n")
    print("="*55)

    # 1. Load + align
    df = load_and_align()

    # 2. Normalize (fit on train only)
    df_scaled, scaler = normalize(df)

    # 3. Create sequences for each target
    for target in ["klci", "usd_myr"]:
        print(f"\n{'─'*55}")
        print(f"Processing target: {target.upper()}")
        X, y, dates = create_sequences(df_scaled, target_col=target)

        # 4. Split by date
        splits = split_sequences(X, y, dates)

        # 5. Save
        save_splits(splits, target)

    print(f"\n{'='*55}")
    print("Preprocessing complete!")
    print("Next: python economic/models/lstm_model.py")