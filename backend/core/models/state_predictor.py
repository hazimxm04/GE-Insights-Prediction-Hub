#!/usr/bin/env python
"""
State election predictor with OOD fallback (binary: opposition vs everyone else).
Usage: from backend.core.models.state_predictor import StatePredictor

HOW THE FORECAST IS BUILT (leak-free)
  * Features come from backend/core/features.py -- the SAME code that built the
    training data -- using only election A (the latest held election) and the one
    before it. Election B (the one being forecast) is never needed.
  * The historical base rate uses only elections up to and including A.
  * `as_of=<year>` pretends A is that year, to reproduce a backtest. For example
    StatePredictor('johor', as_of=2022) forecasts Johor 2026 as it could have been
    forecast in 2022, and can be compared with the real result.
  * The training data has tension_x_mixed fixed at 0 (sentiment was not trained
    on), so it is fixed at 0 here too, unless a caller overrides it.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.core.features import build_features, seat_base_rates, seat_no, MIN_YEAR
from backend.core.states import STATES
from state_data import load_state_data
from training_config import TRAINING_CONFIGS

MODELS_DIR = ROOT / "backend" / "models"

# Matches scripts/training_config.py's FEATURE_NAMES_TRAIN exactly --
# this is what the saved models were actually trained on.
FEATURE_NAMES = [
    'majority_change', 'turnout_change', 'incumbent_held',
    'log_voters', 'majority_perc_change', 'n_candidates_b',
    'malay_pct', 'chinese_pct', 'indian_pct',
    'young_malay_pct', 'young_chinese_pct',
    'older_malay_pct', 'youth_pct', 'median_age',
    'tension_x_mixed',
]

VALID_STATES = list(STATES)

# States whose 2026 election has been held (real results exist to compare with).
VALIDATED_STATES = {k for k, v in STATES.items() if v['final_test']}

# ── OOD confidence thresholds ─────────────────────────────────────
OOD_HIGH   = 1.0
OOD_MEDIUM = 2.5


class StatePredictor:
    """
    Forecasts DUN seats for a Malaysian state (binary: opposition wins or not).

    3-layer prediction:
      Layer 1: RF + XGB ensemble (15 structural + ethnicity features, built by
               backend/core/features.py exactly as in training)
      Layer 2: OOD detection (Mahalanobis distance)
      Layer 3: Historical base rate blended in when a seat is OOD

    Usage:
        predictor = StatePredictor('selangor')              # forecast the next election
        predictor = StatePredictor('johor', as_of=2022)     # backtest of Johor 2026
        results = predictor.predict_all()
        result = predictor.predict_seat('N.10 Perling', {'turnout_change': 0.05})
    """

    def __init__(self, state: str, as_of: int = None):
        state = state.lower()
        if state not in VALID_STATES:
            raise ValueError(f"Invalid state '{state}'. Choose: {VALID_STATES}")

        self.state     = state
        self.model_dir = MODELS_DIR / state
        self.config    = TRAINING_CONFIGS[state]

        self.rf      = pickle.load(open(self.model_dir / "rf_model.pkl",      "rb"))
        self.xgb     = pickle.load(open(self.model_dir / "xgb_model.pkl",     "rb"))
        self.rf_cal  = pickle.load(open(self.model_dir / "rf_calibrated.pkl", "rb"))
        self.ood     = pickle.load(open(self.model_dir / "ood_detector.pkl",  "rb"))

        self._ballots, self._stats = load_state_data(state)
        years = pd.to_datetime(self._stats['date']).dt.year
        # A = the latest election we are allowed to know about
        self.year_a = int(as_of) if as_of is not None else int(years.max())
        if self.year_a not in set(years):
            raise ValueError(f"No {state} election in {self.year_a}; elections: {sorted(set(years))}")

        self.base_rates, self.default_base_rate = seat_base_rates(self._ballots, self.year_a)
        print(f"StatePredictor loaded: {state} (forecasting the election after {self.year_a}; "
              f"{len(self.base_rates)} seats)")

    def _build_live_features(self) -> pd.DataFrame:
        """Features for the election after A, from backend/core/features.py
        (the same code as training), plus ethnicity matched on seat number."""
        df = build_features(self._ballots, self._stats, self.year_a)

        eth_path = ROOT / 'data/raw/ethnicity' / f'ethnicity_{self.state}_{self.config["ethnicity_year"]}.csv'
        eth_cols = ['malay_pct', 'chinese_pct', 'indian_pct', 'young_malay_pct',
                    'young_chinese_pct', 'older_malay_pct', 'youth_pct', 'median_age']

        if eth_path.exists():
            eth_df = pd.read_csv(eth_path)
            eth_df['seat_no'] = eth_df['seat'].map(seat_no)
            df = df.merge(eth_df[['seat_no'] + eth_cols], on='seat_no', how='left')

        for col in eth_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)

        # Trained as a constant 0 (see train_models.py); feeding a real value would put the
        # input outside what the models and the OOD detector saw in training.
        df['tension_x_mixed'] = 0.0
        return df

    # ── Historical base rate (only elections up to A) ─────────────

    def _get_base_rate(self, seat_name: str) -> float:
        return self.base_rates.get(seat_no(seat_name), self.default_base_rate)

    # ── OOD score ────────────────────────────────────────────────

    def _ood_score(self, X: np.ndarray):
        return -self.ood.score_samples(X)

    def _confidence_label(self, score: float, is_ood: bool) -> str:
        if is_ood:
            return "LOW"
        elif score < OOD_HIGH:
            return "HIGH"
        elif score < OOD_MEDIUM:
            return "MEDIUM"
        else:
            return "LOW"

    # ── Core prediction ───────────────────────────────────────────

    def predict_seat_from_row(self, seat_name: str, feature_row: dict) -> dict:
        """
        Predict outcome for a single seat given a fully-built
        feature dict (matching FEATURE_NAMES exactly, computed the
        same way as training).
        """
        X = pd.DataFrame([[feature_row.get(f, 0.0) for f in FEATURE_NAMES]],
                        columns=FEATURE_NAMES)

        rf_prob       = float(self.rf.predict_proba(X)[0][1])
        xgb_prob      = float(self.xgb.predict_proba(X)[0][1])
        ensemble_prob = (rf_prob + xgb_prob) / 2
        cal_prob      = float(self.rf_cal.predict_proba(X)[0][1])

        score  = float(self._ood_score(X)[0])
        is_ood = bool(self.ood.predict(X)[0] == -1)

        base_rate     = self._get_base_rate(seat_name)
        fallback_used = False
        warning       = None
        model_used    = "ensemble"

        if is_ood:
            alpha         = min((score - 1.0) / 5.0, 1.0)
            alpha         = max(alpha, 0.0)
            final_prob    = (1 - alpha) * ensemble_prob + alpha * base_rate
            fallback_used = True
            model_used    = f"ensemble+base_rate(alpha={alpha:.2f})"
            warning       = (
                f"Seat outside training distribution "
                f"(OOD score {score:.2f}). "
                f"Blended model ({1-alpha:.0%}) with "
                f"historical base rate {base_rate:.0%} ({alpha:.0%})."
            )
        else:
            final_prob = cal_prob
            model_used = "calibrated_rf"

        prediction = "non-BN" if final_prob >= 0.5 else "BN"
        confidence = self._confidence_label(score, is_ood)

        return {
            'seat_name':       seat_name,
            'prediction':      prediction,
            'probability':     round(final_prob, 4),
            'rf_prob':         round(rf_prob, 4),
            'xgb_prob':        round(xgb_prob, 4),
            'ensemble_prob':   round(ensemble_prob, 4),
            'calibrated_prob': round(cal_prob, 4),
            'base_rate':       round(base_rate, 4),
            'is_ood':          is_ood,
            'ood_score':       round(score, 4),
            'confidence':      confidence,
            'fallback_used':   fallback_used,
            'model_used':      model_used,
            'warning':         warning,
        }

    def predict_seat(self, seat_name: str, features: dict) -> dict:
        """
        Predict a single seat, merging user-supplied feature
        overrides with this seat's real, currently-computed baseline
        (same build_transition()-based pipeline as predict_all()).
        Used by the chatbot's 'what if' scenario tool-calling.
        """
        all_features = self._build_live_features()
        row = all_features[all_features['seat'] == seat_name]
        if row.empty:
            row = all_features[all_features['seat_no'] == seat_no(seat_name)]

        baseline = {f: float(row.iloc[0].get(f, 0.0)) for f in FEATURE_NAMES} if not row.empty else {f: 0.0 for f in FEATURE_NAMES}
        merged = {**baseline, **features}

        return self.predict_seat_from_row(seat_name, merged)

    def predict_all(self):
        """Predict every seat for this state's most recent transition."""
        df = self._build_live_features()
        if df.empty:
            return pd.DataFrame()

        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0

        X = df[FEATURE_NAMES].fillna(0)

        results = []
        for i, row in df.iterrows():
            feature_row = {f: float(X.loc[i, f]) for f in FEATURE_NAMES}
            result = self.predict_seat_from_row(row['seat'], feature_row)
            results.append(result)

        return pd.DataFrame(results)


if __name__ == "__main__":
    for state in VALID_STATES:
        print(f"\n{'='*60}")
        print(f"  TESTING: {state.upper()}")
        print(f"{'='*60}")

        try:
            predictor = StatePredictor(state)
            df = predictor.predict_all()

            if not df.empty:
                print(f"\n  Sample predictions (first 5 seats):")
                cols = ['seat_name', 'prediction', 'probability',
                        'confidence', 'is_ood', 'fallback_used']
                print(df[cols].head().to_string(index=False))
                print(f"\n  OOD rate: {df['is_ood'].mean():.1%} "
                      f"({df['is_ood'].sum()}/{len(df)} seats)")
            else:
                print("  (No predictions -- data not yet available)")
        except Exception as e:
            print(f"  Error: {e}")