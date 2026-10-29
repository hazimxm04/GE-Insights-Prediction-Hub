#!/usr/bin/env python
"""
State election predictor with OOD fallback (binary: BN vs non-BN).
Usage: from backend.core.models.state_predictor import StatePredictor

IMPORTANT ARCHITECTURAL NOTE:
This file previously built its feature vectors using
backend/core/pipelines/state_pipeline.py's engineer_features(),
a SEPARATE, independently-written function from scripts/
train_models.py's build_transition() -- the function that actually
produced the training data for the saved models. The two computed
the SAME feature names (log_voters, turnout_change, etc.) using
DIFFERENT FORMULAS (e.g. np.log1p vs np.log, different turnout
calculations), producing different numeric values for the same
seat. This caused the OOD detector to flag every single seat as
wildly out-of-distribution (OOD scores in the tens of thousands,
vs. ~1-6 during training) -- because the numbers it was being fed
were never from the same distribution it was fit on.

FIX: this file now calls scripts/train_models.py's build_transition()
directly -- the SAME function used to build the training data -- so
prediction-time features are computed identically to training-time
features. state_pipeline.py is still used for loading raw ballots/
stats and for sentiment/economic/ethnicity data (which ARE
consistent), just not for structural feature engineering.
See docs/multiclass_analysis.md.
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

from backend.core.pipelines.state_pipeline import StateElectionPipeline
from train_models import build_transition, STATE_URL_PREFIX
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

VALID_STATES = ['johor', 'neg_sembilan', 'selangor', 'melaka', 'perak']

# Most recent transition per state: (year_from, year_to) used to
# build LIVE prediction features, matching the final transition each
# state was trained on (see training_config.py's 'transitions').
# year_to for each is the most recent HELD election (not 2026 itself,
# unless that election has actually happened -- see VALIDATED below).
LATEST_TRANSITION = {
    'johor':        (2018, 2022),
    'neg_sembilan': (2018, 2023),
    'selangor':     (2018, 2023),
    'melaka':       (2013, 2018) if True else None,  # see note below
    'perak':        (2018, 2022),
}
# NOTE on melaka: its training transitions are (2013,2018),(2018,2021).
# Its most recent HELD election is 2021 (2026 not yet held/no data).
LATEST_TRANSITION['melaka'] = (2018, 2021)

# States with a real, held 2026 election (ground truth exists)
VALIDATED_STATES = {'johor', 'neg_sembilan'}

# ── OOD confidence thresholds ─────────────────────────────────────
OOD_HIGH   = 1.0
OOD_MEDIUM = 2.5


class StatePredictor:
    """
    Predicts DUN seat outcomes for a Malaysian state (binary: BN vs non-BN).

    3-layer prediction:
      Layer 1: RF + XGB ensemble (main model, 15 structural +
               ethnicity features -- matches scripts/train_models.py
               EXACTLY, including feature engineering formulas)
      Layer 2: OOD detection (Mahalanobis distance)
      Layer 3: Historical base rate fallback (when OOD)

    For VALIDATED_STATES (Johor, Neg Sembilan), predict_all() builds
    features from the (2018->2022)/(2018->2023) transition and can
    be compared against real 2026 results via scripts/validation.py.
    For other states, this produces a genuine live forecast using
    each state's most recent held election as the baseline.

    Usage:
        predictor = StatePredictor('johor')
        result = predictor.predict_seat('N.10 Perling', features_dict)
        results = predictor.predict_all()
    """

    def __init__(self, state: str):
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

        self._ballots = None
        self._stats   = None
        self.base_rates = self._compute_base_rates()

        print(f"StatePredictor loaded: {state} ({len(self.base_rates)} seats)")

    def _load_raw_data(self):
        """Load ballots/stats once, cached, using the SAME source
        train_models.py used (electiondata.my parquet, not
        state_pipeline.py's local file expectations)."""
        if self._ballots is None:
            prefix = STATE_URL_PREFIX[self.state]
            url_ballots = f'https://lake.electiondata.my/results_headline/headline_ballots_state_{prefix}.parquet'
            url_stats   = f'https://lake.electiondata.my/results_headline/headline_stats_state_{prefix}.parquet'
            self._ballots = pd.read_parquet(url_ballots)
            self._stats   = pd.read_parquet(url_stats)
        return self._ballots, self._stats

    def _build_live_features(self) -> pd.DataFrame:
        """
        Build the current feature set using build_transition() --
        the SAME function that produced training data -- for this
        state's most recent transition. Then merges ethnicity, exactly
        as training did.
        """
        ballots, stats = self._load_raw_data()
        year_from, year_to = LATEST_TRANSITION[self.state]
        df = build_transition(ballots, stats, year_from, year_to)

        eth_path = ROOT / 'data/raw/ethnicity' / f'ethnicity_{self.state}_{self.config["ethnicity_year"]}.csv'
        eth_cols = ['malay_pct', 'chinese_pct', 'indian_pct', 'young_malay_pct',
                    'young_chinese_pct', 'older_malay_pct', 'youth_pct', 'median_age']

        if eth_path.exists():
            eth_df = pd.read_csv(eth_path)
            df = df.merge(eth_df[['seat'] + eth_cols], on='seat', how='left')
        else:
            for col in eth_cols:
                df[col] = 0.0

        for col in eth_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] =  df[col].fillna(0.0)

        sentiment_path = ROOT / 'data/processed/state_sentiment_scores.csv'
        tension = 0.0  # safe default       

        if sentiment_path.exists():
            sent_df   = pd.read_csv(sentiment_path)
            state_row = sent_df[sent_df['state'] == self.state]
            if not state_row.empty:
                tension = float(
                    state_row['racial_tension_index'].values[0]
                    if 'racial_tension_index' in state_row.columns
                    else state_row.get('racial_tension_all', pd.Series([0.0])).values[0]
                )
                print(f"  Loaded tension={tension:.4f} for {self.state}")

        #function that computes tension x mixedness per seat (for the model)
        df['mixed_ratio']     = 1 - (df['malay_pct'] - df['chinese_pct']).abs()
        df['tension_x_mixed'] = tension * df['mixed_ratio']
        df = df.drop(columns=['mixed_ratio'])
        
        return df

    # ── Historical base rate ──────────────────────────────────────

    def _compute_base_rates(self) -> dict:
        ballots, _ = self._load_raw_data()
        df = ballots.copy()

        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        winners = df[df['result'].isin(['won', 'won_uncontested'])].copy()

        bn_coalitions = {'BN', 'PERIKATAN'}
        winners['non_bn'] = (~winners['coalition'].isin(bn_coalitions)).astype(int)

        base_rates = {}
        for seat, group in winners.groupby('seat'):
            group = group.sort_values('year')
            n     = len(group)
            weights = np.array([0.5 ** (n - i - 1) for i in range(n)])
            rate    = np.average(group['non_bn'].values, weights=weights)
            base_rates[seat] = round(rate, 4)

        return base_rates

    def _get_base_rate(self, seat_name: str) -> float:
        return self.base_rates.get(seat_name, 0.35)

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