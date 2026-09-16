#!/usr/bin/env python
"""
State election predictor with OOD fallback (binary: BN vs non-BN).
Usage: from backend.core.models.state_predictor import StatePredictor
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

from backend.core.pipelines.state_pipeline import StateElectionPipeline

MODELS_DIR = ROOT / "backend" / "models"

FEATURE_NAMES = [
    'majority_change', 'turnout_change', 'incumbent_held',
    'log_voters', 'majority_perc_change', 'n_candidates_b',
    'bn_sentiment', 'harapan_sentiment', 'pn_sentiment',
    'racial_tension_index', 'economic_pressure',
    'malay_pct', 'chinese_pct', 'indian_pct',
    'young_malay_pct', 'young_chinese_pct',
    'older_malay_pct', 'youth_pct', 'median_age',
    'bn_sent_x_malay', 'harapan_sent_x_chinese',
    'pn_sent_x_young_malay', 'tension_x_mixed',
    'economic_x_youth', 'narrative_pressure',
]

VALID_STATES = ['johor', 'neg_sembilan', 'selangor', 'melaka', 'perak']

# ── OOD confidence thresholds ─────────────────────────────────────
OOD_HIGH   = 1.0   # score < 1.0  -> HIGH confidence
OOD_MEDIUM = 2.5   # score < 2.5  -> MEDIUM confidence
                   # score >= 2.5 -> LOW confidence (OOD)


class StatePredictor:
    """
    Predicts DUN seat outcomes for a Malaysian state (binary: BN vs non-BN).

    3-layer prediction:
      Layer 1: RF + XGB ensemble (main model)
      Layer 2: OOD detection (Mahalanobis distance)
      Layer 3: Historical base rate fallback (when OOD)

    Note on scope: this predicts Pakatan vs establishment (BN or PN
    combined as "non-Pakatan"). A 3-class BN/Pakatan/PN split was
    explored and found not yet reliably learnable -- PN's electoral
    history (since 2020) is too short relative to the training window
    to separate it from BN. See docs/multiclass_analysis.md.

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

        self.rf      = pickle.load(open(self.model_dir / "rf_model.pkl",      "rb"))
        self.xgb     = pickle.load(open(self.model_dir / "xgb_model.pkl",     "rb"))
        self.rf_cal  = pickle.load(open(self.model_dir / "rf_calibrated.pkl", "rb"))
        self.ood     = pickle.load(open(self.model_dir / "ood_detector.pkl",  "rb"))

        self.base_rates = self._compute_base_rates()

        print(f"StatePredictor loaded: {state} ({len(self.base_rates)} seats)")

    def get_seat_context(self, seat_name: str) -> dict:
        """
        Fetch real sentiment/ethnicity/economic data for a seat.
        Used to auto-fill features not provided by the caller.
        """
        from backend.core.pipelines.state_pipeline import StateElectionPipeline
        from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

        pipeline = StateElectionPipeline(self.state)
        sentiment = pipeline.load_sentiment_features()
        economic_pressure = pipeline.load_economic_features()

        df = pd.DataFrame([{'seat': seat_name}])
        df['bn_sentiment']         = sentiment['bn_sentiment']
        df['harapan_sentiment']    = sentiment['harapan_sentiment']
        df['pn_sentiment']         = sentiment['pn_sentiment']
        df['racial_tension_index'] = sentiment['racial_tension_index']
        df['economic_pressure']    = economic_pressure

        year_to = pipeline.config.get('val_year') or pipeline.config['test_year']
        df = merge_ethnicity_into_features(
            df_features=df, state=self.state, year_b=year_to,
            sentiment=sentiment, economic_pressure=economic_pressure
        )

        context_features = [f for f in FEATURE_NAMES if f not in [
            'majority_change', 'turnout_change', 'incumbent_held',
            'log_voters', 'majority_perc_change', 'n_candidates_b'
        ]]
        return {f: float(df.iloc[0].get(f, 0.0)) for f in context_features}

    # ── Historical base rate ──────────────────────────────────────

    def _compute_base_rates(self) -> dict:
        """
        Compute per-seat historical win rate for non-BN across all
        elections on record. Weights recent elections more heavily.
        Used as fallback when a seat is OOD.
        """
        pipeline = StateElectionPipeline(self.state)
        df       = pipeline.ballots.copy()

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

    def predict_seat(self, seat_name: str, features: dict) -> dict:
        """
        Predict outcome for a single DUN seat.

        Returns dict:
            seat_name        -> seat name
            prediction       -> "non-BN" or "BN"
            probability      -> final probability (0-1)
            rf_prob          -> raw RF probability
            xgb_prob         -> raw XGB probability
            ensemble_prob    -> average of RF + XGB
            calibrated_prob  -> calibrated RF probability
            base_rate        -> historical non-BN win rate
            is_ood           -> True if regime shift detected
            ood_score        -> Mahalanobis distance
            confidence       -> "HIGH", "MEDIUM", "LOW"
            fallback_used    -> True if base rate blended in
            model_used       -> which model drove the prediction
            warning          -> human-readable explanation
        """
        context = self.get_seat_context(seat_name)
        full_features = {**context, **features}
        X = pd.DataFrame([[full_features.get(f, 0.0) for f in FEATURE_NAMES]],
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

    def predict_all(self, year_from: int = None, year_to: int = None):
        from backend.core.pipelines.state_pipeline import StateElectionPipeline
        from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

        pipeline = StateElectionPipeline(self.state)

        year_from = year_from or pipeline.config['test_year']
        year_to   = year_to   or pipeline.config['val_year']

        if year_to is None:
            return pd.DataFrame()

        df = pipeline.engineer_features(year_from, year_to)

        sentiment = pipeline.load_sentiment_features()
        df['bn_sentiment']         = sentiment['bn_sentiment']
        df['harapan_sentiment']    = sentiment['harapan_sentiment']
        df['pn_sentiment']         = sentiment['pn_sentiment']
        df['racial_tension_index'] = sentiment['racial_tension_index']

        economic_pressure = pipeline.load_economic_features()
        df['economic_pressure'] = economic_pressure

        df = merge_ethnicity_into_features(
            df_features=df,
            state=self.state,
            year_b=year_to,
            sentiment=sentiment,
            economic_pressure=economic_pressure
        )

        X = df[FEATURE_NAMES].fillna(0)

        results = []
        for i, row in df.iterrows():
            features = {f: float(X.loc[i, f]) for f in FEATURE_NAMES}
            result   = self.predict_seat(row['seat'], features)
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

                fallback = df[df['fallback_used']]
                if len(fallback) > 0:
                    print(f"\n  Seats using OOD fallback ({len(fallback)}):")
                    for _, r in fallback.head(5).iterrows():
                        print(f"    {r['seat_name']:<35} "
                              f"P={r['probability']:.2f} "
                              f"base={r['base_rate']:.2f} "
                              f"OOD={r['ood_score']:.2f}")
        except Exception as e:
            print(f"  Error: {e}")