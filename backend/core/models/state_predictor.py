#!/usr/bin/env python
"""
State election predictor with OOD fallback.
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

VALID_STATES = ['johor', 'neg_sembilan', 'melaka']

# ── OOD confidence thresholds ─────────────────────────────────────
OOD_HIGH   = 1.0   # score < 1.0  → HIGH confidence
OOD_MEDIUM = 2.5   # score < 2.5  → MEDIUM confidence
                   # score >= 2.5 → LOW confidence (OOD)


class StatePredictor:
    """
    Predicts DUN seat outcomes for a Malaysian state.

    3-layer prediction:
      Layer 1: RF + XGB ensemble (main model)
      Layer 2: OOD detection (Mahalanobis distance)
      Layer 3: Historical base rate fallback (when OOD)
    
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

        # Load trained models
        self.rf      = pickle.load(open(self.model_dir / "rf_model.pkl",      "rb"))
        self.xgb     = pickle.load(open(self.model_dir / "xgb_model.pkl",     "rb"))
        self.rf_cal  = pickle.load(open(self.model_dir / "rf_cal_model.pkl",  "rb"))
        self.ood     = pickle.load(open(self.model_dir / "ood_detector.pkl",  "rb"))

        # Load historical base rates (fallback for OOD seats)
        self.base_rates = self._compute_base_rates()

        print(f"✅ StatePredictor loaded: {state} ({len(self.base_rates)} seats)")

    # ── Historical base rate ──────────────────────────────────────

    def _compute_base_rates(self) -> dict:
        """
        Compute per-seat historical win rate for non-BN
        across ALL elections 1959-2026.

        Weights recent elections more (exponential decay).
        Used as fallback when seat is OOD.
        """
        pipeline = StateElectionPipeline(self.state)
        df       = pipeline.ballots.copy()

        # Parse year
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

        # Get winners only
        winners = df[df['result'].isin(['won', 'won_uncontested'])].copy()

        # Map coalition to BN / non-BN
        bn_coalitions = {'BN', 'PERIKATAN'}
        winners['non_bn'] = (~winners['coalition'].isin(bn_coalitions)).astype(int)

        # Per-seat base rate (exponential decay — recent elections weighted more)
        base_rates = {}
        for seat, group in winners.groupby('seat'):
            group = group.sort_values('year')
            n     = len(group)
            # Weight: 0.5^(n-i-1) → most recent election weight = 1.0
            weights = np.array([0.5 ** (n - i - 1) for i in range(n)])
            rate    = np.average(group['non_bn'].values, weights=weights)
            base_rates[seat] = round(rate, 4)

        return base_rates

    def _get_base_rate(self, seat_name: str) -> float:
        """Get historical base rate for a seat. Default 0.35 if unknown."""
        return self.base_rates.get(seat_name, 0.35)

    # ── OOD score ────────────────────────────────────────────────

    def _ood_score(self, X: np.ndarray):
        """
        Compute Mahalanobis distance.
        Higher = more out-of-distribution.
        """
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

        Args:
            seat_name: e.g. "N.10 Perling"
            features:  dict with keys matching FEATURES list

        Returns dict:
            seat_name        → seat name
            prediction       → "non-BN" or "BN"
            probability      → final probability (0-1)
            rf_prob          → raw RF probability
            xgb_prob         → raw XGB probability
            ensemble_prob    → average of RF + XGB
            calibrated_prob  → calibrated RF probability
            base_rate        → historical non-BN win rate
            is_ood           → True if regime shift detected
            ood_score        → Mahalanobis distance
            confidence       → "HIGH", "MEDIUM", "LOW"
            fallback_used    → True if base rate blended in
            model_used       → which model drove the prediction
            warning          → human-readable explanation
        """
        # Build feature vector
        import pandas as pd
        X = pd.DataFrame([[features.get(f, 0.0) for f in FEATURE_NAMES]], 
                        columns=FEATURE_NAMES)

        # Layer 1: Model predictions
        rf_prob       = float(self.rf.predict_proba(X)[0][1])
        xgb_prob      = float(self.xgb.predict_proba(X)[0][1])
        ensemble_prob = (rf_prob + xgb_prob) / 2
        cal_prob      = float(self.rf_cal.predict_proba(X)[0][1])

        # Layer 2: OOD detection
        score  = float(self._ood_score(X)[0])
        is_ood = bool(self.ood.predict(X)[0] == -1)

        # Layer 3: OOD fallback — blend with historical base rate
        base_rate     = self._get_base_rate(seat_name)
        fallback_used = False
        warning       = None
        model_used    = "ensemble"

        if is_ood:
            # Alpha = how much to trust base rate
            # OOD score 2.5 → alpha 0.3 (slight blend)
            # OOD score 4.0 → alpha 0.6 (strong blend)
            # OOD score 6.0 → alpha 1.0 (full fallback)
            alpha         = min((score - 1.0) / 5.0, 1.0)
            alpha         = max(alpha, 0.0)
            final_prob    = (1 - alpha) * ensemble_prob + alpha * base_rate
            fallback_used = True
            model_used    = f"ensemble+base_rate(α={alpha:.2f})"
            warning       = (
                f"Seat outside training distribution "
                f"(OOD score {score:.2f}). "
                f"Blended model ({1-alpha:.0%}) with "
                f"historical base rate {base_rate:.0%} ({alpha:.0%})."
            )
        else:
            # In-distribution: use calibrated model
            final_prob = cal_prob
            model_used = "calibrated_rf"

        # Final prediction
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

        # Add sentiment features
        sentiment = pipeline.load_sentiment_features()
        df['bn_sentiment']         = sentiment['bn_sentiment']
        df['harapan_sentiment']    = sentiment['harapan_sentiment']
        df['pn_sentiment']         = sentiment['pn_sentiment']
        df['racial_tension_index'] = sentiment['racial_tension_index']

        # Add economic feature
        economic_pressure = pipeline.load_economic_features()
        df['economic_pressure'] = economic_pressure

        # Add ethnicity + interactions
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


# ── Quick test ────────────────────────────────────────────────────

if __name__ == "__main__":
    for state in ['johor', 'neg_sembilan', 'melaka']:
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

                # Show fallback seats
                fallback = df[df['fallback_used']]
                if len(fallback) > 0:
                    print(f"\n  Seats using OOD fallback ({len(fallback)}):")
                    for _, r in fallback.head(5).iterrows():
                        print(f"    {r['seat_name']:<35} "
                              f"P={r['probability']:.2f} "
                              f"base={r['base_rate']:.2f} "
                              f"OOD={r['ood_score']:.2f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")