"""
train_melaka_model.py
======================
v2: Trains on TWO historical transitions (2013->2018 AND
    2018->2021), doubling training data from 28 to ~56 rows,
    with stronger regularization to reduce overfitting/
    overconfidence -- same fix applied to Selangor after
    finding the single-transition model was too extreme
    (near-0 or near-1 probabilities on genuinely competitive
    seats).

Note: Melaka 2021 was a genuine 3-way contest (BN vs PH vs
PN independently), unlike Selangor's 2-way pattern. This is
documented as a known difference in what "winning" means
across the two states' most recent elections.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.covariance import EllipticEnvelope

print("TRAINING: MELAKA (native model, v2 - more data + regularization)")
print("="*70)

print("Loading Melaka election data...")
url_ballots = 'https://lake.electiondata.my/results_headline/headline_ballots_state_mlk.parquet'
url_stats   = 'https://lake.electiondata.my/results_headline/headline_stats_state_mlk.parquet'

ballots = pd.read_parquet(url_ballots)
stats   = pd.read_parquet(url_stats)
print(f"  Ballots: {len(ballots)} rows")
print(f"  Stats:   {len(stats)} rows")


def build_transition(year_a, year_b):
    ba = ballots[pd.to_datetime(ballots['date']).dt.year == year_a][
        ['seat', 'coalition', 'result']
    ].copy()
    wa = ba[ba['result'] == 'won'][['seat', 'coalition']].copy()
    wa.columns = ['seat', 'winner_coalition_a']

    bb = ballots[pd.to_datetime(ballots['date']).dt.year == year_b][
        ['seat', 'coalition', 'result']
    ].copy()
    wb = bb[bb['result'] == 'won'][['seat', 'coalition']].copy()
    wb.columns = ['seat', 'winner_coalition_b']

    sa = stats[pd.to_datetime(stats['date']).dt.year == year_a][
        ['seat', 'majority', 'votes_valid', 'voters_total', 'n_candidates']
    ].copy()
    sa.columns = ['seat', 'majority_a', 'votes_total_a', 'voters_total_a', 'n_candidates_a']

    sb = stats[pd.to_datetime(stats['date']).dt.year == year_b][
        ['seat', 'majority', 'votes_valid', 'voters_total', 'n_candidates']
    ].copy()
    sb.columns = ['seat', 'majority_b', 'votes_total_b', 'voters_total_b', 'n_candidates_b']

    d = wa.merge(wb, on='seat')
    d = d.merge(sa, on='seat', how='left')
    d = d.merge(sb, on='seat', how='left')

    d['majority_change']      = d['majority_b'] - d['majority_a']
    d['turnout_a']            = d['votes_total_a'] / d['voters_total_a']
    d['turnout_b']            = d['votes_total_b'] / d['voters_total_b']
    d['turnout_change']       = d['turnout_b'] - d['turnout_a']
    d['incumbent_held']       = (d['winner_coalition_a'] == d['winner_coalition_b']).astype(int)
    d['log_voters']           = np.log(d['voters_total_b'].fillna(d['voters_total_a']))
    d['majority_perc_change'] = d['majority_change'] / d['voters_total_b'].replace(0, np.nan)
    d['n_candidates_b']       = d['n_candidates_b'].fillna(3)

    OPPOSITION = {'Harapan', 'PH', 'PR'}
    d['target_non_bn_won'] = d['winner_coalition_b'].isin(OPPOSITION).astype(int)

    return d


print("\nBuilding training transitions...")
df_2018 = build_transition(2013, 2018)
print(f"  2013->2018: {len(df_2018)} seats, Harapan wins={df_2018['target_non_bn_won'].sum()}")

df_2021 = build_transition(2018, 2021)
print(f"  2018->2021: {len(df_2021)} seats, Harapan wins={df_2021['target_non_bn_won'].sum()}")

df = pd.concat([df_2018, df_2021], ignore_index=True)
print(f"\n  Combined training set: {len(df)} rows")

print("\nAdding sentiment + economic + ethnicity features...")

from backend.core.pipelines.state_pipeline import StateElectionPipeline
from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

pipeline  = StateElectionPipeline('johor')
sentiment = pipeline.load_sentiment_features()
economic  = pipeline.load_economic_features()

df['bn_sentiment']         = sentiment['bn_sentiment']
df['harapan_sentiment']    = sentiment['harapan_sentiment']
df['pn_sentiment']         = sentiment['pn_sentiment']
df['racial_tension_index'] = sentiment['racial_tension_index']
df['economic_pressure']    = economic

# Ethnicity: merge_ethnicity_into_features expects a single year_b;
# use 2021 data for both transitions since that's our only ethnicity source
df = merge_ethnicity_into_features(
    df_features=df, state='melaka', year_b=2021,
    sentiment=sentiment, economic_pressure=economic
)

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

X = df[FEATURE_NAMES].fillna(0)
y = df['target_non_bn_won']

print(f"\n  Features: {len(FEATURE_NAMES)} total")
print(f"  X shape: {X.shape}")
print(f"  Class balance: Harapan={y.sum()}, Non-Harapan={len(y)-y.sum()}")

print("\nTraining models (regularized to reduce overconfidence)...")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=4, min_samples_leaf=5,
    min_samples_split=8, class_weight='balanced', random_state=42
)
rf.fit(X, y)

xgb = XGBClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.03,
    reg_alpha=0.5, reg_lambda=1.5, eval_metric='logloss', random_state=42
)
xgb.fit(X, y)

rf_cal = CalibratedClassifierCV(rf, method='isotonic', cv=3)
rf_cal.fit(X, y)

rf_probs  = rf.predict_proba(X)[:, 1]
xgb_probs = xgb.predict_proba(X)[:, 1]
ensemble_acc = (((rf_probs + xgb_probs) / 2 >= 0.5).astype(int) == y).mean()
print(f"  Ensemble training accuracy: {ensemble_acc:.2%}")

prob_range = ((rf_probs + xgb_probs) / 2)
print(f"  Probability range: {prob_range.min():.3f} to {prob_range.max():.3f}")

ood = EllipticEnvelope(contamination=0.1, random_state=42)
ood.fit(X)
ood_flagged = (ood.predict(X) == -1).sum()
print(f"  OOD seats detected: {ood_flagged}/{len(X)} ({ood_flagged/len(X):.1%})")

models_dir = ROOT / 'backend/models/melaka'
models_dir.mkdir(parents=True, exist_ok=True)

with open(models_dir / 'rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open(models_dir / 'xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open(models_dir / 'rf_calibrated.pkl', 'wb') as f:
    pickle.dump(rf_cal, f)
with open(models_dir / 'ood_detector.pkl', 'wb') as f:
    pickle.dump(ood, f)

print(f"\nSaved models to: {models_dir}")
print("Trained on 2 transitions (2013->2018, 2018->2021) with")
print("stronger regularization to reduce small-sample overconfidence.")