"""
train_ns_weighted_backtest.py
===============================
BACKTEST: Trains Negeri Sembilan using the SAME weighted-
recency multi-transition methodology built for Selangor/
Melaka/Johor, then validates against NS's REAL 2026 results.

Training transitions: 2008->2013, 2013->2018, 2018->2023
Forecast baseline: 2023 (predicting 2026)
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from forecast_engine import (
    get_historical_ph_vote_shares, compute_forecast_score, categorise_risk
)

print("BACKTEST: NEGERI SEMBILAN (weighted-recency methodology)")
print("="*70)

print("Loading NS election data...")
url_ballots = 'https://lake.electiondata.my/results_headline/headline_ballots_state_ns.parquet'
url_stats   = 'https://lake.electiondata.my/results_headline/headline_stats_state_ns.parquet'

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


print("\nBuilding training transitions (historical, pre-2026)...")
df_2013 = build_transition(2008, 2013)
print(f"  2008->2013: {len(df_2013)} seats")
df_2018 = build_transition(2013, 2018)
print(f"  2013->2018: {len(df_2018)} seats")
df_2023 = build_transition(2018, 2023)
print(f"  2018->2023: {len(df_2023)} seats")

df_train = pd.concat([df_2013, df_2018, df_2023], ignore_index=True)
print(f"\n  Combined training set: {len(df_train)} rows")

print("\nAdding sentiment + economic + ethnicity features (training)...")

from backend.core.pipelines.state_pipeline import StateElectionPipeline
from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

pipeline  = StateElectionPipeline('johor')
sentiment = pipeline.load_sentiment_features()
economic  = pipeline.load_economic_features()

df_train['bn_sentiment']         = sentiment['bn_sentiment']
df_train['harapan_sentiment']    = sentiment['harapan_sentiment']
df_train['pn_sentiment']         = sentiment['pn_sentiment']
df_train['racial_tension_index'] = sentiment['racial_tension_index']
df_train['economic_pressure']    = economic

df_train = merge_ethnicity_into_features(
    df_features=df_train, state='neg_sembilan', year_b=2023,
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

X_train = df_train[FEATURE_NAMES].fillna(0)
y_train = df_train['target_non_bn_won']

print(f"\n  Features: {len(FEATURE_NAMES)} total, X shape: {X_train.shape}")

print("\nTraining weighted-recency backtest model...")

rf = RandomForestClassifier(
    n_estimators=200, max_depth=4, min_samples_leaf=5,
    min_samples_split=8, class_weight='balanced', random_state=42
)
rf.fit(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.03,
    reg_alpha=0.5, reg_lambda=1.5, eval_metric='logloss', random_state=42
)
xgb.fit(X_train, y_train)

print("  Model trained.")

print("\nBuilding 2026 forecast (using 2023 as 'current' baseline)...")

df_forecast = build_transition(2018, 2023)

df_forecast, recent_year = get_historical_ph_vote_shares(
    ballots, df_forecast, seat_col='seat', recent_year_cutoff=2024
)
print(f"  Most recent pre-2026 election used for trend: {recent_year}")

df_forecast['bn_sentiment']         = sentiment['bn_sentiment']
df_forecast['harapan_sentiment']    = sentiment['harapan_sentiment']
df_forecast['pn_sentiment']         = sentiment['pn_sentiment']
df_forecast['racial_tension_index'] = sentiment['racial_tension_index']
df_forecast['economic_pressure']    = economic

df_forecast = merge_ethnicity_into_features(
    df_features=df_forecast, state='neg_sembilan', year_b=2023,
    sentiment=sentiment, economic_pressure=economic
)

X_forecast = df_forecast[FEATURE_NAMES].fillna(0)
rf_probs  = rf.predict_proba(X_forecast)[:, 1]
xgb_probs = xgb.predict_proba(X_forecast)[:, 1]
native_ensemble = (rf_probs + xgb_probs) / 2

nat_path = ROOT / 'data/processed/national_narrative_scores.csv'
nat = pd.read_csv(nat_path).iloc[0] if nat_path.exists() else {}
malay_unity = float(nat.get('malay_unity', 0.364))

results = []
for idx, row in df_forecast.iterrows():
    native_prob = float(native_ensemble[idx])
    final_score, components = compute_forecast_score(
        row, native_prob, malay_unity=malay_unity, blend_weight=0.5
    )
    results.append({
        'seat': row['seat'],
        'weighted_avg_share': components['weighted_avg_share'],
        'trend': components['trend'],
        'native_model_prob': components['johor_model_prob'],
        'loyalty_score': components['loyalty_score'],
        'backtest_probability': final_score,
    })

backtest_df = pd.DataFrame(results)

print("\nComparing backtest forecast vs REAL validated 2026 results...")

actual_df = pd.read_csv(ROOT / 'backend/models/neg_sembilan/validation_2026.csv')
comparison = backtest_df.merge(
    actual_df[['seat', 'predicted_non_bn', 'actual_non_bn', 'probability']],
    on='seat', how='left'
)
comparison = comparison.rename(columns={
    'predicted_non_bn': 'original_prediction',
    'probability': 'original_probability',
})

comparison['backtest_correct'] = (
    (comparison['backtest_probability'] >= 0.5).astype(int) ==
    comparison['actual_non_bn']
)
comparison['original_correct'] = (
    comparison['original_prediction'] == comparison['actual_non_bn']
)

backtest_acc = comparison['backtest_correct'].mean()
original_acc = comparison['original_correct'].mean()

print()
print("BACKTEST RESULTS (NEGERI SEMBILAN)")
print("="*60)
print(f"Original model (single-transition):  {original_acc:.2%}")
print(f"Backtest model (weighted-recency):    {backtest_acc:.2%}")
print(f"Difference: {(backtest_acc - original_acc)*100:+.2f} percentage points")
print()

out_path = ROOT / 'data/processed/ns_backtest_comparison.csv'
comparison.to_csv(out_path, index=False)
print(f"Saved comparison: {out_path}")