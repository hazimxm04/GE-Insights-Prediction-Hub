"""
predict_selangor.py
====================
v2: Uses Selangor's OWN trained model (not Johor transfer)
    blended with weighted-recency historical vote share trend.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forecast_engine import (
    get_historical_ph_vote_shares, compute_forecast_score, categorise_risk
)

print("SELANGOR 2026/28 FORECAST (native model + weighted loyalty)")
print("="*60)

# ── Step 1: Load Selangor election data ──────────────────────────

print("Loading Selangor election data...")
url_ballots = 'https://lake.electiondata.my/results_headline/headline_ballots_state_sgr.parquet'
url_stats   = 'https://lake.electiondata.my/results_headline/headline_stats_state_sgr.parquet'

ballots = pd.read_parquet(url_ballots)
stats   = pd.read_parquet(url_stats)
print(f"  Ballots: {len(ballots)} rows")
print(f"  Stats:   {len(stats)} rows")

# ── Step 2: Engineer features (2018 -> 2023) ──────────────────────

print("\nEngineering features (2018 -> 2023)...")

b2018 = ballots[pd.to_datetime(ballots['date']).dt.year == 2018][
    ['seat', 'coalition', 'result']
].copy()
winners_2018 = b2018[b2018['result'] == 'won'][['seat', 'coalition']].copy()
winners_2018.columns = ['seat', 'winner_coalition_a']

b2023 = ballots[pd.to_datetime(ballots['date']).dt.year == 2023][
    ['seat', 'coalition', 'result']
].copy()
winners_2023 = b2023[b2023['result'] == 'won'][['seat', 'coalition']].copy()
winners_2023.columns = ['seat', 'winner_coalition_b']

s2018 = stats[pd.to_datetime(stats['date']).dt.year == 2018][
    ['seat', 'majority', 'votes_valid', 'voters_total', 'n_candidates']
].copy()
s2018.columns = ['seat', 'majority_a', 'votes_total_a', 'voters_total_a', 'n_candidates_a']

s2023 = stats[pd.to_datetime(stats['date']).dt.year == 2023][
    ['seat', 'majority', 'votes_valid', 'voters_total', 'n_candidates']
].copy()
s2023.columns = ['seat', 'majority_b', 'votes_total_b', 'voters_total_b', 'n_candidates_b']

df = winners_2018.merge(winners_2023, on='seat')
df = df.merge(s2018, on='seat', how='left')
df = df.merge(s2023, on='seat', how='left')

df['majority_change']      = df['majority_b'] - df['majority_a']
df['turnout_a']            = df['votes_total_a'] / df['voters_total_a']
df['turnout_b']            = df['votes_total_b'] / df['voters_total_b']
df['turnout_change']       = df['turnout_b'] - df['turnout_a']
df['incumbent_held']       = (df['winner_coalition_a'] == df['winner_coalition_b']).astype(int)
df['log_voters']           = np.log(df['voters_total_b'].fillna(df['voters_total_a']))
df['majority_perc_change'] = df['majority_change'] / df['voters_total_b'].replace(0, np.nan)
df['n_candidates_b']       = df['n_candidates_b'].fillna(3)

print(f"  Selangor seats: {len(df)}")
print(f"  Coalitions 2023: {df['winner_coalition_b'].value_counts().to_dict()}")

# ── Step 3: Weighted-recency historical vote shares ───────────────

print("\nLoading historical PH vote shares (weighted-recency)...")
df, recent_year = get_historical_ph_vote_shares(ballots, df, seat_col='seat')
print(f"  Most recent election used: {recent_year}")
print(f"  ph_share_2008 non-null: {df['ph_share_2008'].notna().sum()}/{len(df)}")
print(f"  ph_share_2013 non-null: {df['ph_share_2013'].notna().sum()}/{len(df)}")
print(f"  ph_share_2018 non-null: {df['ph_share_2018'].notna().sum()}/{len(df)}")
print(f"  ph_share_recent non-null: {df['ph_share_recent'].notna().sum()}/{len(df)}")

# ── Step 4: Sentiment + economic + ethnicity ──────────────────────

print("\nAdding sentiment + economic + ethnicity features...")

from backend.core.pipelines.state_pipeline import StateElectionPipeline

pipeline  = StateElectionPipeline('johor')
sentiment = pipeline.load_sentiment_features()
economic  = pipeline.load_economic_features()

df['bn_sentiment']         = sentiment['bn_sentiment']
df['harapan_sentiment']    = sentiment['harapan_sentiment']
df['pn_sentiment']         = sentiment['pn_sentiment']
df['racial_tension_index'] = sentiment['racial_tension_index']
df['economic_pressure']    = economic

eth_path = ROOT / 'data/raw/ethnicity/ethnicity_selangor_2023.csv'
eth_df   = pd.read_csv(eth_path)
df = df.merge(eth_df[[
    'seat', 'malay_pct', 'chinese_pct', 'indian_pct',
    'young_malay_pct', 'young_chinese_pct', 'older_malay_pct',
    'youth_pct', 'median_age'
]], on='seat', how='left')

df['bn_sent_x_malay']        = df['bn_sentiment']      * df['malay_pct']
df['harapan_sent_x_chinese'] = df['harapan_sentiment'] * df['chinese_pct']
df['pn_sent_x_young_malay']  = df['pn_sentiment']      * df['young_malay_pct']
df['tension_x_mixed']        = df['racial_tension_index'] * (
    1 - abs(df['malay_pct'] - df['chinese_pct'])
)
df['economic_x_youth']       = df['economic_pressure'] * df['youth_pct']

nat_path = ROOT / 'data/processed/national_narrative_scores.csv'
nat = pd.read_csv(nat_path).iloc[0] if nat_path.exists() else {}
islam_threat = float(nat.get('islam_threat', 0.097))
malay_unity  = float(nat.get('malay_unity', 0.364))
cost_living  = float(nat.get('cost_living', 0.076))
corruption   = float(nat.get('corruption', 0.070))

df['narrative_pressure'] = (
    islam_threat * df['malay_pct'] * -0.5 +
    malay_unity  * df['malay_pct'] *  0.3 +
    cost_living  * (df['chinese_pct'] + df['youth_pct']) * -0.4 +
    corruption   * -0.3
)

print(f"  Features ready for {len(df)} seats")

# ── Step 5: Native Selangor model prediction ───────────────────────

print("\nApplying SELANGOR'S OWN trained model...")

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

models_dir = ROOT / 'backend/models/selangor'
rf  = pickle.load(open(models_dir / 'rf_model.pkl',  'rb'))
xgb = pickle.load(open(models_dir / 'xgb_model.pkl', 'rb'))

df = df.reset_index(drop=True)
X  = df[FEATURE_NAMES].fillna(0)

rf_probs  = rf.predict_proba(X)[:, 1]
xgb_probs = xgb.predict_proba(X)[:, 1]
native_ensemble = (rf_probs + xgb_probs) / 2

# ── Step 6: Combine via forecast_engine (weighted-recency) ────────

print("\nCombining native model + weighted-recency loyalty...")

results = []
for idx, row in df.iterrows():
    native_prob = float(native_ensemble[idx])
    final_score, components = compute_forecast_score(
        row, native_prob, malay_unity=malay_unity, blend_weight=0.5
    )

    results.append({
        'seat_name':          row['seat'],
        'winner_2023':        row['winner_coalition_b'],
        'ph_share_2008':      row.get('ph_share_2008'),
        'ph_share_2013':      row.get('ph_share_2013'),
        'ph_share_2018':      row.get('ph_share_2018'),
        'ph_share_recent':    row.get('ph_share_recent'),
        'weighted_avg_share': components['weighted_avg_share'],
        'trend':              components['trend'],
        'native_model_prob':  components['johor_model_prob'],  # var name reused
        'loyalty_score':      components['loyalty_score'],
        'volatility':         components['volatility'],
        'harapan_holds_probability': final_score,
        'is_ood':             components['is_ood_proxy'],
        'malay_pct':          round(float(row['malay_pct']),   3),
        'chinese_pct':        round(float(row['chinese_pct']), 3),
        'indian_pct':         round(float(row.get('indian_pct', 0)), 3),
    })

results_df = pd.DataFrame(results)
results_df['vulnerability'] = results_df['harapan_holds_probability'].apply(categorise_risk)

# ── Step 7: Print results ──────────────────────────────────────────

safe        = (results_df['harapan_holds_probability'] >= 0.70).sum()
likely      = ((results_df['harapan_holds_probability'] >= 0.55) &
               (results_df['harapan_holds_probability'] < 0.70)).sum()
swing       = ((results_df['harapan_holds_probability'] >= 0.45) &
               (results_df['harapan_holds_probability'] < 0.55)).sum()
medium      = ((results_df['harapan_holds_probability'] >= 0.30) &
               (results_df['harapan_holds_probability'] < 0.45)).sum()
high        = (results_df['harapan_holds_probability'] < 0.30).sum()

print()
print("SELANGOR FORECAST SUMMARY (native model)")
print("="*60)
print(f"🟢 SAFE:          {safe} seats")
print(f"🟢 LEANS HARAPAN: {likely} seats")
print(f"🟡 SWING SEAT:     {swing} seats")
print(f"🟠 LEANS OPPOSITION:    {medium} seats")
print(f"🔴 SAFE OPPOSITION:      {high} seats")
print()

for _, row in results_df.sort_values('harapan_holds_probability').iterrows():
    trend_arrow = "↑" if row['trend'] > 0.02 else ("↓" if row['trend'] < -0.02 else "→")
    print(f"{row['seat_name']:<26} "
          f"Trend={trend_arrow}{row['trend']:+.2f} "
          f"WavgShare={row['weighted_avg_share']:.2f} "
          f"Final={row['harapan_holds_probability']:.2f} "
          f"{row['vulnerability']}")

out_path = ROOT / 'data/processed/selangor_2026_bluwave.csv'
results_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print()
print("METHODOLOGY: 50% Selangor's OWN trained ML model (RF+XGB,")
print("trained on Selangor's 2018->2023 data, NOT Johor transfer)")
print("blended with 50% weighted-recency historical vote share")
print("trend (2008=10%, 2013=20%, 2018=30%, 2023=40% weight).")