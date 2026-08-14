# predict_melaka.py - update with this content

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from pathlib import Path
from backend.core.pipelines.state_pipeline import StateElectionPipeline
from backend.core.models.state_predictor import StatePredictor

print('MELAKA 2026 PRE-ELECTION PREDICTION')
print('='*60)
print('Using 2018->2021 patterns to estimate 2026 outcomes')
print('(No 2026 data yet — this is a genuine pre-election forecast)')
print()

# Load pipeline and get 2018->2021 features
# These represent the CURRENT known state of each seat
pipeline = StateElectionPipeline('melaka')
df = pipeline.engineer_features(2018, 2021)

sentiment = pipeline.load_sentiment_features()
economic  = pipeline.load_economic_features()

df['bn_sentiment']         = sentiment['bn_sentiment']
df['harapan_sentiment']    = sentiment['harapan_sentiment']
df['pn_sentiment']         = sentiment['pn_sentiment']
df['racial_tension_index'] = sentiment['racial_tension_index']
df['economic_pressure']    = economic

# THEN the merge line stays as is:
from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features
df = merge_ethnicity_into_features(
    df_features=df,
    state='melaka',
    year_b=2021,
    sentiment=sentiment,
    economic_pressure=economic
)

# Load sentiment + economic + ethnicity
from backend.scripts.add_ethnicity_features import merge_ethnicity_into_features

sentiment = pipeline.load_sentiment_features()
economic  = pipeline.load_economic_features()

df['bn_sentiment']         = sentiment['bn_sentiment']
df['harapan_sentiment']    = sentiment['harapan_sentiment']
df['pn_sentiment']         = sentiment['pn_sentiment']
df['racial_tension_index'] = sentiment['racial_tension_index']
df['economic_pressure']    = economic


df = merge_ethnicity_into_features(
    df_features=df,
    state='melaka',
    year_b=2021,
    sentiment=sentiment,
    economic_pressure=economic
)

FEATURE_COLS = [
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

X = df[FEATURE_COLS].fillna(0)

# Get predictions from loaded predictor
predictor = StatePredictor('melaka')

results = []
for i, row in df.iterrows():
    features = {col: float(X.loc[i, col]) for col in FEATURE_COLS}
    result   = predictor.predict_seat(row['seat'], features)
    result['malay_pct']   = round(df.loc[i, 'malay_pct'], 3)
    result['chinese_pct'] = round(df.loc[i, 'chinese_pct'], 3)
    result['incumbent']   = row['winner_coalition_b']  # 2021 winner
    results.append(result)

results_df = pd.DataFrame(results)

# Summary
govt_wins = (results_df['prediction'] == 'BN').sum()
harapan_wins = (results_df['prediction'] != 'BN').sum()

print(f'Predicted government (BN/PN) wins:  {govt_wins}/28')
print(f'Predicted Harapan wins:             {harapan_wins}/28')
print()

# Per seat breakdown
print(f'{"Seat":<25} {"2021":<10} {"2026 Pred":<12} {"Prob":>6} {"Malay%":>7} {"Chinese%":>9} {"OOD"}')
print('-'*80)

for _, row in results_df.sort_values('probability', ascending=False).iterrows():
    ood = "OOD" if row['is_ood'] else ""
    print(
        f'{row["seat_name"]:<25} '
        f'{row["incumbent"]:<10} '
        f'{row["prediction"]:<12} '
        f'{row["probability"]:>6.2f} '
        f'{row["malay_pct"]:>7.3f} '
        f'{row["chinese_pct"]:>9.3f} '
        f'{ood}'
    )

# Save
out_path = Path('data/processed/melaka_2026_prediction.csv')
results_df.to_csv(out_path, index=False)
print(f'\nSaved: {out_path}')
print()
print('DISCLAIMER: This is a pre-election forecast based on')
print('2018->2021 historical patterns + current sentiment/economic data.')
print('Accuracy will be validated when Melaka 2026 results are announced.')