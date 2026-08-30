"""
build_map_data.py
==================
One-time script: download DUN boundaries, filter to our 4 states,
merge with prediction data, save as a single GeoJSON ready for
Streamlit to render as a 270towin-style choropleth map.

Run once (or whenever predictions update):
    python build_map_data.py
"""

import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OUR_STATES = ['Johor', 'Negeri Sembilan', 'Melaka', 'Selangor', 'Perak']

print("Downloading DUN boundaries...")
url = 'https://lake.electiondata.my/maps/delimitations/peninsular_2018_dun.parquet'
r = requests.get(url)
tmp_path = ROOT / '_dun_boundaries.parquet'
with open(tmp_path, 'wb') as f:
    f.write(r.content)

gdf = gpd.read_parquet(tmp_path)
print(f"  Total seats (all Peninsular): {len(gdf)}")

gdf = gdf[gdf['state'].isin(OUR_STATES)].copy()
print(f"  Filtered to our 4 states: {len(gdf)}")
print(f"  Seats per state: {gdf['state'].value_counts().to_dict()}")

# ── Load predictions per state ─────────────────────────────────────

def load_johor_ns(state_key, state_name):
    path = ROOT / f"backend/models/{state_key}/validation_2026.csv"
    df = pd.read_csv(path)
    df['dun'] = df['seat']
    df['status'] = 'validated'
    df['prediction_label'] = df['predicted_non_bn'].apply(
        lambda x: 'Harapan' if x == 1 else 'BN/PN'
    )
    df['actual_label'] = df['actual_non_bn'].apply(
        lambda x: 'Harapan' if x == 1 else 'BN/PN'
    )
    df['display_winner'] = df['actual_label']  # show actual for validated
    df['probability'] = df['probability']
    df['correct'] = df['correct']
    return df[['dun', 'status', 'prediction_label', 'actual_label',
               'display_winner', 'probability', 'correct']]

def load_forecast(path, state_key):
    df = pd.read_csv(path)
    df['dun'] = df['seat_name']
    df['status'] = 'forecast'
    prob_col = 'probability' if 'probability' in df.columns else 'harapan_holds_probability'
    df['probability'] = df[prob_col]
    df['prediction_label'] = df['probability'].apply(
        lambda x: 'Harapan' if x >= 0.5 else 'BN/PN'
    )
    df['actual_label'] = None
    df['display_winner'] = df['prediction_label']  # show forecast for forecast states
    df['correct'] = None
    return df[['dun', 'status', 'prediction_label', 'actual_label',
               'display_winner', 'probability', 'correct']]

print("\nLoading prediction data...")

johor_df = load_johor_ns('johor', 'Johor')
ns_df    = load_johor_ns('neg_sembilan', 'Negeri Sembilan')
melaka_df = load_forecast(ROOT / 'data/processed/melaka_2026_prediction.csv', 'melaka')
selangor_df = load_forecast(ROOT / 'data/processed/selangor_2026_bluwave.csv', 'selangor')
perak_df = load_forecast(ROOT / 'data/processed/perak_2026_prediction.csv', 'perak')

all_predictions = pd.concat(
    [johor_df, ns_df, melaka_df, selangor_df, perak_df], ignore_index=True
)
print(f"  Total prediction rows: {len(all_predictions)}")

# ── Merge geometry with predictions ─────────────────────────────────

print("\nMerging boundaries with predictions...")
merged = gdf.merge(all_predictions, on='dun', how='left')

matched = merged['status'].notna().sum()
print(f"  Matched: {matched}/{len(merged)} seats")

unmatched = merged[merged['status'].isna()]
if len(unmatched) > 0:
    print(f"  Unmatched seats (no prediction data): {len(unmatched)}")
    print(unmatched[['state', 'dun']].to_string())

# ── Save as GeoJSON ─────────────────────────────────────────────────

out_path = ROOT / 'data/processed/map_data.geojson'
merged.to_file(out_path, driver='GeoJSON')
print(f"\nSaved: {out_path}")

tmp_path.unlink()  # cleanup temp file