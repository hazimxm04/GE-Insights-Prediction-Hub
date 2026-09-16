"""
train_perak_model.py
======================
Trains a Perak-SPECIFIC RF+XGB ensemble, following the same
proven methodology as Selangor/Melaka: multiple historical
transitions (2013->2018, 2018->2022) combined, with stronger
regularization to reduce small-sample overconfidence.

Perak 2022 was a genuine 3-way contest (PN=26, PH=24, BN=9),
similar structure to Melaka 2021 -- not a clean 2-way PH-vs-PN
pattern like Selangor 2023.

Target: "Did Harapan (PH/PR/Harapan) win this seat?" -- binary,
treating BN and PN both as "not-Harapan" regardless of whether
they contested together or separately.

No 2026/2027/2028 validation target exists yet for Perak's
next election -- trained but not accuracy-validated the way
Johor/NS are.
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

print("TRAINING: PERAK (native model)")
print("="*60)

print("Loading Perak election data...")
url_ballots = 'https://lake.electiondata.my/results_headline/headline_ballots_state_prk.parquet'
url_stats   = 'https://lake.electiondata.my/results_headline/headline_stats_state_prk.parquet'

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

df_2022 = build_transition(2018, 2022)
print(f"  2018->2022: {len(df_2022)} seats, Harapan wins={df_2022['target_non_bn_won'].sum()}")

df = pd.concat([df_2018, df_2022], ignore_index=True)
print(f"\n  Combined training set: {len(df)} rows")

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

# ── Ethnicity: download Perak voter roll if not already present ──

eth_path = ROOT / 'data/raw/ethnicity/ethnicity_perak_2022.csv'

if not eth_path.exists():
    print("  Perak ethnicity not found - downloading voter roll...")
    url_roll = 'https://lake.electiondata.my/voter_rolls/ge15_2022.parquet'
    filters  = [('state', '=', 'Perak')]
    vr       = pd.read_parquet(url_roll, filters=filters)
    vr['age'] = 2022 - vr['birth_year']

    eth_results = []
    for seat, group in vr.groupby('dun'):
        total         = len(group)
        malay         = group[group['ethnicity'] == 'Malay']
        chinese       = group[group['ethnicity'] == 'Chinese']
        indian        = group[group['ethnicity'] == 'Indian']
        young_malay   = malay[(malay['age'] >= 18) & (malay['age'] <= 35)]
        older_malay   = malay[malay['age'] >= 50]
        young_chinese = chinese[(chinese['age'] >= 18) & (chinese['age'] <= 35)]
        youth         = group[(group['age'] >= 18) & (group['age'] <= 29)]

        eth_results.append({
            'seat':              seat,
            'malay_pct':         round(len(malay)         / total, 4),
            'chinese_pct':       round(len(chinese)       / total, 4),
            'indian_pct':        round(len(indian)        / total, 4),
            'young_malay_pct':   round(len(young_malay)   / total, 4),
            'young_chinese_pct': round(len(young_chinese) / total, 4),
            'older_malay_pct':   round(len(older_malay)   / total, 4),
            'youth_pct':         round(len(youth)         / total, 4),
            'median_age':        round(group['age'].median(), 1),
        })

    eth_df = pd.DataFrame(eth_results)
    eth_path.parent.mkdir(parents=True, exist_ok=True)
    eth_df.to_csv(eth_path, index=False)
    print(f"  Saved ethnicity: {eth_path} ({len(eth_df)} seats)")
else:
    eth_df = pd.read_csv(eth_path)
    print(f"  Loaded ethnicity: {len(eth_df)} seats")

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

print("\nTraining models (regularized)...")

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

models_dir = ROOT / 'backend/models/perak'
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
print("NOTE: Perak 2022 was a genuine 3-way contest (PN=26,")
print("PH=24, BN=9) -- similar structure to Melaka 2021, unlike")
print("Selangor's 2-way PH-vs-PN pattern.")
print()
print("No future election has occurred yet to validate against.")