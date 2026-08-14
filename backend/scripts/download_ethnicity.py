"""
download_ethnicity.py
=====================
Downloads anonymised voter rolls from electiondata.my and
computes ethnicity + age composition per DUN seat.

These become SEAT-LEVEL features for Phase 1 election predictor:
  malay_pct, chinese_pct, indian_pct (overall ethnicity)
  young_malay_pct   (Malay voters aged 18-35) ← PN risk signal
  young_chinese_pct (Chinese voters aged 18-35) ← Harapan signal
  older_malay_pct   (Malay voters aged 50+) ← BN stability signal
  youth_pct         (all voters aged 18-29) ← overall volatility
  median_age        (median age of registered voters)

Usage:
    python backend/scripts/download_ethnicity.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/raw/ethnicity")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_YEAR = datetime.now().year

# ── Voter roll URLs (from lake.electiondata.my) ───────────────────

VOTER_ROLLS = {
    'johor_2022': {
        'url':   'https://lake.electiondata.my/voter_rolls/jhr_se15_2022.parquet',
        'state': 'johor',
        'year':  2022,
    },
    'johor_2026': {
        'url':   'https://lake.electiondata.my/voter_rolls/jhr_se16_2026.parquet',
        'state': 'johor',
        'year':  2026,
    },
    'ns_2023': {
        'url':   'https://lake.electiondata.my/voter_rolls/nsn_se15_2023.parquet',
        'state': 'neg_sembilan',
        'year':  2023,
    },
    'ns_2026': {
        'url':   'https://lake.electiondata.my/voter_rolls/nsn_se16_2026.parquet',
        'state': 'neg_sembilan',
        'year':  2026,
    },
}

# ── Feature computation ───────────────────────────────────────────

def compute_seat_features(df: pd.DataFrame, election_year: int) -> pd.DataFrame:
    """
    Compute ethnicity + age features per DUN seat.

    Input: raw voter roll (one row per voter)
    Output: one row per seat with 9 demographic features
    """
    df = df.copy()

    # Compute age at election year
    df['age'] = election_year - df['birth_year']

    # Clean DUN seat name (strip leading/trailing spaces)
    df['dun'] = df['dun'].str.strip()

    results = []

    for seat, group in df.groupby('dun'):
        total = len(group)
        if total == 0:
            continue

        # Ethnicity groups
        malay   = group[group['ethnicity'] == 'Malay']
        chinese = group[group['ethnicity'] == 'Chinese']
        indian  = group[group['ethnicity'] == 'Indian']

        # Age groups within Malay community
        young_malay  = malay[(malay['age'] >= 18) & (malay['age'] <= 35)]
        older_malay  = malay[malay['age'] >= 50]

        # Young Chinese
        young_chinese = chinese[(chinese['age'] >= 18) & (chinese['age'] <= 35)]

        # Overall youth (18-29, Undi18 generation)
        youth = group[(group['age'] >= 18) & (group['age'] <= 29)]

        results.append({
            'seat':               seat,
            'election_year':      election_year,
            'total_voters':       total,

            # Overall ethnicity composition
            'malay_pct':          round(len(malay)   / total, 4),
            'chinese_pct':        round(len(chinese) / total, 4),
            'indian_pct':         round(len(indian)  / total, 4),

            # Age-ethnicity interactions
            'young_malay_pct':    round(len(young_malay)   / total, 4),
            'young_chinese_pct':  round(len(young_chinese) / total, 4),
            'older_malay_pct':    round(len(older_malay)   / total, 4),

            # Overall youth signal
            'youth_pct':          round(len(youth) / total, 4),

            # Median age
            'median_age':         round(group['age'].median(), 1),
        })

    return pd.DataFrame(results)

# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading voter rolls + computing ethnicity features...")
    print("="*60)
    print("Note: Each file is ~100-200MB. May take 1-3 mins each.")

    all_features = []

    for key, config in VOTER_ROLLS.items():
        print(f"\n[{key}] Downloading from electiondata.my...")

        try:
            df = pd.read_parquet(config['url'])
            print(f"  Rows: {len(df):,}")
            print(f"  Columns: {df.columns.tolist()}")

            features = compute_seat_features(df, config['year'])
            features['state'] = config['state']

            save_path = DATA_DIR / f"ethnicity_{key}.csv"
            features.to_csv(save_path, index=False)

            print(f"  Seats processed: {len(features)}")
            print(f"  Saved: {save_path}")

            # Sample output
            print(f"\n  Sample (first 3 seats):")
            print(features[['seat', 'malay_pct', 'chinese_pct',
                            'young_malay_pct', 'youth_pct']].head(3).to_string())

            all_features.append(features)

        except Exception as e:
            print(f"  Failed: {e}")

    # Combine all into one master file
    if all_features:
        combined = pd.concat(all_features, ignore_index=True)
        master_path = DATA_DIR / "ethnicity_all.csv"
        combined.to_csv(master_path, index=False)

        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total rows: {len(combined)}")
        print(f"States: {combined['state'].unique()}")
        print(f"Years: {combined['election_year'].unique()}")
        print(f"Master file: {master_path}")

        # Show range of key features
        print(f"\nFeature ranges:")
        for col in ['malay_pct', 'chinese_pct', 'young_malay_pct',
                    'young_chinese_pct', 'youth_pct', 'median_age']:
            print(f"  {col:<22}: "
                  f"{combined[col].min():.3f} to {combined[col].max():.3f}")

        print(f"\nNext: python backend/scripts/add_ethnicity_features.py")