"""
add_ethnicity_features.py
=========================
Loads ethnicity + age features from voter rolls and
applies national narrative themes weighted by seat demographics.

KEY INSIGHT:
  National narratives ("Islam terancam", "kos sara hidup")
  affect different seats differently based on who lives there.

  Same "Islam threat" narrative:
    Malay-majority seat (malay_pct=0.75): strong PN risk signal
    Chinese-majority seat (chinese_pct=0.70): weak/irrelevant signal

  Same "cost of living" narrative:
    Youth-heavy urban seat (youth_pct=0.40): strong anti-incumbent
    Older rural seat (youth_pct=0.18): weaker signal

  By weighting themes by demographics, we get GENUINE
  seat-level variation from national-level news signals.

Usage:
    python backend/scripts/add_ethnicity_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ETHNICITY_DIR = Path("data/raw/ethnicity")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Load ethnicity data ────────────────────────────────────────────

def load_ethnicity_for_state(state: str, year: int) -> pd.DataFrame:
    """Load pre-computed ethnicity features for a state/year."""
    file_map = {
        ('johor', 2022):        'ethnicity_johor_2022.csv',
        ('johor', 2026):        'ethnicity_johor_2026.csv',
        ('neg_sembilan', 2023): 'ethnicity_ns_2023.csv',
        ('neg_sembilan', 2026): 'ethnicity_ns_2026.csv',
        ('melaka', 2021):       'ethnicity_melaka_2021.csv',
        ('melaka', 2026):       'ethnicity_melaka_2026.csv',
    }

    key = (state, year)
    if key not in file_map:
        print(f"  No ethnicity file mapped for {state} {year}")
        return pd.DataFrame()

    path = ETHNICITY_DIR / file_map[key]
    if not path.exists():
        print(f"  File not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded ethnicity: {state} {year} -> {len(df)} seats")
    return df


def load_national_narratives() -> dict:
    """
    Load national narrative scores computed by sentiment_scorer.py.
    These are national-level signals (not state-specific).
    """
    narrative_path = PROCESSED_DIR / "national_narrative_scores.csv"

    default = {
        'islam_threat': 0.0, 'malay_unity': 0.0,
        'cost_living': 0.0, 'corruption': 0.0,
        'govt_performance': 0.0,
    }

    if not narrative_path.exists():
        print(f"  No national narrative scores found, using zeros")
        return default

    df = pd.read_csv(narrative_path)
    if df.empty:
        return default

    row = df.iloc[0]
    narratives = {
        'islam_threat':     float(row.get('islam_threat', 0.0)),
        'malay_unity':      float(row.get('malay_unity', 0.0)),
        'cost_living':      float(row.get('cost_living', 0.0)),
        'corruption':       float(row.get('corruption', 0.0)),
        'govt_performance': float(row.get('govt_performance', 0.0)),
    }

    print(f"  Loaded national narratives: "
          f"Islam={narratives['islam_threat']:.3f} "
          f"Unity={narratives['malay_unity']:.3f} "
          f"Cost={narratives['cost_living']:.3f} "
          f"Corruption={narratives['corruption']:.3f}")

    return narratives


def clean_seat_name(seat: str) -> str:
    return str(seat).strip()


# ── Apply narratives per seat ──────────────────────────────────────

def apply_narratives_to_seat(narratives: dict,
                              malay_pct: float,
                              chinese_pct: float,
                              youth_pct: float) -> float:
    """
    Weight national narratives by seat demographics.

    Returns a narrative_pressure score per seat:
      Negative = more pressure on incumbent
      Positive = less pressure on incumbent

    "Islam terancam" strong + Malay-majority seat:
      -> Malays consolidate toward PN (away from BN/Harapan)
      -> Negative pressure on non-PN incumbents in Malay seats

    "Cost of living" strong + Chinese/youth-heavy seat:
      -> Anti-incumbent sentiment in urban/younger voter seats
      -> Negative pressure on whoever is in power

    "Malay unity" strong + Malay-majority seat:
      -> BN-PN bloc benefits
      -> Positive pressure for government coalition in Malay seats

    "Corruption" strong (all seats):
      -> Hurts incumbents universally

    "Govt performance" strong (all seats):
      -> Helps incumbents universally
    """
    islam_effect       = narratives['islam_threat']    * malay_pct   * -0.5
    unity_effect       = narratives['malay_unity']     * malay_pct   *  0.3
    cost_effect        = narratives['cost_living']     * (chinese_pct + youth_pct) * -0.4
    corruption_effect  = narratives['corruption']                     * -0.3
    performance_effect = narratives['govt_performance']               *  0.2

    pressure = (
        islam_effect +
        unity_effect +
        cost_effect +
        corruption_effect +
        performance_effect
    )

    return round(float(pressure), 4)


# ── Main merge function ────────────────────────────────────────────

def merge_ethnicity_into_features(
        df_features: pd.DataFrame,
        state: str,
        year_b: int,
        sentiment: dict,
        economic_pressure: float
) -> pd.DataFrame:
    """
    Merge ethnicity + age features and compute all interactions.

    Adds per seat:
      Ethnicity (8): malay_pct, chinese_pct, indian_pct,
                     young_malay_pct, young_chinese_pct,
                     older_malay_pct, youth_pct, median_age

      Sentiment x Ethnicity interactions (5):
        bn_sent_x_malay, harapan_sent_x_chinese,
        pn_sent_x_young_malay, tension_x_mixed,
        economic_x_youth

      National narrative x Demographics (1):
        narrative_pressure  <- national themes weighted by seat demographics
    """
    df = df_features.copy()

    # Guard: already merged
    if 'malay_pct' in df.columns:
        print(f"  Ethnicity features already present, skipping merge")
        return df

    df['seat_clean'] = df['seat'].apply(clean_seat_name)

    # Load ethnicity
    eth = load_ethnicity_for_state(state, year_b)

    if eth.empty:
        print(f"  No ethnicity data for {state} {year_b}, using zeros")
        for col in ['malay_pct', 'chinese_pct', 'indian_pct',
                    'young_malay_pct', 'young_chinese_pct',
                    'older_malay_pct', 'youth_pct', 'median_age']:
            df[col] = 0.0
    else:
        eth['seat_clean'] = eth['seat'].apply(clean_seat_name)
        df = df.merge(
            eth[['seat_clean', 'malay_pct', 'chinese_pct', 'indian_pct',
                 'young_malay_pct', 'young_chinese_pct', 'older_malay_pct',
                 'youth_pct', 'median_age']],
            on='seat_clean', how='left'
        )
        eth_cols = ['malay_pct', 'chinese_pct', 'indian_pct',
                    'young_malay_pct', 'young_chinese_pct',
                    'older_malay_pct', 'youth_pct', 'median_age']
        for col in eth_cols:
            df[col] = df[col].fillna(df[col].mean())

        matched = df['malay_pct'].notna().sum()
        print(f"  Matched {matched}/{len(df)} seats on ethnicity data")

    # ── Sentiment x Ethnicity interactions ─────────────────────────
    bn_sent      = sentiment['bn_sentiment']
    harapan_sent = sentiment['harapan_sentiment']
    pn_sent      = sentiment['pn_sentiment']
    tension      = sentiment['racial_tension_index']

    df['bn_sent_x_malay']         = bn_sent      * df['malay_pct']
    df['harapan_sent_x_chinese']  = harapan_sent * df['chinese_pct']
    df['pn_sent_x_young_malay']   = pn_sent      * df['young_malay_pct']
    df['mixed_ratio']             = 1 - (df['malay_pct'] - df['chinese_pct']).abs()
    df['tension_x_mixed']         = tension      * df['mixed_ratio']
    df['economic_x_youth']        = economic_pressure * df['youth_pct']

    print(f"  Added 5 sentiment x ethnicity interaction features")

    # ── National narrative x Demographics (NEW) ────────────────────
    narratives = load_national_narratives()

    df['narrative_pressure'] = df.apply(
        lambda row: apply_narratives_to_seat(
            narratives,
            malay_pct=row.get('malay_pct', 0.0),
            chinese_pct=row.get('chinese_pct', 0.0),
            youth_pct=row.get('youth_pct', 0.0),
        ),
        axis=1
    )

    print(f"  Added narrative_pressure feature")
    print(f"    Range: {df['narrative_pressure'].min():.4f} "
          f"to {df['narrative_pressure'].max():.4f}")
    print(f"    (Different values per seat based on demographics)")

    df = df.drop(columns=['seat_clean', 'mixed_ratio'], errors='ignore')

    return df


# ── Validation ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Validating ethnicity + narrative features...\n")
    print("="*60)

    master = pd.read_csv(ETHNICITY_DIR / "ethnicity_all.csv")
    print(f"Total seat-year rows: {len(master)}")

    narratives = load_national_narratives()

    print(f"\nNarrative pressure per seat type:")
    examples = [
        ('Rural Malay (BN/PN safe)',    0.80, 0.08, 0.18),
        ('Urban Chinese (DAP safe)',    0.20, 0.72, 0.35),
        ('Mixed suburban (swing seat)', 0.50, 0.42, 0.30),
        ('Young Malay urban',           0.55, 0.35, 0.45),
    ]
    for label, m, c, y in examples:
        p = apply_narratives_to_seat(narratives, m, c, y)
        direction = "more pressure on incumbent" if p < 0 else "less pressure"
        print(f"  {label:<35}: {p:+.4f}  ({direction})")

    print(f"\n6 wrong Johor predictions (Chinese-majority):")
    wrong = ['N.12 Bentayan', 'N.13 Simpang Jeram', 'N.41 Puteri Wangsa',
             'N.45 Stulang', 'N.48 Skudai', 'N.52 Senai']
    johor = master[(master['state'] == 'johor') &
                   (master['election_year'] == 2026)]
    wrong_df = johor[johor['seat'].isin(wrong)]

    for _, row in wrong_df.iterrows():
        p = apply_narratives_to_seat(
            narratives, row['malay_pct'], row['chinese_pct'], row['youth_pct']
        )
        print(f"  {row['seat']:<25}: chinese={row['chinese_pct']:.2f} "
              f"narrative_pressure={p:+.4f}")