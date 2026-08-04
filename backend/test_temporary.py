import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\hazim\Downloads\GE Insights Predictor")

files = {
    "johor_ballots":  BASE / "data/raw/johor_ballots.parquet",
    "johor_stats":    BASE / "data/raw/johor_stats.parquet",
    "ns_ballots":     BASE / "data/raw/ns_ballots.parquet",
    "ns_stats":       BASE / "data/raw/ns_stats.parquet",
    "melaka_ballots": BASE / "data/raw/melaka_ballots.parquet",
    "melaka_stats":   BASE / "data/raw/melaka_stats.parquet",
}

for name, path in files.items():
    if not path.exists():
        print(f"❌ NOT FOUND: {path}")
        continue

    print(f"\n{'='*55}")
    print(f"  {name.upper()}")
    print(f"{'='*55}")
    df = pd.read_parquet(path)
    print(f"Shape:   {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Fix: parse date as string, extract year manually
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year
        print(f"Unique years: {sorted(df['year'].dropna().astype(int).unique())}")

    for col in ['election', 'coalition', 'result', 'party']:
        if col in df.columns:
            print(f"Unique {col}: {df[col].unique().tolist()}")

    print(f"\nSample (2 rows):")
    print(df.head(2).to_string())
    print(f"\nNull counts (non-zero only):")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls[nulls > 0].any() else "None")