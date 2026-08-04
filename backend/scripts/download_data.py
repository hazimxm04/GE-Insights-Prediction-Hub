#!/usr/bin/env python
"""
Download election data from electiondata.my
Run once: python backend/scripts/download_data.py
Re-run after new elections only.
"""

import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────

DATA_LAKE = "https://lake.electiondata.my/results_headline"

DATASETS = {
    # Ballots (candidate-level: who ran, how many votes)
    "johor_ballots":      f"{DATA_LAKE}/headline_ballots_state_jhr.parquet",
    "ns_ballots":         f"{DATA_LAKE}/headline_ballots_state_nsn.parquet",
    "melaka_ballots":     f"{DATA_LAKE}/headline_ballots_state_mlk.parquet",

    # Stats (seat-level: turnout, total voters, majority)
    "johor_stats":        f"{DATA_LAKE}/headline_stats_state_jhr.parquet",
    "ns_stats":           f"{DATA_LAKE}/headline_stats_state_nsn.parquet",
    "melaka_stats":       f"{DATA_LAKE}/headline_stats_state_mlk.parquet",
}

RAW_DATA_PATH = Path("data/raw")

# ── Download ──────────────────────────────────────────────────────

def download_all():
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for name, url in DATASETS.items():
        save_path = RAW_DATA_PATH / f"{name}.parquet"

        # Skip if already downloaded
        if save_path.exists():
            print(f"⏭️  Skipping {name} (already exists)")
            continue

        print(f"⬇️  Downloading {name}...")
        try:
            df = pd.read_parquet(url)
            df.to_parquet(save_path, index=False)
            print(f"✅ Saved: {save_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
            print(f"   Columns: {df.columns.tolist()}")
            print(f"   Years: {sorted(df['date'].dt.year.unique()) if 'date' in df.columns else 'N/A'}")
        except Exception as e:
            print(f"❌ Failed {name}: {e}")

    print("\n✅ Download complete!")

def force_refresh():
    """Re-download everything (use after new election)"""
    for name in DATASETS:
        save_path = RAW_DATA_PATH / f"{name}.parquet"
        if save_path.exists():
            save_path.unlink()
            print(f"🗑️  Deleted: {save_path}")
    download_all()

if __name__ == "__main__":
    import sys
    if "--refresh" in sys.argv:
        force_refresh()
    else:
        download_all()