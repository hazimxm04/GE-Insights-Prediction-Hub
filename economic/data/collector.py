import os
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("economic/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────

INDICATORS = {
    "klci": {
        "ticker":      "^KLSE",
        "name":        "FTSE Bursa Malaysia KLCI",
        "start":       "2010-01-01",
        "description": "Malaysian stock market index — proxy for economic confidence"
    },
    "usd_myr": {
        "ticker":      "MYR=X",
        "name":        "USD/MYR Exchange Rate",
        "start":       "2010-01-01",
        "description": "Ringgit weakness signals economic stress → hurts incumbent"
    },
}

# ── Downloaders ────────────────────────────────────────────────────

def download_yfinance(indicator_id: str, config: dict) -> pd.DataFrame:
    """Download price data from Yahoo Finance"""
    print(f"\nDownloading {config['name']}...")

    ticker = yf.Ticker(config["ticker"])
    df = ticker.history(start=config["start"], end=datetime.today().strftime("%Y-%m-%d"))

    if df.empty:
        print(f"  No data returned for {config['ticker']}")
        return pd.DataFrame()

    df = df[["Close"]].copy()
    df.columns = ["value"]
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # Remove weekends/holidays where market was closed
    df = df.dropna()

    save_path = DATA_DIR / f"{indicator_id}.csv"
    df.to_csv(save_path)

    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Latest value: {df['value'].iloc[-1]:.4f}")
    print(f"  Saved to: {save_path}")

    return df


def download_bnm_rate() -> pd.DataFrame:
    """
    Download BNM overnight policy rate (OPR) via Bank Negara API.
    Free, official Malaysian central bank data.
    """
    print("\nDownloading BNM Overnight Policy Rate...")

    url = "https://api.bnm.gov.my/public/opr"
    headers = {"Accept": "application/vnd.BNM.API.v1+json"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()["data"]

        records = []
        for item in data:
            records.append({
                'date':  item.get('date') or item.get('date_start'),
                'value': item.get('rate') or item.get('opr_rate'),
            })

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df.dropna()

        save_path = DATA_DIR / "bnm_opr.csv"
        df.to_csv(save_path)

        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Latest OPR: {df['value'].iloc[-1]}%")
        print(f"  Saved to: {save_path}")

        return df

    except Exception as e:
        print(f"  BNM API failed: {e}")
        return pd.DataFrame()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading Malaysian economic indicators...\n")
    print("="*55)

    results = {}

    # Download KLCI and USD/MYR
    for indicator_id, config in INDICATORS.items():
        df = download_yfinance(indicator_id, config)
        if not df.empty:
            results[indicator_id] = df

    # Download BNM OPR
    bnm_df = download_bnm_rate()
    if not bnm_df.empty:
        results["bnm_opr"] = bnm_df

    # Summary
    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    for name, df in results.items():
        print(f"  {name:<12}: {len(df):>5} rows | "
              f"{df.index.min().date()} to {df.index.max().date()}")

    print(f"\nData saved to: {DATA_DIR}")
    print("Next: python economic/data/preprocessor.py")