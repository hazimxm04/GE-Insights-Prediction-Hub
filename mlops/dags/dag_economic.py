"""
dag_economic.py
===============
DAG 2: Weekly economic pipeline
Runs every Monday at 6am.

Tasks:
  1. fetch_economic_data  → download KLCI + USD/MYR via yfinance
  2. run_lstm_forecast    → load saved model, generate forecast
  3. update_pressure      → compute election period pressure scores
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def task_fetch_economic_data() -> dict:
    """Task 1: Download latest KLCI + USD/MYR data."""
    logger.info("Fetching economic data...")

    try:
        import yfinance as yf
        import pandas as pd

        tickers = {"klci": "^KLSE", "usd_myr": "MYR=X"}
        save_dir = ROOT / "economic/data/raw"
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, ticker in tickers.items():
            df = yf.Ticker(ticker).history(start="2010-01-01")
            df = df[["Close"]].rename(columns={"Close": "value"})
            df.index.name = "date"
            df.index = df.index.tz_localize(None)
            df = df.dropna()
            df.to_csv(save_dir / f"{name}.csv")
            logger.info(f"  {name}: {len(df)} rows")

        return {'status': 'success'}

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_run_lstm_forecast() -> dict:
    """Task 2: Load saved LSTM model and generate new forecasts."""
    logger.info("Running LSTM forecast...")

    try:
        import numpy as np
        import pandas as pd
        import torch
        import pickle

        sys.path.insert(0, str(ROOT))
        from economic.models.lstm_model import EconomicLSTM, DEVICE

        proc_dir   = ROOT / "economic/data/processed"
        models_dir = ROOT / "economic/models/saved"

        # Re-run preprocessor to get fresh sequences
        from economic.data.preprocessor import (
            load_and_align, normalize, create_sequences
        )

        df = load_and_align()
        df_scaled, scaler = normalize(df)

        results = {}
        for target in ["klci", "usd_myr"]:
            model = EconomicLSTM().to(DEVICE)
            model.load_state_dict(
                torch.load(models_dir / f"lstm_{target}_best.pt",
                           map_location=DEVICE)
            )
            model.eval()

            X, y, dates = create_sequences(df_scaled, target_col=target)
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X[-100:]).to(DEVICE)
                preds = model(X_tensor).cpu().numpy()

            results[target] = {
                'latest_pred': float(preds[-1]),
                'n_samples': len(preds)
            }
            logger.info(f"  {target}: latest pred = {preds[-1]:.4f}")

        return {'status': 'success', 'results': results}

    except Exception as e:
        logger.error(f"LSTM forecast failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_update_pressure_scores() -> dict:
    """Task 3: Recompute economic pressure scores."""
    logger.info("Updating economic pressure scores...")

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(ROOT / "economic/evaluation/evaluator.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )

        if result.returncode == 0:
            logger.info("Pressure scores updated successfully")
            return {'status': 'success'}
        else:
            logger.error(f"Evaluator failed: {result.stderr[:200]}")
            return {'status': 'error', 'error': result.stderr[:200]}

    except Exception as e:
        logger.error(f"Pressure update failed: {e}")
        return {'status': 'error', 'error': str(e)}


def run_dag():
    dag_name = "dag_02_weekly_economic"
    start_time = datetime.now()
    logger.info(f"{'='*50}")
    logger.info(f"DAG START: {dag_name} at {start_time}")
    logger.info(f"{'='*50}")

    tasks = [
        ("fetch_economic_data",   task_fetch_economic_data),
        ("run_lstm_forecast",     task_run_lstm_forecast),
        ("update_pressure_scores", task_update_pressure_scores),
    ]

    results = {}
    for task_name, task_fn in tasks:
        logger.info(f"\n[TASK] {task_name}")
        result = task_fn()
        results[task_name] = result

        if result.get('status') == 'error':
            logger.error(f"[TASK FAILED] {task_name}: {result.get('error')}")
            break
        else:
            logger.info(f"[TASK OK] {task_name}: {result}")

    duration = (datetime.now() - start_time).seconds
    logger.info(f"\nDAG COMPLETE: {dag_name} in {duration}s")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dag()