"""
dag_drift.py
============
DAG 3: Model drift detection
Triggered manually after new election results available.

Tasks:
  1. load_new_results   → load actual election results
  2. compute_accuracy   → compare predictions vs actuals
  3. detect_drift       → check if accuracy below threshold
  4. retrain_if_needed  → auto retrain if drift detected
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.85  # retrain if accuracy drops below 85%


def task_load_predictions() -> dict:
    """Task 1: Load existing predictions for all states."""
    logger.info("Loading predictions...")

    try:
        import pandas as pd

        predictions = {}
        for state in ['johor', 'neg_sembilan']:
            path = ROOT / f"backend/models/{state}/validation_2026.csv"
            if path.exists():
                df = pd.read_csv(path)
                predictions[state] = df.to_dict('records')
                logger.info(f"  {state}: {len(df)} seats loaded")

        return {'status': 'success', 'states': list(predictions.keys())}

    except Exception as e:
        logger.error(f"Load failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_compute_accuracy() -> dict:
    """Task 2: Compute current model accuracy per state."""
    logger.info("Computing accuracy...")

    try:
        import pandas as pd

        accuracy = {}
        for state in ['johor', 'neg_sembilan']:
            path = ROOT / f"backend/models/{state}/validation_2026.csv"
            if not path.exists():
                continue

            df = pd.read_csv(path)
            if 'prediction' not in df.columns or 'actual' not in df.columns:
                continue

            correct = (df['prediction'].str.strip() ==
                      df['actual'].str.strip()).sum()
            total   = len(df)
            acc     = correct / total if total > 0 else 0

            accuracy[state] = {
                'accuracy':  round(acc, 4),
                'correct':   int(correct),
                'total':     int(total),
                'wrong':     int(total - correct),
            }
            logger.info(f"  {state}: {acc:.2%} ({correct}/{total})")

        save_path = ROOT / "data/processed/current_accuracy.json"
        with open(save_path, 'w') as f:
            json.dump(accuracy, f, indent=2)

        return {'status': 'success', 'accuracy': accuracy}

    except Exception as e:
        logger.error(f"Accuracy computation failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_detect_drift() -> dict:
    """
    Task 3: Check if model accuracy has drifted below threshold.
    Drift = accuracy dropped significantly from expected level.
    """
    logger.info(f"Detecting drift (threshold: {DRIFT_THRESHOLD:.0%})...")

    try:
        accuracy_path = ROOT / "data/processed/current_accuracy.json"
        if not accuracy_path.exists():
            return {'status': 'error', 'error': 'No accuracy data found'}

        with open(accuracy_path) as f:
            accuracy = json.load(f)

        drift_detected = False
        drift_states   = []

        for state, metrics in accuracy.items():
            acc = metrics['accuracy']
            if acc < DRIFT_THRESHOLD:
                drift_detected = True
                drift_states.append(state)
                logger.warning(
                    f"  DRIFT DETECTED: {state} accuracy = {acc:.2%} "
                    f"(below threshold {DRIFT_THRESHOLD:.0%})"
                )
            else:
                logger.info(
                    f"  OK: {state} accuracy = {acc:.2%} "
                    f"(above threshold)"
                )

        return {
            'status':         'success',
            'drift_detected': drift_detected,
            'drift_states':   drift_states,
            'accuracy':       accuracy,
        }

    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        return {'status': 'error', 'error': str(e)}


def task_retrain_if_needed(drift_result: dict) -> dict:
    """
    Task 4: Retrain models if drift was detected.
    Logs the retraining event with timestamp.
    """
    if not drift_result.get('drift_detected'):
        logger.info("No drift detected — retraining not needed")
        return {'status': 'success', 'retrained': False}

    drift_states = drift_result.get('drift_states', [])
    logger.warning(f"Retraining models for: {drift_states}")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable,
             str(ROOT / "backend/scripts/train_models.py")],
            capture_output=True, text=True, cwd=str(ROOT)
        )

        if result.returncode == 0:
            logger.info("Retraining completed successfully")

            # Log drift event
            log_path = ROOT / "mlops/logs/drift_events.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)

            events = []
            if log_path.exists():
                with open(log_path) as f:
                    events = json.load(f)

            events.append({
                'timestamp':    datetime.now().isoformat(),
                'drift_states': drift_states,
                'accuracy':     drift_result.get('accuracy', {}),
                'action':       'retrained',
            })

            with open(log_path, 'w') as f:
                json.dump(events, f, indent=2)

            return {'status': 'success', 'retrained': True}
        else:
            logger.error(f"Retraining failed: {result.stderr[:300]}")
            return {'status': 'error', 'error': result.stderr[:300]}

    except Exception as e:
        logger.error(f"Retrain failed: {e}")
        return {'status': 'error', 'error': str(e)}


def run_dag():
    dag_name = "dag_03_drift_detection"
    start_time = datetime.now()
    logger.info(f"{'='*50}")
    logger.info(f"DAG START: {dag_name} at {start_time}")
    logger.info(f"{'='*50}")

    # Tasks 1-3 run sequentially
    r1 = task_load_predictions()
    if r1.get('status') == 'error':
        logger.error(f"[FAILED] load_predictions: {r1}")
        return

    r2 = task_compute_accuracy()
    if r2.get('status') == 'error':
        logger.error(f"[FAILED] compute_accuracy: {r2}")
        return

    r3 = task_detect_drift()
    if r3.get('status') == 'error':
        logger.error(f"[FAILED] detect_drift: {r3}")
        return

    # Task 4 receives drift result as input
    r4 = task_retrain_if_needed(r3)

    duration = (datetime.now() - start_time).seconds
    logger.info(f"\nDAG COMPLETE: {dag_name} in {duration}s")
    logger.info(f"  Drift detected: {r3.get('drift_detected')}")
    logger.info(f"  Retrained: {r4.get('retrained')}")

    return {'drift': r3, 'retrain': r4}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_dag()