"""
scheduler.py
============
Main APScheduler entry point.
Runs all 3 DAGs on their respective schedules.

Schedules:
  DAG 1 (sentiment):  daily at 8:00am
  DAG 2 (economic):   every Monday at 6:00am
  DAG 3 (drift):      manual trigger only

Usage:
  python mlops/scheduler.py          ← start scheduler
  python mlops/scheduler.py --test   ← run all DAGs once now
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging setup ────────────────────────────────────────────────

LOG_DIR = ROOT / "mlops/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("ge_insights.scheduler")

# ── Import DAGs ──────────────────────────────────────────────────

from mlops.dags.dag_sentiment import run_dag as run_sentiment_dag
from mlops.dags.dag_economic  import run_dag as run_economic_dag
from mlops.dags.dag_drift     import run_dag as run_drift_dag

# ── Scheduler ────────────────────────────────────────────────────

def start_scheduler():
    """Start APScheduler with all 3 DAGs on schedule."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Asia/Kuala_Lumpur")

    # DAG 1: Daily at 8am (sentiment + RAG rebuild)
    scheduler.add_job(
        run_sentiment_dag,
        trigger=CronTrigger(hour=8, minute=0),
        id="dag_01_daily_sentiment",
        name="Daily sentiment pipeline",
        max_instances=1,
        misfire_grace_time=3600,
    )

    # DAG 2: Every Monday at 6am (economic data)
    scheduler.add_job(
        run_economic_dag,
        trigger=CronTrigger(day_of_week="mon", hour=6, minute=0),
        id="dag_02_weekly_economic",
        name="Weekly economic pipeline",
        max_instances=1,
        misfire_grace_time=3600,
    )

    # DAG 3: Not scheduled — triggered manually after elections
    # Run: python -c "from mlops.dags.dag_drift import run_dag; run_dag()"

    logger.info("="*55)
    logger.info("GE-Insights MLOps Scheduler started")
    logger.info("="*55)
    logger.info("Scheduled jobs:")
    logger.info("  DAG 1 (sentiment): daily at 08:00 MYT")
    logger.info("  DAG 2 (economic):  every Monday at 06:00 MYT")
    logger.info("  DAG 3 (drift):     manual trigger only")
    logger.info("Logs: mlops/logs/pipeline.log")
    logger.info("Press Ctrl+C to stop")
    logger.info("="*55)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
        scheduler.shutdown()


def test_all_dags():
    """Run all DAGs once immediately for testing."""
    logger.info("TEST MODE: Running all DAGs once")

    logger.info("\n" + "="*40)
    logger.info("Running DAG 1: Daily sentiment")
    logger.info("="*40)
    run_sentiment_dag()

    logger.info("\n" + "="*40)
    logger.info("Running DAG 2: Weekly economic")
    logger.info("="*40)
    run_economic_dag()

    logger.info("\n" + "="*40)
    logger.info("Running DAG 3: Drift detection")
    logger.info("="*40)
    run_drift_dag()

    logger.info("\nAll DAGs completed. Check mlops/logs/pipeline.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                        help='Run all DAGs once immediately')
    args = parser.parse_args()

    if args.test:
        test_all_dags()
    else:
        start_scheduler()