import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from train_models import main


def run_dag():
    print("=" * 60)
    print("  DAG: Scheduled Retraining")
    print("=" * 60)
    try:
        main()
        print("\nRetraining DAG completed.")
    except Exception as e:
        print(f"\nRetraining DAG failed: {e}")
        raise


if __name__ == "__main__":
    run_dag()