#!/usr/bin/env python
"""
Download election data (ballots + stats) for every state in the registry
(backend/core/states.py) from electiondata.my into data/raw/.

    python backend/scripts/download_data.py            # only what is missing
    python backend/scripts/download_data.py --refresh  # re-download everything (after a new election)

This is a thin wrapper around scripts/state_data.py, the one shared loader, so the file
names and URLs are defined in one place. A refresh downloads to a temporary file first, so a
failed download never deletes the copy you already have.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.core.states import STATES
from state_data import load_state_data


def main():
    refresh = '--refresh' in sys.argv
    failed = []
    for state in STATES:
        print(f"{state}: {'refreshing' if refresh else 'checking'} ...")
        try:
            ballots, stats = load_state_data(state, refresh=refresh)
            print(f"  ok ({len(ballots)} ballot rows, {len(stats)} stats rows)")
        except Exception as e:
            failed.append(state)
            print(f"  FAILED: {e}")
    if failed:
        print(f"\nFailed: {', '.join(failed)}")
        sys.exit(1)
    print("\nDownload complete.")


if __name__ == "__main__":
    main()