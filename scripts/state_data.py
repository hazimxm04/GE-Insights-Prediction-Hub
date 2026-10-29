"""
One place that loads a state's election data (ballots + stats).

Order of preference:
  1. the local copy in data/raw/  (reproducible, offline)
  2. download it from lake.electiondata.my and save it there (only if missing)

To refresh after a new election:   python backend/scripts/download_data.py --refresh
File names match backend/scripts/download_data.py and the pipeline config.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.core.states import STATES
RAW = ROOT / 'data' / 'raw'
LAKE = 'https://lake.electiondata.my/results_headline'

STATE_URL_PREFIX = {k: v['url_prefix'] for k, v in STATES.items()}
FILE_STEM = {k: v['file_stem'] for k, v in STATES.items()}


def _paths(state):
    stem = FILE_STEM[state]
    return RAW / f'{stem}_ballots.parquet', RAW / f'{stem}_stats.parquet'


def _download(url, dest):
    try:
        df = pd.read_parquet(url)
    except Exception as e:
        raise RuntimeError(
            f'No local copy at {dest} and could not download {url} '
            f'({type(e).__name__}: {e}). Run backend/scripts/download_data.py '
            f'when you have network access.') from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(dest)
    print(f'  downloaded and saved {dest.name} ({len(df)} rows)')


def load_state_data(state, refresh=False):
    """Return (ballots, stats) for a state. Uses the local copy unless it is
    missing (then downloads once) or refresh=True."""
    if state not in STATE_URL_PREFIX:
        raise KeyError(f'Unknown state: {state}')
    p = STATE_URL_PREFIX[state]
    b_path, s_path = _paths(state)
    if refresh or not b_path.exists():
        _download(f'{LAKE}/headline_ballots_state_{p}.parquet', b_path)
    if refresh or not s_path.exists():
        _download(f'{LAKE}/headline_stats_state_{p}.parquet', s_path)
    return pd.read_parquet(b_path), pd.read_parquet(s_path)
