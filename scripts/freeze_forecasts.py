"""
scripts/freeze_forecasts.py
===========================
Save a dated, checksummed copy of the model's forecasts for states whose next
election has NOT happened yet (Selangor, Melaka, Perak).

Why: when the election happens, the score is only believable if the predictions
were written down BEFORE the result. A frozen folder is that record.

What is saved (one folder per state and date, never edited afterwards):

    forecasts/<state>/<YYYY-MM-DD>/forecast.csv    one row per seat
    forecasts/<state>/<YYYY-MM-DD>/manifest.json   the receipt

forecast.csv has the model's call and probability for every seat, and the two
simple rules it will be compared with ("never opposition" and "same as last time"),
saved at the same moment so the later comparison is fair.

manifest.json is the receipt: date, code commit, SHA-256 of the forecast file, of
the model files and of the data files used, and the cut-off.

Usage:
    python scripts/freeze_forecasts.py                    # all pending states
    python scripts/freeze_forecasts.py selangor perak
    python scripts/freeze_forecasts.py --verify           # check every frozen folder

Rules:
  * only states with no 2026 result yet (final_test is None) can be frozen
  * a folder that already exists is never overwritten
  * freezing requires a clean git tree (so the recorded commit is the code that ran)
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from backend.core.features import OPPOSITION, FEATURE_VERSION
from backend.core.states import STATES

FORECAST_DIR = ROOT / 'forecasts'
CUTOFF = 0.5

FORECAST_COLUMNS = [
    'seat_no', 'seat_name', 'predicted_opposition', 'prediction', 'probability',
    'rf_prob', 'xgb_prob', 'calibrated_prob', 'confidence', 'is_ood', 'ood_score',
    'model_used', 'base_rate', 'baseline_never_opposition', 'baseline_same_as_last',
]
MODEL_FILES = ['rf_model.pkl', 'xgb_model.pkl', 'rf_calibrated.pkl', 'ood_detector.pkl', 'metadata.json']


# ── hashing ────────────────────────────────────────────────────────────

def sha256_file(path: Path, text: bool = False) -> str:
    """SHA-256 of a file. For text files, line endings are normalised first so a
    Windows checkout (CRLF) gives the same hash as the committed file (LF)."""
    data = Path(path).read_bytes()
    if text:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def is_text(name: str) -> bool:
    """Text files are hashed with line endings normalised (Windows may check them out as CRLF).
    Model pickles and parquet files are binary and are hashed as they are."""
    return str(name).endswith(('.csv', '.json'))


def pending_states():
    """States whose next election has not been held yet (no real result to score)."""
    return [s for s, v in STATES.items() if v['final_test'] is None]


# ── git helpers ────────────────────────────────────────────────────────

def _git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)


def code_commit():
    r = _git('rev-parse', 'HEAD')
    return r.stdout.strip() if r.returncode == 0 else None


def tree_is_clean():
    """True when no tracked file has uncommitted changes (untracked files are ignored)."""
    r = _git('status', '--porcelain', '--untracked-files=no')
    return r.returncode == 0 and r.stdout.strip() == ''


# ── building one forecast ──────────────────────────────────────────────

def build_forecast(state: str):
    """Returns (forecast_df, predictor). Uses the same StatePredictor as the API."""
    from backend.core.models.state_predictor import StatePredictor
    predictor = StatePredictor(state)
    live = predictor._build_live_features().reset_index(drop=True)
    preds = predictor.predict_all().reset_index(drop=True)
    if len(live) != len(preds):
        raise RuntimeError(f'{state}: {len(live)} feature rows but {len(preds)} predictions')

    out = preds.copy()
    out.insert(0, 'seat_no', live['seat_no'].values)
    out['predicted_opposition'] = (out['probability'] >= CUTOFF).astype(int)
    # the two simple rules, from election A only (known today)
    out['baseline_never_opposition'] = 0
    out['baseline_same_as_last'] = live['winner_coalition_a'].isin(OPPOSITION).astype(int).values
    out = out[FORECAST_COLUMNS].sort_values('seat_no').reset_index(drop=True)
    return out, predictor


def freeze_state(state: str, allow_dirty: bool = False, today: str = None) -> Path:
    if state not in STATES:
        raise SystemExit(f'Unknown state: {state}')
    if STATES[state]['final_test'] is not None:
        raise SystemExit(f'{state} has already held its election; only pending states are frozen.')
    if not allow_dirty and not tree_is_clean():
        raise SystemExit('Tracked files have uncommitted changes. Commit them first '
                         '(so the recorded commit is the code that ran), or use --allow-dirty.')

    now = datetime.now(timezone.utc)
    day = today or now.strftime('%Y-%m-%d')
    folder = FORECAST_DIR / state / day
    if folder.exists():
        raise SystemExit(f'{folder} already exists. Frozen forecasts are never overwritten; '
                         f'use a later date for a new freeze.')

    forecast, predictor = build_forecast(state)
    folder.mkdir(parents=True)
    csv_path = folder / 'forecast.csv'
    forecast.to_csv(csv_path, index=False, lineterminator='\n')

    cfg = STATES[state]
    model_dir = ROOT / 'backend' / 'models' / state
    stem = cfg['file_stem']
    data_files = [ROOT / 'data' / 'raw' / f'{stem}_ballots.parquet',
                  ROOT / 'data' / 'raw' / f'{stem}_stats.parquet',
                  ROOT / 'data' / 'raw' / 'ethnicity' / f'ethnicity_{state}_{cfg["ethnicity_year"]}.csv']
    meta = json.loads((model_dir / 'metadata.json').read_text(encoding='utf-8'))

    manifest = {
        'state': state,
        'frozen_at_utc': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'forecasting_election_after': int(predictor.year_a),
        'n_seats': int(len(forecast)),
        'cutoff': CUTOFF,
        'target': 'opposition (Pakatan Harapan) wins the seat; BN and PN count together as "no"',
        'feature_version': FEATURE_VERSION,
        'model_feature_version': meta.get('feature_version'),
        'code_commit': code_commit(),
        'forecast_csv_sha256': sha256_file(csv_path, text=True),
        'model_files_sha256': {f: sha256_file(model_dir / f, text=is_text(f)) for f in MODEL_FILES if (model_dir / f).exists()},
        'data_files_sha256': {f.name: sha256_file(f, text=is_text(f.name)) for f in data_files if f.exists()},
        'notes': [
            'Sentiment and narrative features are not used by these models (held at 0).',
            'Baselines "never opposition" and "same as last time" are saved in forecast.csv.',
            'Do not edit these files. A new forecast goes in a new dated folder.',
        ],
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    shown = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
    print(f'  froze {state}: {len(forecast)} seats -> {shown}')
    return folder


# ── verification ───────────────────────────────────────────────────────

def verify_folder(folder: Path, check_models: bool = False) -> list:
    """Returns a list of problems (empty list = fine)."""
    folder = Path(folder)
    problems = []
    mpath, cpath = folder / 'manifest.json', folder / 'forecast.csv'
    if not mpath.exists() or not cpath.exists():
        return [f'{folder}: manifest.json or forecast.csv is missing']
    manifest = json.loads(mpath.read_text(encoding='utf-8'))

    if sha256_file(cpath, text=True) != manifest['forecast_csv_sha256']:
        problems.append(f'{folder}: forecast.csv was changed after it was frozen')

    df = pd.read_csv(cpath)
    if list(df.columns) != FORECAST_COLUMNS:
        problems.append(f'{folder}: unexpected columns')
    if len(df) != manifest['n_seats']:
        problems.append(f'{folder}: {len(df)} rows but manifest says {manifest["n_seats"]} seats')
    if df['seat_no'].duplicated().any():
        problems.append(f'{folder}: duplicate seats')
    if not df['probability'].between(0, 1).all():
        problems.append(f'{folder}: probability outside 0..1')

    if check_models:   # compare the model files with the ones at the recorded commit
        for name, expected in manifest['model_files_sha256'].items():
            rel = f'backend/models/{manifest["state"]}/{name}'
            r = subprocess.run(['git', 'show', f'{manifest["code_commit"]}:{rel}'], cwd=ROOT, capture_output=True)
            if r.returncode != 0:
                problems.append(f'{folder}: {rel} not found at commit {manifest["code_commit"]}')
            elif hashlib.sha256(r.stdout.replace(b'\r\n', b'\n') if is_text(name) else r.stdout).hexdigest() != expected:
                problems.append(f'{folder}: {rel} at the recorded commit does not match the manifest')
    return problems


def frozen_folders(root: Path = None):
    root = Path(root) if root else FORECAST_DIR
    return sorted(p.parent for p in root.glob('*/*/manifest.json'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('states', nargs='*', help='states to freeze (default: all pending states)')
    ap.add_argument('--verify', action='store_true', help='check every frozen folder, then exit')
    ap.add_argument('--check-models', action='store_true', help='with --verify: also check model files at the recorded commit')
    ap.add_argument('--allow-dirty', action='store_true', help='freeze even with uncommitted changes (not recommended)')
    args = ap.parse_args()

    if args.verify:
        folders = frozen_folders()
        bad = [p for f in folders for p in verify_folder(f, check_models=args.check_models)]
        print(f'Checked {len(folders)} frozen folder(s).')
        for p in bad:
            print('  PROBLEM:', p)
        sys.exit(1 if bad else 0)

    states = args.states or pending_states()
    print('Freezing forecasts for:', ', '.join(states))
    for s in states:
        freeze_state(s, allow_dirty=args.allow_dirty)
    print('Next: git add forecasts/, commit, open a PR, merge, then tag the release.')


if __name__ == '__main__':
    main()