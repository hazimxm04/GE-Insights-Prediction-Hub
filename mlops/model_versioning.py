import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
from backend.core.features import FEATURE_VERSION
MODELS_DIR = ROOT / "backend" / "models"


def version_and_promote(state: str, new_metadata: dict, min_improvement: float = 0.0,
                        candidate_dir=None) -> bool:
    """
    Archive the current 'latest' model under a timestamped folder,
    then decide whether the NEW model should become 'latest' based
    on the held-out score (time-ordered test), never on training accuracy.

    KNOWN LIMITATION: if the current 'latest' model has a
    validated_accuracy (from validation.py) but the newly-trained
    model does not yet (validation.py hasn't been run on it), this
    function compares validated (trustworthy) against training
    (potentially misleading) accuracy -- an apples-to-oranges
    comparison. Correct workflow: run validation.py immediately
    after train_models.py, before training again, so
    validated_accuracy is populated before the NEXT comparison runs.

    Args:
        candidate_dir: folder holding the NEWLY trained model files, staged OUTSIDE 'latest'.
                       'latest' must still hold the PRIOR model, so the comparison is new vs prior
                       (before this, training overwrote 'latest' first and the gate compared the new
                       model with itself, so it always promoted).
        state: e.g. 'johor'
        new_metadata: the metadata dict just produced by train_state()
        min_improvement: require at least this much accuracy gain
                          over the current model to promote (0.0 =
                          promote if equal or better)

    Returns:
        bool: True if the new model was promoted to 'latest', False
              if the previous model was kept instead.

    IMPORTANT: new_metadata['models']['ensemble']['accuracy'] is
    TRAINING accuracy, not validated accuracy. Training accuracy on
    a small dataset (53-160 rows) is not a trustworthy promotion
    signal on its own -- as found during development, a model can
    show 100% training accuracy while performing far worse on real
    validation data (78.57% for Johor). Wherever possible, pass a
    validated_accuracy value (from validation.py, for states with
    real ground truth) instead of relying on training accuracy alone.
    """
    latest_dir = MODELS_DIR / state
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_dir = latest_dir / f'v_{timestamp}'
    version_dir.mkdir(parents=True, exist_ok=True)

    model_files = ['rf_model.pkl', 'xgb_model.pkl', 'rf_calibrated.pkl',
                   'ood_detector.pkl', 'metadata.json']

    # Archive whatever is CURRENTLY in 'latest' (the pre-existing model,
    # if any) into the timestamped folder before it gets overwritten.
    archived_prior = False
    for fname in model_files:
        src = latest_dir / fname
        if src.exists():
            shutil.copy(src, version_dir / fname)
            archived_prior = True

    if not archived_prior:
        print(f"  No prior model found for {state} -- this is the first version")
        promote = True
    else:
        current_metadata_path = version_dir / 'metadata.json'
        with open(current_metadata_path) as f:
            current_meta = json.load(f)

        # Compare on the HELD-OUT score only (time-ordered test), never on training
        # accuracy. Only compare when both models were built with the same feature
        # definitions and scored on the same test transition; otherwise the numbers
        # are not comparable and the new model is promoted with a note.
        cur_h, new_h = current_meta.get('holdout'), new_metadata.get('holdout')
        if current_meta.get('feature_version') != FEATURE_VERSION:
            print("  Current model was built with old/leaky features (or has no feature_version) "
                  "- replacing it")
            promote = True
        elif not cur_h or not new_h:
            print("  Cannot compare held-out scores (missing) - promoting the new model")
            promote = True
        elif cur_h['test_transition'] != new_h['test_transition']:
            print(f"  Held-out test changed ({cur_h['test_transition']} -> {new_h['test_transition']}); "
                  f"scores are not comparable - promoting the new model")
            promote = True
        else:
            current_acc, new_acc = cur_h['accuracy'], new_h['accuracy']
            if new_acc < current_acc + min_improvement:
                print(f"  New model held-out {new_acc:.2%} not better than current "
                      f"{current_acc:.2%} (+{min_improvement:.0%} threshold)")
                print(f"  KEEPING current model as 'latest'. New attempt archived at {version_dir}")
                promote = False
            else:
                print(f"  New model held-out {new_acc:.2%} beats current {current_acc:.2%} - promoting")
                promote = True

    if candidate_dir is not None:
        candidate_dir = Path(candidate_dir)
        if promote:
            for fname in model_files:
                src = candidate_dir / fname
                if src.exists():
                    shutil.copy(src, latest_dir / fname)
            shutil.rmtree(candidate_dir, ignore_errors=True)
            print(f"  '{state}' latest model updated (prior model archived at {version_dir.name})")
        else:
            rejected = latest_dir / f'rejected_{timestamp}'
            shutil.move(str(candidate_dir), str(rejected))
            print(f"  Kept current model as 'latest'; rejected candidate saved at {rejected.name}")
    elif promote:
        # Legacy call (models already written to 'latest'): nothing further to do.
        print(f"  '{state}' latest model updated (v_{timestamp} archived as prior version)")
    else:
        # Legacy call: restore the archived prior model into latest_dir.
        for fname in model_files:
            src = version_dir / fname
            if src.exists():
                shutil.copy(src, latest_dir / fname)
        print(f"  Restored prior model as 'latest' for {state}")

    return promote


def list_versions(state: str) -> list:
    """List all archived versions for a state, most recent first."""
    state_dir = MODELS_DIR / state
    if not state_dir.exists():
        return []
    versions = sorted(
        [d.name for d in state_dir.iterdir() if d.is_dir() and d.name.startswith('v_')],
        reverse=True
    )
    return versions


def get_version_metadata(state: str, version: str) -> dict:
    """Read metadata.json for a specific archived version."""
    path = MODELS_DIR / state / version / 'metadata.json'
    if not path.exists():
        raise FileNotFoundError(f"No metadata for {state}/{version}")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inspect model version history')
    parser.add_argument('state', help='State to inspect')
    args = parser.parse_args()

    versions = list_versions(args.state)
    if not versions:
        print(f"No archived versions for {args.state}")
    else:
        print(f"Version history for {args.state}:")
        for v in versions:
            meta = get_version_metadata(args.state, v)
            acc = meta.get('holdout_accuracy')
            label = 'held-out' if acc is not None else 'no held-out score (old model)'
            acc = acc if acc is not None else float('nan')
            print(f"  {v}: {acc:.2%} ({label})")