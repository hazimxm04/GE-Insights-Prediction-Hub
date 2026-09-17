import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "backend" / "models"


def version_and_promote(state: str, new_metadata: dict, min_improvement: float = 0.0) -> bool:
    """
    Archive the current 'latest' model under a timestamped folder,
    then decide whether the NEW model should become 'latest' based
    on validated accuracy (not training accuracy -- see note below).

    KNOWN LIMITATION: if the current 'latest' model has a
    validated_accuracy (from validation.py) but the newly-trained
    model does not yet (validation.py hasn't been run on it), this
    function compares validated (trustworthy) against training
    (potentially misleading) accuracy -- an apples-to-oranges
    comparison. Correct workflow: run validation.py immediately
    after train_models.py, before training again, so
    validated_accuracy is populated before the NEXT comparison runs.

    Args:
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

        # Prefer validated accuracy if present in metadata, else fall
        # back to training accuracy with an explicit warning.
        current_acc = current_meta.get('validated_accuracy',
                                        current_meta['models']['ensemble']['accuracy'])
        new_acc = new_metadata.get('validated_accuracy',
                                    new_metadata['models']['ensemble']['accuracy'])

        if 'validated_accuracy' not in current_meta or 'validated_accuracy' not in new_metadata:
            print(f"  WARNING: comparing on TRAINING accuracy, not validated "
                  f"accuracy -- this can be misleading (see docstring)")

        if new_acc < current_acc + min_improvement:
            print(f"  New model ({new_acc:.2%}) not better than current "
                  f"({current_acc:.2%} + {min_improvement:.0%} threshold)")
            print(f"  KEEPING current model as 'latest'. New attempt archived "
                  f"at {version_dir}")
            promote = False
        else:
            print(f"  New model ({new_acc:.2%}) beats current ({current_acc:.2%}) "
                  f"-- promoting")
            promote = True

    if promote:
        # The caller is responsible for having already written the NEW
        # model files directly to latest_dir (train_models.py does this).
        # We've already archived what was there before overwriting, so
        # nothing further to do here except confirm.
        print(f"  '{state}' latest model updated (v_{timestamp} archived as prior version)")
    else:
        # Restore the archived prior model back into latest_dir, since
        # train_models.py already overwrote it with the new (rejected) one.
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
            acc = meta.get('validated_accuracy', meta['models']['ensemble']['accuracy'])
            label = "validated" if 'validated_accuracy' in meta else "training (unvalidated)"
            print(f"  {v}: {acc:.2%} ({label})")