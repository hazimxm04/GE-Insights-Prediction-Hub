"""
Frozen forecasts must stay frozen.

  * every folder under forecasts/ still matches the checksum in its manifest
    (this is the test that fails if someone edits a frozen file)
  * freeze_forecasts.py writes the expected columns, refuses to overwrite a folder,
    and refuses states whose election has already been held
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import freeze_forecasts as ff
from backend.core.states import STATES


def test_existing_frozen_forecasts_are_unchanged():
    problems = [p for f in ff.frozen_folders() for p in ff.verify_folder(f)]
    assert problems == []


def test_pending_states_are_the_ones_without_a_result():
    assert set(ff.pending_states()) == {s for s, v in STATES.items() if v['final_test'] is None}
    assert 'johor' not in ff.pending_states()


def test_refuses_a_state_that_already_voted():
    with pytest.raises(SystemExit):
        ff.freeze_state('johor', allow_dirty=True)


def _have_files(state):
    cfg = STATES[state]
    raw = ROOT / 'data' / 'raw'
    return ((ROOT / 'backend' / 'models' / state / 'rf_model.pkl').exists()
            and (raw / f'{cfg["file_stem"]}_ballots.parquet').exists())


@pytest.mark.skipif(not _have_files('selangor'), reason='needs the Selangor model and data')
def test_freeze_writes_receipt_and_cannot_be_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, 'FORECAST_DIR', tmp_path)
    folder = ff.freeze_state('selangor', allow_dirty=True, today='2000-01-01')

    df = pd.read_csv(folder / 'forecast.csv')
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))

    assert list(df.columns) == ff.FORECAST_COLUMNS
    assert len(df) == manifest['n_seats'] > 0
    assert df['seat_no'].is_unique
    assert df['probability'].between(0, 1).all()
    assert set(df['baseline_never_opposition']) == {0}
    assert set(df['baseline_same_as_last']) <= {0, 1}
    assert (df['predicted_opposition'] == (df['probability'] >= ff.CUTOFF).astype(int)).all()
    assert manifest['model_feature_version'] == manifest['feature_version']
    assert ff.verify_folder(folder) == []
    # text files are hashed with line endings normalised, binary files as they are
    meta = ROOT / 'backend' / 'models' / 'selangor' / 'metadata.json'
    assert manifest['model_files_sha256']['metadata.json'] == ff.sha256_file(meta, text=True)
    assert manifest['model_files_sha256']['rf_model.pkl'] == ff.sha256_file(ROOT / 'backend' / 'models' / 'selangor' / 'rf_model.pkl')

    # a second freeze for the same state and day must not overwrite
    with pytest.raises(SystemExit):
        ff.freeze_state('selangor', allow_dirty=True, today='2000-01-01')

    # editing a frozen file must be detected
    text = (folder / 'forecast.csv').read_text(encoding='utf-8')
    (folder / 'forecast.csv').write_text(text.replace('0.', '0.9', 1), encoding='utf-8', newline='\n')
    assert any('changed after it was frozen' in p for p in ff.verify_folder(folder))


def test_text_and_binary_files_are_told_apart():
    assert ff.is_text('metadata.json') and ff.is_text('ethnicity_x.csv')
    assert not ff.is_text('rf_model.pkl') and not ff.is_text('johor_ballots.parquet')


def test_checksum_ignores_windows_line_endings(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_bytes(b'x,y\n1,2\n')
    b.write_bytes(b'x,y\r\n1,2\r\n')
    assert ff.sha256_file(a, text=True) == ff.sha256_file(b, text=True)