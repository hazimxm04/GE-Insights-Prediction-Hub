"""
Tests for what training SAVES and how models are promoted / validated:
  - the held-out score is time-ordered (train earlier, test the latest)
  - validation refuses old (leaky) models and elections at or before the training cutoff
  - promotion compares held-out scores only, and only when they are comparable
All offline: no data download, no real models touched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'mlops'))

import model_versioning
import validation
from backend.core.features import FEATURE_VERSION
from train_models import compute_holdout
from training_config import FEATURE_NAMES_TRAIN, TRAINING_CONFIGS


def _fake_training_frame():
    rng = np.random.RandomState(1)
    rows = []
    for a, b in [(2008, 2013), (2013, 2018), (2018, 2022)]:
        for _ in range(30):
            r = {f: float(rng.randn()) for f in FEATURE_NAMES_TRAIN}
            r.update(year_a=a, year_b=b, target_non_bn_won=int(rng.rand() > 0.6),
                     winner_coalition_a=str(rng.choice(['BN', 'PH'])))
            rows.append(r)
    return pd.DataFrame(rows)


def test_holdout_trains_on_earlier_and_tests_on_the_latest_transition():
    h = compute_holdout(_fake_training_frame(), TRAINING_CONFIGS['johor'])
    assert h['test_transition'] == '2018->2022'
    assert h['train_transitions'] == ['2008->2013', '2013->2018']
    assert h['seats'] == 30 and 0 <= h['accuracy'] <= 1
    assert h['baseline_never_opposition'] <= 30 and h['baseline_same_as_last'] <= 30


def test_holdout_is_none_with_a_single_transition():
    df = _fake_training_frame()
    assert compute_holdout(df[df.year_b == 2013], TRAINING_CONFIGS['johor']) is None


# ── validation guard ─────────────────────────────────────────────────────────
def test_validation_refuses_a_model_without_feature_version():
    assert 'leaky' in validation.model_problem({'training_cutoff_year': 2022}, 2026)


def test_validation_refuses_elections_at_or_before_the_cutoff():
    meta = {'feature_version': FEATURE_VERSION, 'training_cutoff_year': 2022}
    assert validation.model_problem(meta, 2022) is not None
    assert validation.model_problem(meta, 2018) is not None
    assert validation.model_problem(meta, 2026) is None


def test_validation_refuses_missing_metadata():
    assert validation.model_problem(None, 2026) is not None


# ── promotion gate ───────────────────────────────────────────────────────────
def _meta(acc, test='2018->2022', version=FEATURE_VERSION):
    m = {'models': {'ensemble': {'accuracy': 1.0}}}   # in-sample: must be ignored
    if version:
        m['feature_version'] = version
    if acc is not None:
        m['holdout'] = {'test_transition': test, 'accuracy': acc}
    return m


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(model_versioning, 'MODELS_DIR', tmp_path)
    (tmp_path / 'x').mkdir()
    return tmp_path / 'x'


def _promote(models_dir, current, new, **kw):
    (models_dir / 'metadata.json').write_text(json.dumps(current))
    return model_versioning.version_and_promote('x', new, **kw)


def test_old_leaky_model_is_always_replaced(models_dir):
    assert _promote(models_dir, _meta(0.99, version=None), _meta(0.50)) is True


def test_worse_heldout_score_on_same_test_is_not_promoted(models_dir):
    assert _promote(models_dir, _meta(0.85), _meta(0.80)) is False


def test_better_heldout_score_on_same_test_is_promoted(models_dir):
    assert _promote(models_dir, _meta(0.80), _meta(0.85)) is True


def test_training_accuracy_is_ignored_in_the_comparison(models_dir):
    # both have in-sample accuracy 1.0, so only the held-out score can decide
    assert _promote(models_dir, _meta(0.90), _meta(0.80)) is False


def test_different_test_transition_is_not_comparable_so_new_model_is_promoted(models_dir):
    assert _promote(models_dir, _meta(0.90, test='2018->2022'), _meta(0.60, test='2022->2026')) is True


# ── staged candidate flow (what train_models.py now does) ────────────────────
def _stage(models_dir, prior_meta, cand_meta):
    (models_dir / 'metadata.json').write_text(json.dumps(prior_meta))
    (models_dir / 'rf_model.pkl').write_bytes(b'PRIOR')
    cand = models_dir / '_candidate'
    cand.mkdir()
    (cand / 'metadata.json').write_text(json.dumps(cand_meta))
    (cand / 'rf_model.pkl').write_bytes(b'NEW')
    return cand


def test_staged_candidate_is_compared_with_the_prior_model_and_can_be_rejected(models_dir):
    cand = _stage(models_dir, _meta(0.90), _meta(0.80))
    assert model_versioning.version_and_promote('x', _meta(0.80), candidate_dir=cand) is False
    assert (models_dir / 'rf_model.pkl').read_bytes() == b'PRIOR'      # prior model kept
    assert not cand.exists() and any(p.name.startswith('rejected_') for p in models_dir.iterdir())


def test_staged_candidate_replaces_the_prior_model_when_better(models_dir):
    cand = _stage(models_dir, _meta(0.80), _meta(0.90))
    assert model_versioning.version_and_promote('x', _meta(0.90), candidate_dir=cand) is True
    assert (models_dir / 'rf_model.pkl').read_bytes() == b'NEW'
    assert json.loads((models_dir / 'metadata.json').read_text())['holdout']['accuracy'] == 0.90
    archived = [p for p in models_dir.iterdir() if p.name.startswith('v_')]
    assert archived and (archived[0] / 'rf_model.pkl').read_bytes() == b'PRIOR'   # prior archived
