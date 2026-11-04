"""Failed scoring must be missing, never zero (sentiment/scoring/score_quality.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sentiment.scoring import score_quality as q

GOOD = """BN: 0.2
Harapan: -0.1
PN: 0.0
Tension: 0.3
Islam_threat: 0.8
Malay_unity: 0.4
Cost_living: 0.1
Corruption: 0.0
Govt_performance: 0.5"""


def test_parse_good_reply():
    s = q.parse_scores(GOOD)
    assert s['islam_threat'] == 0.8 and s['harapan_sentiment'] == -0.1
    assert q.is_ok(s)


def test_parse_clamps_out_of_range():
    s = q.parse_scores(GOOD.replace('Islam_threat: 0.8', 'Islam_threat: 7').replace('BN: 0.2', 'BN: -3'))
    assert s['islam_threat'] == 1.0 and s['bn_sentiment'] == -1.0


def test_missing_label_is_failure_not_zero():
    assert q.parse_scores(GOOD.replace('Corruption: 0.0\n', '')) is None
    assert q.parse_scores('') is None and q.parse_scores(None) is None


def test_failed_scores_are_nan():
    f = q.failed_scores()
    assert set(f) == set(q.SCORE_FIELDS) and not q.is_ok(f)


def _df():
    ok = {**q.parse_scores(GOOD), 'scoring_ok': True}
    ok2 = {**q.parse_scores(GOOD.replace('Islam_threat: 0.8', 'Islam_threat: 0.4')), 'scoring_ok': True}
    bad = {**q.failed_scores(), 'scoring_ok': False}
    return pd.DataFrame([ok, ok2, bad])


def test_means_skip_failed_rows():
    m = q.theme_means(_df())
    assert m['islam_threat'] == pytest.approx(0.6)       # (0.8 + 0.4) / 2, not / 3


def test_failure_counts_and_usable():
    df = _df()
    assert q.failure_counts(df) == (1, 3) and len(q.usable(df)) == 2


def test_old_tables_without_flag_still_work():
    df = _df().drop(columns='scoring_ok').iloc[:2]
    assert len(q.usable(df)) == 2 and q.theme_means(df)['malay_unity'] == pytest.approx(0.4)


def test_flag_read_back_from_csv_strings(tmp_path):
    p = tmp_path / 's.csv'
    _df().to_csv(p, index=False)
    assert len(q.usable(pd.read_csv(p))) == 2


def test_all_failed_raises_instead_of_zeros():
    df = pd.DataFrame([{**q.failed_scores(), 'scoring_ok': False}] * 3)
    with pytest.raises(ValueError):
        q.theme_means(df)


def test_scorer_returns_nan_on_api_error(monkeypatch):
    pytest.importorskip('groq')
    monkeypatch.setenv('GROQ_API_KEY', 'dummy')
    from sentiment.scoring import sentiment_scorer as ss

    class Boom:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError('timeout')
    monkeypatch.setattr(ss, 'client', Boom)
    assert not q.is_ok(ss.score_article_all('headline'))


def test_classify_api_error():
    daily = "Error code: 429 - Rate limit reached ... on tokens per day (TPD): Limit 200000"
    minute = "Error code: 429 - Rate limit reached ... on tokens per minute (TPM): Limit 8000"
    assert q.classify_api_error(daily) == 'daily'
    assert q.classify_api_error(minute) == 'minute'
    assert q.classify_api_error("Error code: 403 - Access denied") == 'other'
    assert q.classify_api_error("") == 'other'


def test_resume_skips_only_successful_titles():
    import score_news_file as sf
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raw = pd.DataFrame({'title': ['a', 'b', 'c']})
    done = pd.DataFrame({'title': ['a', 'b'], 'scoring_ok': [True, False]})
    assert list(sf.pending(raw, done)['title']) == ['b', 'c']      # failed 'b' is retried
    assert list(sf.pending(raw, None)['title']) == ['a', 'b', 'c']
