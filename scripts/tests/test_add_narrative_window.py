import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.scoring import score_quality as q
import add_narrative_window as w


def scored():
    base = {c: 0.0 for c in q.SCORE_FIELDS}
    rows = []
    for i, (day, isl, ok) in enumerate([('2026-10-02', .2, True), ('2026-10-20', .4, True),
                                        ('2026-10-31', .9, False), ('2026-11-01', .8, True)]):
        r = dict(base, islam_threat=isl, harapan_sentiment=.5, bn_sentiment=.1, pn_sentiment=-.1,
                 published=f'{day}T10:00:00Z', scoring_ok=ok)
        if not ok:
            r.update(q.failed_scores())
        rows.append(r)
    return pd.DataFrame(rows)


def test_window_uses_dates_inclusive_and_skips_failed():
    row = q.window_row(scored(), '2026-10-01', '2026-10-31', 'Oct')
    assert row['n_articles'] == 2 and row['islam_threat'] == pytest.approx(.3)
    assert row['harapan_sentiment'] == pytest.approx(.5)


def test_empty_window_raises():
    with pytest.raises(ValueError):
        q.window_row(scored(), '2025-01-01', '2025-01-31', 'old')


def test_add_window_appends_and_replaces_same_label(tmp_path):
    s, o = tmp_path / 's.csv', tmp_path / 'w.csv'
    scored().to_csv(s, index=False)
    w.add_window(s, o, 'Oct', '2026-10-01', '2026-10-31')
    w.add_window(s, o, 'Nov', '2026-11-01', '2026-11-30')
    w.add_window(s, o, 'Oct', '2026-10-01', '2026-10-31')       # same label again
    df = pd.read_csv(o)
    assert list(df['label']) == ['Oct', 'Nov'] and len(df) == 2
