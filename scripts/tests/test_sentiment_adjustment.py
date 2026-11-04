import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.core import sentiment_adjustment as sa

REF = {'islam_threat': .10, 'malay_unity': .30, 'cost_living': .10, 'corruption': .10,
       'govt_performance': .10, 'n_articles': 400}


def cur(**kw):
    d = dict(REF); d.update(kw); return d


def demo(malay, chinese, indian, young=.3, older=.3):
    return pd.Series({'malay_pct': malay, 'chinese_pct': chinese, 'indian_pct': indian,
                      'young_malay_pct': young, 'older_malay_pct': older})


def test_shares_sum_to_one():
    s = sa.segment_shares(demo(.6, .3, .05))
    assert sum(s.values()) == pytest.approx(1.0) and s['other'] == pytest.approx(.05, abs=1e-6)


def test_islam_threat_moves_malay_seat_to_bnpn_and_chinese_seat_to_harapan():
    ch = sa.theme_change(cur(islam_threat=.20), REF)
    malay = sa.seat_shift(sa.segment_shares(demo(.85, .08, .03)), ch, False)
    chinese = sa.seat_shift(sa.segment_shares(demo(.2, .7, .05)), ch, False)
    assert malay < 0 < chinese


def test_cost_living_hurts_whoever_holds_the_seat_both_ways():
    ch = sa.theme_change(cur(cost_living=.20), REF)
    s = sa.segment_shares(demo(.5, .35, .1))
    assert sa.seat_shift(s, ch, held_by_opposition=True) < 0    # opposition-held seat loses support
    assert sa.seat_shift(s, ch, held_by_opposition=False) > 0   # BN/PN-held seat loses support


def test_no_change_no_shift():
    assert sa.seat_shift(sa.segment_shares(demo(.6, .3, .05)), sa.theme_change(cur(), REF), False) == 0


def test_govt_performance_has_no_effect():
    ch = sa.theme_change(cur(govt_performance=.9), REF)
    assert sa.seat_shift(sa.segment_shares(demo(.6, .3, .05)), ch, False) == 0


def test_shift_is_capped():
    ch = {t: 5.0 for t in sa.THEMES}
    assert abs(sa.seat_shift(sa.segment_shares(demo(.9, .05, .03)), ch, False)) == sa.MAX_ABS_SHIFT


def test_too_few_headlines_means_no_adjustment():
    f = pd.DataFrame({'seat_no': ['N.01'], 'probability': [.4], 'baseline_same_as_last': [0]})
    d = pd.DataFrame({'seat': ['N.01 A'], 'malay_pct': [.8], 'chinese_pct': [.1], 'indian_pct': [.05],
                      'young_malay_pct': [.3], 'older_malay_pct': [.3]})
    out, msg = sa.adjust_forecast(f, d, dict(cur(islam_threat=.5), n_articles=211), REF)
    assert out['sentiment_shift_logit'][0] == 0 and 'Not enough headlines' in msg
    assert out['sentiment_probability'][0] == .4


def test_adjust_forecast_leaves_input_alone_and_flags_flips():
    f = pd.DataFrame({'seat_no': ['N.01', 'N.02'], 'probability': [.49, .0], 'baseline_same_as_last': [0, 0]})
    d = pd.DataFrame({'seat': ['N.01 A', 'N.02 B'], 'malay_pct': [.2, .9], 'chinese_pct': [.7, .05],
                      'indian_pct': [.05, .03], 'young_malay_pct': [.1, .3], 'older_malay_pct': [.1, .3]})
    before = f.copy()
    out, msg = sa.adjust_forecast(f, d, cur(islam_threat=.25, malay_unity=.45), REF)
    pd.testing.assert_frame_equal(f, before)
    assert out.loc[0, 'sentiment_probability'] > .49
    assert bool(out.loc[0, 'sentiment_flips_call'])           # .49 -> above .5
    assert 0 <= out.loc[1, 'sentiment_probability'] < .05     # frozen 0.0 hardly moves


def test_assumption_table_lists_every_theme():
    t = sa.assumption_table()
    assert set(sa.THEMES) <= set(t['theme'])


def test_load_windows(tmp_path):
    p = tmp_path / 'w.csv'
    assert sa.load_windows(p) is None
    rows = [dict(REF, date='2026-10-01', label='a'), dict(REF, date='2026-11-01', label='b', islam_threat=.2)]
    pd.DataFrame(rows).to_csv(p, index=False)
    cur_w, ref_w = sa.load_windows(p)
    assert cur_w['label'] == 'b' and ref_w['label'] == 'a' and cur_w['islam_threat'] == .2
    pd.DataFrame(rows[:1]).to_csv(p, index=False)
    assert sa.load_windows(p) is None


def coal(h, b, p):
    return dict(REF, harapan_sentiment=h, bn_sentiment=b, pn_sentiment=p)


def test_coalition_lean_sign():
    assert sa.coalition_lean(coal(.4, 0, 0)) == pytest.approx(.4)
    assert sa.coalition_lean(coal(0, .2, .4)) == pytest.approx(-.3)
    assert sa.coalition_lean(REF) is None


def test_lean_moves_every_seat_toward_the_favoured_side():
    s = sa.segment_shares(demo(.6, .3, .05))
    zero = {t: 0.0 for t in sa.THEMES}
    assert sa.seat_shift(s, zero, False, lean_delta=.2) > 0      # Harapan coverage improves
    assert sa.seat_shift(s, zero, True, lean_delta=-.2) < 0      # BN/PN coverage improves
    assert sa.lean_change(coal(.4, 0, 0), coal(.1, 0, 0)) == pytest.approx(.3)
    assert sa.lean_change(REF, REF) == 0.0                       # missing columns: channel off


def test_symmetry_each_theme_helps_and_hurts_both_sides():
    """Same-size changes should not push every seat one way."""
    d = demo(.55, .3, .08)
    s = sa.segment_shares(d)
    for theme in ['cost_living', 'corruption']:
        ch = {t: 0.0 for t in sa.THEMES}; ch[theme] = .1
        assert sa.seat_shift(s, ch, True) < 0 < sa.seat_shift(s, ch, False)


def test_lean_is_smaller_in_older_malay_seats_than_in_mixed_seats():
    zero = {t: 0.0 for t in sa.THEMES}
    rural = sa.seat_shift(sa.segment_shares(demo(.9, .05, .03, young=.1, older=.5)), zero, False, lean_delta=.2)
    mixed = sa.seat_shift(sa.segment_shares(demo(.3, .55, .1, young=.1, older=.1)), zero, False, lean_delta=.2)
    assert 0 < rural < mixed


def test_update_summary(tmp_path):
    p = tmp_path / 'w.csv'
    assert sa.update_summary(p) is None
    rows = [dict(REF, date='2026-09-30', label='Sep 2026', start='2026-09-01'),
            dict(REF, date='2026-10-31', label='Oct 2026', start='2026-10-01', n_articles=312)]
    pd.DataFrame(rows).to_csv(p, index=False)
    assert sa.update_summary(p) == 'Oct 2026: news 2026-10-01 to 2026-10-31, 312 headlines scored'