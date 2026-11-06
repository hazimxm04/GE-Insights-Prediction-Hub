import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentiment.scoring import score_quality as q
import refresh_sentiment as r

TODAY = date(2026, 11, 4)


def raw_rows(days_ago):
    return pd.DataFrame({'title': [f'h{d}-{i}' for d in days_ago for i in range(2)],
                         'published': [(pd.Timestamp(TODAY, tz='UTC') - pd.Timedelta(days=d)).strftime('%a, %d %b %Y 08:00:00 GMT')
                                       for d in days_ago for _ in range(2)]})


def fake_score(chunk):
    out = chunk.copy()
    for c in q.SCORE_FIELDS:
        out[c] = 0.1
    out['scoring_ok'] = True
    return out


def test_rolling_dates_are_adjacent_30_day_windows():
    cs, ce, rs, re_ = r.rolling_dates(TODAY)
    assert (ce - cs).days == 29 and (re_ - rs).days == 29
    assert (cs - re_).days == 1 and ce == TODAY


def test_pick_new_newest_first_capped_and_skips_done():
    raw = raw_rows([0, 1, 5, 40, 90])
    pool = fake_score(raw.iloc[:1])                      # h0-0 already scored
    todo = r.pick_new(raw, pool, 3, TODAY)
    assert len(todo) == 3
    assert 'h0-0' not in set(todo['title'])
    assert not todo['title'].str.startswith('h90').any()     # outside both windows
    assert todo.iloc[0]['title'] == 'h0-1'


def test_pick_new_retries_failed_rows():
    raw = raw_rows([1])
    pool = fake_score(raw)
    pool.loc[0, 'scoring_ok'] = False
    pool.loc[0, 'bn_sentiment'] = float('nan')
    assert 'h1-0' in set(r.pick_new(raw, pool, 10, TODAY)['title'])


def test_refresh_end_to_end_and_state_rows_kept(tmp_path):
    raw = raw_rows(list(range(0, 55)))
    windows = tmp_path / 'w.csv'
    pd.DataFrame([{'date': '2023-08-01', 'label': 'Pre-NS', 'start': '2023-07-01', 'state': 'neg_sembilan',
                   'n_articles': 400}]).to_csv(windows, index=False)
    out = r.refresh(TODAY, max_new=1000, scrape=True, scrape_fn=lambda a, b: raw, score_fn=fake_score,
                    raw_path=tmp_path / 'raw.csv', pool_path=tmp_path / 'pool.csv', windows_path=windows)
    assert [l for l, _ in out['windows']] == [r.REFERENCE_LABEL, r.CURRENT_LABEL]
    w = pd.read_csv(windows)
    assert set(w['label']) == {'Pre-NS', r.REFERENCE_LABEL, r.CURRENT_LABEL}
    nat = w[w['state'].isna()].sort_values('date')
    assert list(nat['label']) == [r.REFERENCE_LABEL, r.CURRENT_LABEL]


def test_cap_limits_new_scoring_and_second_run_adds_more(tmp_path):
    raw = raw_rows(list(range(0, 55)))
    kw = dict(scrape=True, scrape_fn=lambda a, b: raw, score_fn=fake_score, raw_path=tmp_path / 'raw.csv',
              pool_path=tmp_path / 'pool.csv', windows_path=tmp_path / 'w.csv')
    r.refresh(TODAY, max_new=40, **kw)
    assert len(q.usable(pd.read_csv(tmp_path / 'pool.csv'))) == 40
    r.refresh(TODAY, max_new=40, **kw)
    assert len(q.usable(pd.read_csv(tmp_path / 'pool.csv'))) == 80


def test_daily_limit_stops_cleanly(tmp_path):
    raw = raw_rows(list(range(0, 55)))
    calls = []

    def limited(chunk):
        calls.append(1)
        return fake_score(chunk.iloc[:5]) if len(calls) == 2 else fake_score(chunk)

    out = r.refresh(TODAY, max_new=1000, scrape=True, scrape_fn=lambda a, b: raw, score_fn=limited,
                    raw_path=tmp_path / 'raw.csv', pool_path=tmp_path / 'pool.csv', windows_path=tmp_path / 'w.csv')
    assert out['stopped'] and len(calls) == 2
    assert len(q.usable(pd.read_csv(tmp_path / 'pool.csv'))) == 25


def test_seed_from_existing_scored_windows(tmp_path):
    raw = raw_rows([2, 3])
    fake_score(raw).to_csv(tmp_path / 'scored_window_Oct_2026.csv', index=False)
    pool = r.seed_pool(tmp_path / 'pool.csv', tmp_path)
    assert len(pool) == 4


def test_state_end_date_is_the_day_before_polling():
    assert r.state_end_date('johor') == date(2026, 7, 10)
    assert r.state_end_date('neg_sembilan') == date(2026, 7, 31)
    assert r.state_end_date('johor', '2026-07-12') == date(2026, 7, 11)


def test_johor_and_ns_windows_match_the_plan():
    cs, ce, rs, re_ = r.rolling_dates(r.state_end_date('johor'))
    assert (str(rs), str(re_), str(cs), str(ce)) == ('2026-05-12', '2026-06-10', '2026-06-11', '2026-07-10')
    cs, ce, rs, re_ = r.rolling_dates(r.state_end_date('neg_sembilan'))
    assert (str(rs), str(re_), str(cs), str(ce)) == ('2026-06-02', '2026-07-01', '2026-07-02', '2026-07-31')


def test_state_windows_replace_only_that_state_and_keep_national(tmp_path):
    end = r.state_end_date('johor')
    pool = fake_score(pd.DataFrame({
        'title': [f's{i}' for i in range(60)],
        'published': [(pd.Timestamp(end, tz='UTC') - pd.Timedelta(days=i)).strftime('%a, %d %b %Y 08:00:00 GMT')
                      for i in range(60)]}))
    windows = tmp_path / 'w.csv'
    pd.DataFrame([
        {'date': '2026-11-06', 'label': 'Latest 30 days', 'start': '2026-10-08', 'state': '', 'n_articles': 392},
        {'date': '2026-10-07', 'label': 'Previous 30 days', 'start': '2026-09-08', 'state': '', 'n_articles': 384},
        {'date': '2026-07-31', 'label': 'old NS', 'start': '2026-07-02', 'state': 'neg_sembilan', 'n_articles': 5},
        {'date': '2026-01-01', 'label': 'stale johor', 'start': '2025-12-01', 'state': 'johor', 'n_articles': 1},
    ]).to_csv(windows, index=False)
    counts = r.rebuild_windows(pool, windows, end, state='johor')
    out = pd.read_csv(windows).fillna({'state': ''})
    johor = out[out['state'] == 'johor'].sort_values('date')
    assert list(johor['label']) == [r.STATE_REFERENCE_LABEL, r.STATE_CURRENT_LABEL]
    assert str(johor.iloc[-1]['date']) == '2026-07-10'
    assert 'stale johor' not in set(out['label'])
    assert (out['state'] == '').sum() == 2 and (out['state'] == 'neg_sembilan').sum() == 1
    assert [n for _, n in counts] == [30, 30]


def test_overlapping_state_windows_score_each_headline_once():
    # Johor's last window (11 Jun - 10 Jul) and NS's reference window (2 Jun - 1 Jul) share 11 Jun - 1 Jul.
    raw = pd.DataFrame({'title': ['shared'],
                        'published': [pd.Timestamp('2026-06-20', tz='UTC').strftime('%a, %d %b %Y 08:00:00 GMT')]})
    johor = r.pick_new(raw, pd.DataFrame(), 10, r.state_end_date('johor'))
    ns = r.pick_new(raw, fake_score(johor), 10, r.state_end_date('neg_sembilan'))
    assert len(johor) == 1 and ns.empty