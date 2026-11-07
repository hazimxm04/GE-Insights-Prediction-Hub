import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.core import frozen_forecast as ff


def test_side_changes_sums_bn_and_pn_as_one_group_and_lists_the_flips():
    df = pd.DataFrame({'seat_name': ['A', 'B', 'C', 'D'],
                       'probability': [0.9, 0.6, 0.4, 0.1],
                       'baseline_same_as_last': [1, 0, 1, 0]})          # last: H, BN/PN, H, BN/PN
    ch = ff.side_changes(df)
    assert (ch['last_harapan'], ch['now_harapan']) == (2, 2)
    assert list(ch['to_harapan']['seat_name']) == ['B']                 # BN/PN -> Harapan
    assert list(ch['to_bnpn']['seat_name']) == ['C']                    # Harapan -> BN/PN
    assert ch['delta_harapan'] == 0 and ch['delta_bnpn'] == 0


def test_side_changes_uses_the_probability_column_it_is_given():
    df = pd.DataFrame({'seat_name': ['A'], 'probability': [0.8], 'adjusted': [0.3], 'baseline_same_as_last': [1]})
    assert ff.side_changes(df)['to_bnpn'].empty
    assert list(ff.side_changes(df, 'adjusted')['to_bnpn']['seat_name']) == ['A']


def test_counts_always_add_up():
    df = pd.DataFrame({'seat_name': list('ABCDE'), 'probability': [.7, .2, .55, .45, .9],
                       'baseline_same_as_last': [0, 0, 1, 1, 0]})
    ch = ff.side_changes(df)
    assert ch['now_harapan'] + ch['now_bnpn'] == ch['total'] == 5
    assert ch['delta_harapan'] == len(ch['to_harapan']) - len(ch['to_bnpn']) == -ch['delta_bnpn']


def test_result_changes_johor_matches_the_known_2022_and_2026_results():
    df = pd.read_csv(ROOT / 'backend/models/johor/validation_2026.csv')
    ch = ff.result_changes(df)
    assert (ch['last_harapan'], ch['now_harapan']) == (12, 8)              # PH won 12 seats in 2022, 8 in 2026
    assert ch['total'] == 56 and ch['delta_harapan'] == -4
    assert list(ch['to_harapan']['seat']) == ['N.41 Puteri Wangsa']       # the only seat Harapan gained
    assert len(ch['to_bnpn']) == 5


def test_result_changes_ns_counts_add_up():
    df = pd.read_csv(ROOT / 'backend/models/neg_sembilan/validation_2026.csv')
    ch = ff.result_changes(df)
    assert ch['now_harapan'] == int(df['actual_non_bn'].sum())
    assert ch['delta_harapan'] == len(ch['to_harapan']) - len(ch['to_bnpn'])
