"""
Single definition of how election data becomes model features.

Training, validation and (later) the live predictor must all import from here,
so there is only ONE definition of every feature.

Leakage rule: to predict election B, features use only election A and the
election just before it (LOOKBACK = 2 elections). Elections before MIN_YEAR are
not used because the number of state seats was different then.
"""
import re

import numpy as np
import pandas as pd

LOOKBACK = 2
MIN_YEAR = 2004

# Opposition coalitions treated as "non-BN" for the binary target.
# Note: PN is intentionally NOT distinguished from BN here. Explored
# a 3-class BN/Pakatan/PN split (see docs/multiclass_analysis.md) and
# found the BN-vs-PN boundary is not yet learnable: PN only emerged in
# 2020, so no training transition has PN as a "previous" winner, and
# only the most recent transition per state has any PN examples at
# all (2-26 seats depending on state). Binary framing (Pakatan vs
# everyone else) is answerable well with current data; BN-vs-PN is not.
OPPOSITION = {'Harapan', 'PH', 'PR'}


COALITION_RENAME = {
    'Harapan': 'PH',
    'Pakatan Harapan': 'PH',
    'Pakatan Rakyat': 'PR',
    'PERIKATAN': 'PN',
    'Perikatan Nasional': 'PN',
}


def seat_no(name):
    """'N.07 Bukit Kepong' -> 'N.07'.  Seats keep their number when they are
    renamed between elections, so we match seats on the number, not the name.
    Falls back to the full string when there is no 'N.xx' code."""
    m = re.match(r'^\s*([A-Za-z]\.?\s*\d+)', str(name))
    return m.group(1).replace(' ', '') if m else str(name)


def _year(df):
    return pd.to_datetime(df['date']).dt.year


def previous_election_year(stats, year_a):
    """The state's own general-election year immediately before year_a (or None)."""
    prior = sorted(y for y in _year(stats).unique() if y < year_a)
    return int(prior[-1]) if prior else None


def _winners(ballots, year):
    b = ballots[_year(ballots) == year]
    w = b[b['result'] == 'won'][['seat', 'coalition']].copy()
    w['seat_no'] = w['seat'].map(seat_no)
    w['coalition'] = w['coalition'].replace(COALITION_RENAME)
    return w.drop_duplicates('seat_no')


def _stats(stats, year, suffix):
    s = stats[_year(stats) == year][['seat', 'majority', 'votes_valid', 'voters_total', 'n_candidates']].copy()
    s['seat_no'] = s['seat'].map(seat_no)
    s = s.drop(columns='seat').drop_duplicates('seat_no')
    return s.rename(columns={
        'majority': f'majority_{suffix}',
        'votes_valid': f'votes_total_{suffix}',
        'voters_total': f'voters_total_{suffix}',
        'n_candidates': f'n_candidates_{suffix}',
    })


def build_transition(ballots, stats, year_a, year_b, min_year=MIN_YEAR):
    """
    Build one training row per seat for predicting election B.

    RULE (no leakage): features may use only election A and the election just
    before A (LOOKBACK = 2 elections). Nothing from election B is used as a
    feature; B supplies only the target (who won).

      majority_change      = majority in A  - majority in the election before A
      turnout_change       = turnout in A   - turnout in the election before A
      incumbent_held       = the winner of A is the same coalition as the winner before A
      log_voters           = log(electorate in A)
      majority_perc_change = majority_change / electorate in A
      n_candidates_b       = candidates in A  (legacy column name; now means election A)

    Seats are matched on their number (N.07), not their name, because some
    seats were renamed between 2013 and 2018. Elections before `min_year`
    are not used: the number of seats was different then.
    """
    if year_a >= year_b:
        raise ValueError(f'year_a ({year_a}) must be earlier than year_b ({year_b})')
    year_p = previous_election_year(stats, year_a)
    if year_p is None or year_p < min_year:
        raise ValueError(
            f'No usable election before {year_a} (found {year_p}); '
            f'need one in {min_year} or later for the change features.')

    wa = _winners(ballots, year_a).rename(columns={'coalition': 'winner_coalition_a', 'seat': 'seat_a'})
    wb = _winners(ballots, year_b).rename(columns={'coalition': 'winner_coalition_b', 'seat': 'seat'})
    wp = _winners(ballots, year_p)[['seat_no', 'coalition']].rename(columns={'coalition': 'winner_coalition_p'})

    d = wa[['seat_no', 'winner_coalition_a']].merge(wb, on='seat_no')
    d = d.merge(wp, on='seat_no', how='left')
    d = d.merge(_stats(stats, year_a, 'a'), on='seat_no', how='left')
    d = d.merge(_stats(stats, year_p, 'p'), on='seat_no', how='left')

    d['majority_change']      = d['majority_a'] - d['majority_p']
    d['turnout_a']            = d['votes_total_a'] / d['voters_total_a']
    d['turnout_p']            = d['votes_total_p'] / d['voters_total_p']
    d['turnout_change']       = d['turnout_a'] - d['turnout_p']
    d['incumbent_held']       = ((d['winner_coalition_a'] == d['winner_coalition_p'])
                                 & d['winner_coalition_p'].notna()).astype(int)
    d['log_voters']           = np.log(d['voters_total_a'])
    d['majority_perc_change'] = d['majority_change'] / d['voters_total_a'].replace(0, np.nan)
    d['n_candidates_b']       = d['n_candidates_a'].fillna(3)   # legacy name: candidates in A

    d['year_prev'] = year_p
    d['target_non_bn_won'] = d['winner_coalition_b'].isin(OPPOSITION).astype(int)
    return d