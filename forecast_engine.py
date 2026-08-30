"""
forecast_engine.py
===================
v3: Weighted-recency historical loyalty using ACTUAL vote SHARE
    trend across 2008/2013/2018/2023(or 2021), not just binary
    win/loss. Recent elections weighted more heavily than older
    ones, capturing whether a seat's Harapan support is
    strengthening, stable, or eroding over time.

    Matches seats by NUMBER (e.g. "N.21") not full name, since
    names change across elections (e.g. "Chempaka" -> "Pandan
    Indah") even when the underlying seat/boundary is stable.

Combines:
  1. Weighted-recency vote share trend (2008-2023/2021)
  2. Johor-trained ML model transfer (structural + sentiment +
     economic + demographic patterns)
  3. Current blue-wave sentiment pressure

Also computes a simplified OOD proxy based on vote share volatility.
"""

import re
import numpy as np
import pandas as pd

# Recency weights: most recent election counts most.
# 2023/2021 : 2018 : 2013 : 2008 = 0.40 : 0.30 : 0.20 : 0.10
RECENCY_WEIGHTS = {
    'recent': 0.40,
    2018:     0.30,
    2013:     0.20,
    2008:     0.10,
}


def extract_seat_number(seat_name: str) -> str:
    match = re.match(r'(N\.\d+)', str(seat_name))
    return match.group(1) if match else None


def get_historical_ph_vote_shares(ballots: pd.DataFrame, df: pd.DataFrame,
                                    seat_col: str = 'seat',
                                    recent_year_cutoff: int = 2024) -> tuple:
    """
    Build historical PH vote SHARE (not just win/loss) for
    2008, 2013, 2018, and the most recent completed election,
    matched by seat NUMBER to handle name changes across years.

    Returns (df_with_shares, recent_year_used)
    """
    PH_COALITIONS = {'Harapan', 'PH', 'PR'}

    ballots = ballots.copy()
    ballots['seat_num'] = ballots['seat'].apply(extract_seat_number)
    df = df.copy()
    df['seat_num'] = df[seat_col].apply(extract_seat_number)

    for year in [2008, 2013, 2018]:
        b_year = ballots[pd.to_datetime(ballots['date']).dt.year == year]
        ph_share = b_year[
            (b_year['coalition'].isin(PH_COALITIONS)) &
            (b_year['result'] == 'won')
        ][['seat_num', 'votes_perc']].copy()
        ph_share.columns = ['seat_num', f'ph_share_{year}']
        df = df.merge(ph_share, on='seat_num', how='left')

    all_years = sorted(pd.to_datetime(ballots['date']).dt.year.unique())
    recent_candidates = [y for y in all_years if 2018 < y < recent_year_cutoff]
    recent_year = recent_candidates[-1] if recent_candidates else max(all_years)

    b_recent = ballots[pd.to_datetime(ballots['date']).dt.year == recent_year]
    ph_recent = b_recent[
        (b_recent['coalition'].isin(PH_COALITIONS)) &
        (b_recent['result'] == 'won')
    ][['seat_num', 'votes_perc']].copy()
    ph_recent.columns = ['seat_num', 'ph_share_recent']
    df = df.merge(ph_recent, on='seat_num', how='left')

    df = df.drop(columns=['seat_num'], errors='ignore')
    return df, recent_year


def compute_ph_loyalty_weighted(row) -> tuple:
    """
    Weighted-recency loyalty score using actual vote share trend.

    Returns (weighted_avg_share, trend, volatility, n_data_points)

    weighted_avg_share: 0-1, recency-weighted average PH vote share
    trend: last_share - first_share (positive = strengthening)
    volatility: std deviation of available shares (normalized)
    """
    points = {
        2008:     row.get('ph_share_2008'),
        2013:     row.get('ph_share_2013'),
        2018:     row.get('ph_share_2018'),
        'recent': row.get('ph_share_recent'),
    }
    valid = {k: v for k, v in points.items() if v is not None and not pd.isna(v)}

    if len(valid) == 0:
        return 0.5, 0.0, 0.0, 0

    # Weighted average, renormalizing weights to only valid years
    total_weight = sum(RECENCY_WEIGHTS[k] for k in valid)
    weighted_avg = sum(
        (v / 100) * RECENCY_WEIGHTS[k] for k, v in valid.items()
    ) / total_weight

    # Trend: most recent available vs earliest available
    ordered_keys = sorted(valid.keys(), key=lambda k: (
        9999 if k == 'recent' else k
    ))
    if len(ordered_keys) >= 2:
        trend = (valid[ordered_keys[-1]] - valid[ordered_keys[0]]) / 100
    else:
        trend = 0.0

    # Volatility: normalized std deviation of vote shares
    shares = list(valid.values())
    volatility = (np.std(shares) / 100) if len(shares) > 1 else 0.0

    return weighted_avg, trend, volatility, len(valid)


def compute_forecast_score(
    row,
    johor_model_prob: float,
    malay_unity: float = 0.364,
    blend_weight: float = 0.5,
) -> tuple:
    """
    Combine weighted-recency loyalty + Johor ML transfer.
    Trend adjusts the loyalty score up/down: a seat trending
    toward PH gets a boost beyond its raw average, and vice versa.
    """
    weighted_avg, trend, volatility, n_points = compute_ph_loyalty_weighted(row)

    # Trend adjustment: strengthening seats get a boost,
    # eroding seats get a penalty, scaled modestly (max +/-15%)
    trend_adjustment = np.clip(trend * 0.5, -0.15, 0.15)
    adjusted_loyalty = np.clip(weighted_avg + trend_adjustment, 0, 1)

    blue_wave_pressure = malay_unity * row['malay_pct'] * 2.0

    loyalty_score = np.clip(adjusted_loyalty - blue_wave_pressure * 0.5, 0, 1)
    # blue_wave_pressure weight halved here since weighted_avg
    # already incorporates the most recent (post-Sheraton-Move-era)
    # election result, partially capturing current political climate

    final_score = np.clip(
        blend_weight * johor_model_prob + (1 - blend_weight) * loyalty_score,
        0, 1
    )

    is_ood_proxy = volatility > 0.15  # >15pp swing in vote share = volatile

    components = {
        'weighted_avg_share': round(weighted_avg, 3),
        'trend':              round(trend, 3),
        'trend_adjustment':   round(trend_adjustment, 3),
        'adjusted_loyalty':   round(adjusted_loyalty, 3),
        'volatility':         round(volatility, 3),
        'n_data_points':      n_points,
        'blue_wave_pressure': round(blue_wave_pressure, 3),
        'loyalty_score':      round(loyalty_score, 3),
        'johor_model_prob':   round(johor_model_prob, 3),
        'final_score':        round(float(final_score), 3),
        'is_ood_proxy':       bool(is_ood_proxy),
    }

    return float(final_score), components


def categorise_risk(score: float) -> str:
    if score < 0.20:
        return "🔴 SAFE OPPOSITION"       #
    elif score < 0.35:
        return "🟠 LIKELY OPPOSITION"
    elif score < 0.45:
        return "🟡 LEAN OPPOSITION"
    elif score < 0.55:
        return "⚪ TOSS-UP"
    elif score < 0.65:
        return "🟡 LEAN HARAPAN"
    elif score < 0.80:
        return "🟢 LIKELY HARAPAN"
    else:
        return "🟢 SAFE HARAPAN"

    