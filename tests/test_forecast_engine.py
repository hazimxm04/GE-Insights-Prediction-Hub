"""
tests/test_forecast_engine.py
===============================
Unit tests for forecast_engine.py's pure, testable functions.
No network/API dependencies — these test core business logic only.
"""

import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forecast_engine import (
    extract_seat_number,
    compute_ph_loyalty_weighted,
    compute_forecast_score,
    categorise_risk,
)


# ── extract_seat_number ─────────────────────────────────────────

def test_extract_seat_number_standard():
    assert extract_seat_number("N.21 Pandan Indah") == "N.21"

def test_extract_seat_number_single_digit():
    assert extract_seat_number("N.1 Buloh Kasap") == "N.1"

def test_extract_seat_number_double_digit():
    assert extract_seat_number("N.56 Sungai Pelek") == "N.56"

def test_extract_seat_number_no_match():
    assert extract_seat_number("Invalid Seat Name") is None

def test_extract_seat_number_empty_string():
    assert extract_seat_number("") is None

def test_extract_seat_number_none_input():
    assert extract_seat_number(None) is None


# ── categorise_risk ──────────────────────────────────────────────

def test_categorise_risk_safe_govt():
    assert categorise_risk(0.10) == "🔴 SAFE OPPOSITION"


def test_categorise_risk_likely_govt():
    assert categorise_risk(0.25) == "🟠 LIKELY OPPOSITION"

def test_categorise_risk_lean_govt():
    assert categorise_risk(0.40) == "🟡 LEAN OPPOSITION"

def test_categorise_risk_tossup():
    assert categorise_risk(0.50) == "⚪ TOSS-UP"

def test_categorise_risk_lean_harapan():
    assert categorise_risk(0.60) == "🟡 LEAN HARAPAN"

def test_categorise_risk_likely_harapan():
    assert categorise_risk(0.70) == "🟢 LIKELY HARAPAN"

def test_categorise_risk_safe_harapan():
    assert categorise_risk(0.90) == "🟢 SAFE HARAPAN"

def test_categorise_risk_boundary_exact_020():
    # Boundary case: exactly 0.20 should fall into LIKELY OPPOSITION (>=0.20)
    assert categorise_risk(0.20) == "🟠 LIKELY OPPOSITION"

def test_categorise_risk_boundary_exact_055():
    assert categorise_risk(0.55) == "🟡 LEAN HARAPAN"


# ── compute_ph_loyalty_weighted ───────────────────────────────────

def test_loyalty_weighted_no_data_returns_neutral():
    row = {
        'ph_share_2008': None, 'ph_share_2013': None,
        'ph_share_2018': None, 'ph_share_recent': None,
    }
    avg, trend, vol, n = compute_ph_loyalty_weighted(row)
    assert n == 0
    assert avg == 0.5
    assert trend == 0.0
    assert vol == 0.0

def test_loyalty_weighted_single_data_point():
    row = {
        'ph_share_2008': None, 'ph_share_2013': None,
        'ph_share_2018': None, 'ph_share_recent': 60.0,
    }
    avg, trend, vol, n = compute_ph_loyalty_weighted(row)
    assert n == 1
    assert math.isclose(avg, 0.60, rel_tol=1e-6)
    assert trend == 0.0  # can't compute trend with only 1 point

def test_loyalty_weighted_consistent_history():
    # Seat that won 60% in every recorded election - stable, no trend
    row = {
        'ph_share_2008': 60.0, 'ph_share_2013': 60.0,
        'ph_share_2018': 60.0, 'ph_share_recent': 60.0,
    }
    avg, trend, vol, n = compute_ph_loyalty_weighted(row)
    assert n == 4
    assert math.isclose(avg, 0.60, rel_tol=1e-6)
    assert math.isclose(trend, 0.0, abs_tol=1e-6)
    assert math.isclose(vol, 0.0, abs_tol=1e-6)

def test_loyalty_weighted_declining_trend():
    # Seat trending downward over time - trend should be negative
    row = {
        'ph_share_2008': 70.0, 'ph_share_2013': 60.0,
        'ph_share_2018': 55.0, 'ph_share_recent': 50.0,
    }
    avg, trend, vol, n = compute_ph_loyalty_weighted(row)
    assert n == 4
    assert trend < 0  # declining
    assert math.isclose(trend, -0.20, rel_tol=1e-6)  # 50-70 = -20pp

def test_loyalty_weighted_recency_weighting():
    # Most recent share should dominate the weighted average
    # since it has the highest weight (40%)
    row_high_recent = {
        'ph_share_2008': 20.0, 'ph_share_2013': 20.0,
        'ph_share_2018': 20.0, 'ph_share_recent': 80.0,
    }
    avg_high, _, _, _ = compute_ph_loyalty_weighted(row_high_recent)

    row_low_recent = {
        'ph_share_2008': 80.0, 'ph_share_2013': 80.0,
        'ph_share_2018': 80.0, 'ph_share_recent': 20.0,
    }
    avg_low, _, _, _ = compute_ph_loyalty_weighted(row_low_recent)

    # Same set of numbers {20,20,20,80} vs {80,80,80,20}, but the
    # recency-weighted average should differ based on which value
    # sits in the 'recent' (40% weight) slot
    assert avg_high != avg_low
    assert math.isclose(avg_high + avg_low, 1.0, rel_tol=1e-6)


# ── compute_forecast_score ─────────────────────────────────────────

def test_forecast_score_returns_valid_probability():
    row = {
        'ph_share_2008': 55.0, 'ph_share_2013': 55.0,
        'ph_share_2018': 55.0, 'ph_share_recent': 55.0,
        'malay_pct': 0.5,
    }
    score, components = compute_forecast_score(row, johor_model_prob=0.5)
    assert 0.0 <= score <= 1.0

def test_forecast_score_extreme_johor_prob_still_bounded():
    row = {
        'ph_share_2008': 90.0, 'ph_share_2013': 90.0,
        'ph_share_2018': 90.0, 'ph_share_recent': 90.0,
        'malay_pct': 0.1,
    }
    score, components = compute_forecast_score(row, johor_model_prob=1.0)
    assert score <= 1.0  # must be clipped, never exceed 1.0

def test_forecast_score_zero_inputs_bounded():
    row = {
        'ph_share_2008': 0.0, 'ph_share_2013': 0.0,
        'ph_share_2018': 0.0, 'ph_share_recent': 0.0,
        'malay_pct': 1.0,
    }
    score, components = compute_forecast_score(row, johor_model_prob=0.0)
    assert score >= 0.0  # must be clipped, never go below 0.0

def test_forecast_score_high_volatility_flags_ood():
    # Seat with wildly swinging vote share should be flagged
    # as a simplified OOD proxy
    row = {
        'ph_share_2008': 10.0, 'ph_share_2013': 90.0,
        'ph_share_2018': 10.0, 'ph_share_recent': 90.0,
        'malay_pct': 0.5,
    }
    score, components = compute_forecast_score(row, johor_model_prob=0.5)
    assert components['is_ood_proxy'] is True