import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

# ── Health ────────────────────────────────────────────────────────

def test_health_check():
    """API is alive"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_returns_endpoints():
    """Root endpoint lists all routes"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "states" in data

# ── Predict All ───────────────────────────────────────────────────

def test_predict_all_johor():
    """Predict all Johor seats"""
    response = client.get("/predict/all/johor")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "johor"
    assert data["total_seats"] == 56
    assert data["predicted_bn"] + data["predicted_non_bn"] == 56
    assert len(data["predictions"]) == 56

def test_predict_all_neg_sembilan():
    """Predict all NS seats"""
    response = client.get("/predict/all/neg_sembilan")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "neg_sembilan"
    assert data["total_seats"] == 36

def test_predict_all_invalid_state():
    """Invalid state returns 400"""
    response = client.get("/predict/all/selangor")
    assert response.status_code == 400

# ── Predict Single Seat ───────────────────────────────────────────

def test_predict_single_seat():
    """Predict single seat returns valid structure"""
    response = client.post("/predict/seat/johor", json={
        "seat_name":            "N.10 Tangkak",
        "majority_change":      -2000,
        "turnout_change":       -3.5,
        "incumbent_held":       1,
        "log_voters":           10.67,
        "majority_perc_change": -0.05,
        "n_candidates_b":       4
    })
    assert response.status_code == 200
    data = response.json()

    # Check all required fields
    assert "prediction"    in data
    assert "probability"   in data
    assert "confidence"    in data
    assert "is_ood"        in data
    assert "fallback_used" in data

    # Check valid values
    assert data["prediction"] in ["BN", "non-BN"]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["confidence"] in ["HIGH", "MEDIUM", "LOW"]
    assert isinstance(data["is_ood"], bool)

def test_predict_probability_range():
    """Probability always between 0 and 1"""
    response = client.post("/predict/seat/johor", json={
        "seat_name":   "N.01 Buloh Kasap",
        "log_voters":  10.5,
    })
    assert response.status_code == 200
    prob = response.json()["probability"]
    assert 0.0 <= prob <= 1.0

def test_predict_seat_invalid_state():
    """Invalid state returns 400"""
    response = client.post("/predict/seat/sabah", json={
        "seat_name": "N.01 Somewhere"
    })
    assert response.status_code == 400

# ── Analysis ──────────────────────────────────────────────────────

def test_validation_summary():
    """Validation summary returns all states"""
    response = client.get("/analysis/validation-summary")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "johor" in data["results"]
    assert "neg_sembilan" in data["results"]

def test_validation_accuracy_range():
    """Validation accuracy is between 0 and 1"""
    response = client.get("/analysis/validation-summary")
    data = response.json()
    johor_acc = data["results"]["johor"]["accuracy"]
    assert 0.0 <= johor_acc <= 1.0

def test_feature_importance_johor():
    """Feature importance returns ranked list"""
    response = client.get("/analysis/feature-importance/johor")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importance" in data
    features = data["feature_importance"]
    assert len(features) > 0
    # First feature should be most important (sorted desc)
    assert features[0]["importance"] >= features[-1]["importance"]
    # majority_change should be top feature
    assert features[0]["feature"] == "majority_change"

def test_ood_analysis_neg_sembilan():
    """NS 2026 is 100% OOD (regime shift)"""
    response = client.get("/analysis/ood/neg_sembilan")
    assert response.status_code == 200
    data = response.json()
    assert data["ood_seats"]["percentage"] == 100.0
    assert data["in_distribution_seats"]["count"] == 0

def test_ood_analysis_johor():
    """Johor has partial OOD (not full regime shift)"""
    response = client.get("/analysis/ood/johor")
    assert response.status_code == 200
    data = response.json()
    # Johor should have some in-distribution seats
    assert data["in_distribution_seats"]["count"] > 0
    assert data["ood_seats"]["count"] > 0

def test_metadata_returns_model_info():
    """Metadata returns model accuracy info"""
    response = client.get("/analysis/metadata/johor")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "features" in data

def test_metadata_invalid_state():
    """Invalid state returns 400"""
    response = client.get("/analysis/metadata/penang")
    assert response.status_code == 400