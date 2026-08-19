"""
api.py
======
Connects Streamlit frontend to GE-Insights Railway API.
All API calls go through here.
"""

import requests

BASE_URL = "http://localhost:8000"

def get_all_predictions(state: str) -> dict:
    """Get predictions for all seats in a state."""
    try:
        r = requests.get(f"{BASE_URL}/predict/all/{state}", timeout=30)
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def get_seat_prediction(state: str, seat_name: str,
                         majority_change: float = 0.0,
                         turnout_change: float = 0.0,
                         incumbent_held: int = 1,
                         log_voters: float = 10.5,
                         majority_perc_change: float = 0.0,
                         n_candidates_b: int = 3) -> dict:
    """Predict a single seat with custom features."""
    try:
        r = requests.post(
            f"{BASE_URL}/predict/seat/{state}",
            json={
                "seat_name":            seat_name,
                "majority_change":      majority_change,
                "turnout_change":       turnout_change,
                "incumbent_held":       incumbent_held,
                "log_voters":           log_voters,
                "majority_perc_change": majority_perc_change,
                "n_candidates_b":       n_candidates_b,
            },
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def ask_chatbot(question: str) -> dict:
    """Ask the RAG chatbot a question."""
    try:
        r = requests.post(
            f"{BASE_URL}/chatbot/ask",
            json={"question": question},
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def get_metadata(state: str) -> dict:
    """Get model metadata for a state."""
    try:
        r = requests.get(
            f"{BASE_URL}/analysis/metadata/{state}",
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def health_check() -> bool:
    """Check if API is running."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        return r.status_code == 200
    except:
        return False