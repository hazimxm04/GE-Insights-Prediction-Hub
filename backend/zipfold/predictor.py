import os
import pandas as pd
import numpy as np
import joblib
from gedataset_pipeline import clean_dataset

BASEDIR = os.path.dirname(os.path.abspath(__file__))
_model_cache = {}

def get_model(year: str):
    year = year.strip().upper()
    if year in _model_cache:
        return _model_cache[year]

    model_path = os.path.join(BASEDIR, f"model_{year}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    _model_cache[year] = model
    return model

def prediction_engine(input_data: dict, electionyear: str = "GE14"):
    model = get_model(electionyear)
    df_input = pd.DataFrame([input_data])
    
    # --- CRITICAL FIX START ---
    # Your analysis.py uses 'standardized_coalition' in the ColumnTransformer.
    # We must ensure the input dict has this key before it hits the model.
    if "standardized_coalition" not in df_input.columns:
        df_input["standardized_coalition"] = df_input.get("coalition", "Others")
    # --- CRITICAL FIX END ---

    if "election_year" not in df_input.columns:
        df_input["election_year"] = electionyear

    # Clean the dataset (calculates relative_vote_margin, etc.)
    df_processed = clean_dataset(df_input)
    
    try:
        # The model is a Pipeline; it will handle OneHotEncoding for
        # 'state', 'region', and 'standardized_coalition' automatically.
        prob = model.predict_proba(df_processed)[0, 1]
        verdict = "WIN" if prob >= 0.5 else "LOSS"
        return verdict, float(prob)
    except Exception as e:
        # This will now tell you exactly which feature is missing if it fails
        print(f"Prediction Error for {electionyear}: {e}")
        return "ERROR", 0.0

if __name__ == "__main__":
    # Test with the new mapping
    sample_seat = {
        "seat_id": "P.001",
        "state": "Perlis",
        "party": "UMNO",
        "coalition": "BN",
        "total_votes": 40000,
        "votes_for_candidate": 25000,
        "turnout_rate": 80.0,
    }

    res, p = prediction_engine(sample_seat, electionyear="GE14")
    print(f"Result: {res} ({p:.2%})")