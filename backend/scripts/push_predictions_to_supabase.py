# backend/scripts/push_predictions_to_supabase.py
# Run AFTER training, push predictions only

from database import supabase
import pandas as pd
import json

def push_predictions(state: str, predictions: list):
    """Push model predictions to Supabase for frontend"""
    
    records = [
        {
            'state': state,
            'seat_name': p['seat_name'],
            'prediction': p['prediction'],
            'probability': p['probability'],
            'confidence': p['final_confidence'],
            'is_ood': p['is_ood'],
            'model_version': 'v1.0',
            'updated_at': 'now()'
        }
        for p in predictions
    ]
    
    # Upsert (update if exists, insert if not)
    supabase.table("state_predictions").upsert(records).execute()
    print(f"✅ Pushed {len(records)} predictions for {state}")

# Supabase table schema:
"""
CREATE TABLE state_predictions (
    id SERIAL PRIMARY KEY,
    state TEXT,
    seat_name TEXT,
    prediction TEXT,
    probability FLOAT,
    confidence TEXT,
    is_ood BOOLEAN,
    model_version TEXT,
    updated_at TIMESTAMP
);
"""