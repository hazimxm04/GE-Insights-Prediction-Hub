from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

#the trained model
model = joblib.load("stable_election_model.joblib")

app = FastAPI()

class ElectionInput(BaseModel):
    state: str
    coalition: str
    total_registered: int
    turnout_rate: float
    base_support: float
    swing: float

#wtf is this
@app.post("/predict")

def get_prediction(data: ElectionInput):
    # --- THIS IS THE BACK-END MATH YOU ALREADY KNOW ---
    total_votes_cast = data.total_registered * (data.turnout_rate / 100)
    final_vote_share = (data.base_support + data.swing) / 100
    
    candidate_votes = total_votes_cast * final_vote_share
    opponent_votes = total_votes_cast - candidate_votes
    relative_margin = (candidate_votes - opponent_votes) / total_votes_cast

    if total_votes_cast > 0:
        relative_margin = (candidate_votes - opponent_votes) / total_votes_cast
    else:
        relative_margin = 0  # This prevents the 500 Internal Server Error
    
    # Create the DataFrame just like your model expects
    input_df = pd.DataFrame([{
        'State': data.state,
        'Coalition': data.coalition,
        'Candidate Vote Share': float(final_vote_share * 100),
        'Total Votes': total_votes_cast,
        'Relative Vote Margin': relative_margin,
        'Log Total Votes': np.log1p(total_votes_cast),
        'Region': 'West Coast',       # Placeholder logic
        'Log Votes vs State Avg': 0.0 # Placeholder logic
    }])
    
    # 3. Ask the AI for the answer
    prob = model.predict_proba(input_df)[0][1]
    
    # 4. Send back a JSON "Message"
    return {
        "probability": round(float(prob), 4),
        "verdict": "WIN" if prob > 0.5 else "LOSS",
        "margin_calculated": round(relative_margin, 4)
    }