from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
import gedataset_pipeline as gep


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "stable_election_model.joblib")
model = joblib.load(model_path)

#use class when request input from user (Request Body)
class SeatInput(BaseModel):
    seat_id: str
    state: str
    coalition: str
    total_registered: int
    turnout_rate: float
    base_support: float
    swing: float

class StateInput(BaseModel):
    state_filter: str = "Nationwide" # Defaults to Nationwide if you leave it blank
    turnout_rate: float = 80.0       # Defaults to 80%
    swing: float = 0.0
    coalition: str = "All"

@app.get("/seats")
def get_seat_state_list():
    df = pd.read_csv("Parliment_GE14.csv")

    menu_data = df[['Seat ID', 'Seat Name', 'State' ]].to_dict(orient="records")
    return menu_data


@app.post("/predict")

def seat_prediction(data: SeatInput):

    total_votes_cast = data.total_registered * (data.turnout_rate / 100)
    final_vote_share = (data.base_support + data.swing) / 100
    candidate_votes = total_votes_cast * final_vote_share
    
    #remove as nmoved to gedataset_pipeline because??
    #opponent_votes = total_votes_cast - candidate_votes
    #relative_margin = (candidate_votes - opponent_votes) / total_votes_cast


    raw_data = {
        'Seat ID': data.seat_id,
        'Coalition': data.coalition,
        'Total Votes': total_votes_cast,
        'Votes for Candidate': candidate_votes,
        'Candidate Vote Share': final_vote_share * 100
    }
    

    # Create the DataFrame just like your model expects
    input_df = pd.DataFrame([raw_data])
    
    #the list [] inside the () because it's a single row).
    processed_df = gep.clean_dataset(input_df)
    features = processed_df[model.feature_names_in_]
    # 3. Ask the AI for the answer
    prob = model.predict_proba(features)[0][1]
    #features contains the filtered rows

    
    # 4. Send back a JSON "Message"
    return {
        "probability": round(float(prob), 4),
        "verdict": "WIN" if prob > 0.5 else "LOSS",
        "state": processed_df['State'].iloc[0],
        "region": processed_df['Region'].iloc[0]
    }

@app.post("/predict-summary")
def state_prediction(data: StateInput, coalition: str = "All"):
    df = pd.read_csv("Parliment_GE14.csv")
    df = gep.clean_dataset(df)
    #processed_df = gep.clean_dataset(df_pipe)
    #must follow order of 

    if data.state_filter != "Nationwide":
        # This checks if the filter matches a State OR a Region
        df = df[(df['State'] == data.state_filter) | (df['Region'] == data.state_filter)].copy()

    if data.coalition != "All":
        df = df[df["Coalition"] == data.coalition].copy()
    
    if df.empty:
        return {"error": "No matching data found."}

    #new logic
    df['Hist_Share'] = (df['Votes for Candidate'] / df['Total Votes']) * 100
    df['Sim_Turnout'] = df['Total Votes'] * (data.turnout_rate / 100)
    df['Sim_Votes'] = df['Sim_Turnout'] * ((df['Hist_Share'] + data.swing) / 100)

    #df_pipe = df.drop(columns=['Votes for Candidate', 'Total Votes'], errors='ignore')
    df_pipe = df.rename(columns={'Sim_Votes': 'Votes for Candidate', 'Sim_Turnout': 'Total Votes'}).copy()

    # df_pipe = df.rename(columns={'Sim_Votes': 'Votes for Candidate', 'Sim_Turnout': 'Total Votes'}).copy()
    processed_df = gep.clean_dataset(df_pipe)

    features = processed_df[model.feature_names_in_]
    probabilities = model.predict_proba(features)[:, 1] 
    df['Win_Probability'] = probabilities.round(4)

    # Convert "WIN"/"LOSS" strings to 1/0 for math
    df['Verdict_Num'] = (df['Win_Probability'] > 0.5).astype(int)

    #region win summary
    #region_perf = processed_df[processed_df['Verdict'] == 'WIN'].groupby('Region').size().to_dict()

    total_wins = int(df['Verdict_Num'].sum())
    total_losses = len(df) - total_wins

    region_perf = df[df['Verdict_Num'] == 1].groupby('Region').size().to_dict()

    return {
        "summary": {
            "view": data.state_filter,
            "coalition": data.coalition,
            "total_seats": len(df),
            "wins": total_wins,
            "losses": len(df) - total_wins
        },
        "by_region": region_perf,
        "seats": df[['Seat ID', 'Seat Name', 'State', 'Verdict_Num', 'Win_Probability']].to_dict(orient="records")
    }

@app.get("/api/options")
def get_filter_options():
    df = pd.read_csv("Parliment_GE14.csv")
    df_full = gep.clean_dataset(df)
    
    states = sorted(df_full['State'].unique().tolist())
    regions = sorted(df_full['Region'].unique().tolist())
    coalitions = sorted(df_full['Coalition'].unique().tolist())
    seats = sorted((df_full['Seat ID'] + " " + df_full['Seat Name']).unique().tolist())

    

    return {
        "states": states,
        "regions": regions,
        "coalitions": coalitions,
        "seats": seats
    }