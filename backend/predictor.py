import pandas as pd
import numpy as np
import joblib
from gedataset_pipeline import clean_dataset

def prediction_engine(input_data):
    model = joblib.load('stable_election_model.joblib')

    df_input = pd.DataFrame([input_data])
    df_processed = clean_dataset(df_input)
    
    # 2. Fill missing categories (State, Region, Jobs) with 'Other'
    for col in model.feature_names_in_:
        if col not in df_processed.columns:
            df_processed[col] = 0

    final_df = df_processed[model.feature_names_in_]
    prob = model.predict_proba(df_input)[0][1]
    
    return ("WIN" if prob >= 0.5 else "LOSS"), prob

if __name__ == "__main__":
    sample_seat = {
        'State': 'Kelantan',
        'Coalition': 'PAS',
        'Total Votes': 44680,
        'Votes for Candidate': 35000 
    }
    
    res, p = prediction_engine(sample_seat)
    print(f"Result: {res} ({p:.2%})")