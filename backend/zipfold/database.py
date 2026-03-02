import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file automatically

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_KEY. "
        "Create a .env file in the backend folder with these values."
    )

def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_supabase_client()

# In database.py

# database.py
def get_database(year=None):
    query = supabase.table('historical_results').select('*')
    
    if year:
        # Filtering by year bypasses the 1,000 row limit 
        # because each GE year has only ~222 seats.
        response = query.eq('election_year', str(year).strip().upper()).execute()
    else:
        # Without a year, this still hits the 'invisible wall' at 1,000 rows.
        response = query.limit(5000).execute()
    
    data = response.data
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)

    # Align with main.py by making sure winners are only calculated if votes are present
    if 'votes_for_candidate' in df.columns:
        df['votes_for_candidate'] = pd.to_numeric(df['votes_for_candidate'], errors='coerce').fillna(0)
        max_v = df.groupby(['election_year', 'seat_id'])['votes_for_candidate'].transform('max')
        df['is_winner'] = ((df['votes_for_candidate'] == max_v) & (max_v > 0)).astype(int)
    
    return df