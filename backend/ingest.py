import os
import pandas as pd
import numpy as np
from supabase import create_client
# Import only the cleaning function from your pipeline
from gedataset_pipeline import prepare_and_clean

# 1. Connection settings (No more importing from database.py!)
URL = "https://csezluavrtajzkggpzvd.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNzZXpsdWF2cnRhanprZ2dwenZkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA4NzkxODksImV4cCI6MjA4NjQ1NTE4OX0.YAclimmL-K3sV-4FCUJj6Chd8tuEC3760eikaSMiByQ"

# Initialize the client directly
supabase = create_client(URL, KEY)


def run_ingestion():
    years = ["GE12", "GE13", "GE14"]
    turnout_master = "GE12_13_14_clean.csv"
    
    # Define the EXACT columns that exist in your Supabase table
    SQL_COLUMNS = [
        'election_year', 'state', 'seat_id', 'seat_name', 'candidate', 
        'party', 'coalition', 'votes_for_candidate', 'total_votes', 
        'turnout_rate', 'relative_vote_margin', 'log_total_voters', 'region', 'ethnicity'
    ]
    
    for year in years:
        results_file = f"candidates_{year.lower()}.csv"
        
        if os.path.exists(results_file):
            print(f"🚀 Processing and Uploading {year}...")
            
            try:
                # 1. Clean data
                df_clean = prepare_and_clean(results_file, turnout_master, year)
                df_clean['election_year'] = year

                from gedataset_pipeline import COALITION_LIST
                df_clean['coalition'] = df_clean.apply(
                    lambda row: COALITION_LIST.get(year, {}).get(str(row['party']).strip(), "Others"), 
                    axis=1
                )

                # 2. Schema Filter: Only keep columns that match Supabase
                # This automatically ignores 'age', 'parlimen_raw', etc.
                existing_cols = [c for c in df_clean.columns if c in SQL_COLUMNS]
                df_final = df_clean[existing_cols].copy()

                df_final = df_final.loc[:, ~df_final.columns.duplicated()]

                # 3. Handle Math Errors
                df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                df_final['total_votes'] = df_final['total_votes'].astype(int)
                df_final['votes_for_candidate'] = df_final['votes_for_candidate'].astype(int)

                # 4. Upload to Supabase
                records = df_final.to_dict(orient='records')
                
                if records:
                    supabase.table("historical_results").insert(records).execute()
                    print(f"✅ {year} is live in Supabase!")
            except Exception as e:
                print(f"❌ Error during {year}: {e}")
        else:
            print(f"⚠️ Missing: {results_file}")

if __name__ == "__main__":
    run_ingestion()