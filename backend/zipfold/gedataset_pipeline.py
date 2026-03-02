import pandas as pd
import numpy as np

#maps for cleaning column names across different GE datasets
#aka map for each dataset
COLUMN_MAPS = {
    "GE12": {"parlimen": "parlimen_raw", "state": "State", "party": "party", "votes": "votes_for_candidate", "name": "candidate", "ethnicity": "ethnicity"},
    "GE13": {"parlimen": "parlimen_raw", "state": "State", "party": "party", "votes": "votes_for_candidate", "name": "candidate", "ethnicity": "ethnicity"},
    "GE14": {"parlimen": "parlimen_raw", "state": "State", "party": "party", "votes": "votes_for_candidate", "name": "candidate", "ethnicity": "ethnicity"}
}

# This map is for GE12_13_14_clean.csv
TURNOUT_MAPS = {
    "GE12": {
        "PARLIAMENT CODE": "seat_id",
        "GE12 VOTERS": "total_votes",
        "GE12 VOTER TURNOUT %": "turnout_rate",
    },
    "GE13": {
        "PARLIAMENT CODE": "seat_id",
        "GE13 VOTERS": "total_votes",
        "GE13 VOTER TURNOUT %": "turnout_rate"
    },
    "GE14": {
        "PARLIAMENT CODE.1": "seat_id",  # <--- Note the .1 here!
        "GE14 VOTERS": "total_votes",
        "GE14 VOTER TURNOUT %": "turnout_rate"
    }
}

COALITION_LIST = {
    "GE12": {
        "BARISAN NASIONAL (BN)": "Barisan",
        "PARTI ISLAM SE MALAYSIA (PAS)": "Pakatan Rakyat",
        "PARTI KEADILAN RAKYAT (PKR)": "Pakatan Rakyat",
        "PARTI TINDAKAN DEMOKRATIK (DAP)": "Pakatan Rakyat",
    },
    "GE13": {
        "BARISAN NASIONAL (BN)": "Barisan",
        "PARTI ISLAM SE MALAYSIA (PAS)": "Pakatan Rakyat",
        "PARTI KEADILAN RAKYAT (PKR)": "Pakatan Rakyat",
        "PARTI TINDAKAN DEMOKRATIK (DAP)": "Pakatan Rakyat",
        "BARISAN JEMAAH ISLAMIAH SE MALAYSIA (BERJASA)": "Others",
    },
    "GE14": {
        "BARISAN NASIONAL (BN)": "Barisan",
        "PARTI ISLAM SE MALAYSIA (PAS)": "PAS",        # Standalone for GE14
        "BARISAN JEMAAH ISLAMIAH SE MALAYSIA (BERJASA)": "BERJASA", # Standalone for GE14
        "PARTI KEADILAN RAKYAT (PKR)": "Harapan",
        "PARTI TINDAKAN DEMOKRATIK (DAP)": "Harapan",
        "PARTI AMANAH NEGARA (AMANAH)": "Harapan",
        "PARTI WARISAN SABAH (WARISAN)": "Warisan",
    }
}

num_cols = ['Total Votes', 'Candidate Vote Share', 'Relative Vote Margin', 'Log Total Votes', 'Log Votes vs State Avg']
cat_cols = ['State', 'Region', 'Coalition']

state_seatID = {
        "Perlis": range(1,4), "Kedah": range(4, 18), "Kelantan": range(18, 32),
        "Terengganu": range(32, 40), "Pulau Pinang": range(40, 53), "Perak": range(53, 77),
        "Pahang": range(77, 91), "Selangor": range(91, 114), "Kuala Lumpur": range(114, 125),
        "Putrajaya": range(125, 126), "Negeri Sembilan": range(126, 134), "Melaka": range(134, 140),
        "Johor": range(140, 166), "Labuan": range(166, 167), "Sabah": range(167, 193), "Sarawak": range(193, 223),
    }

region_seatID = {
        "North": set(state_seatID["Perlis"]) | set(state_seatID["Kedah"]) | set(state_seatID["Pulau Pinang"]),
        "West Coast": set(state_seatID["Selangor"]) | set(state_seatID["Perak"]) | set(state_seatID["Negeri Sembilan"]) | set(state_seatID["Melaka"]) | set(state_seatID["Kuala Lumpur"]) | set(state_seatID["Putrajaya"]),
        "East Coast": set(state_seatID["Kelantan"]) | set(state_seatID["Pahang"]) | set(state_seatID["Terengganu"]),
        "South": set(state_seatID["Johor"]),
        "East Malaysia (Sabah)": set(state_seatID["Sabah"]),
        "East Malaysia (Sarawak)": set(state_seatID["Sarawak"]),
        "East Malaysia (Labuan)": set(state_seatID["Labuan"]),
}

# Add this mapping to gedataset_pipeline.py
STRICT_COALITION_MAP = {
    # GE14 Names
    "BARISAN NASIONAL (BN)": "BN",
    "PAKATAN HARAPAN (PH)": "PH",
    "PARTI ISLAM SE-MALAYSIA (PAS)": "PAS",
    # GE13/12 Names
    "BARISAN NASIONAL": "BN",
    "PAKATAN RAKYAT": "PH", # Logic: PR is the predecessor to PH
    "PARTI TINDAKAN DEMOKRATIK (DAP)": "PH",
    "PARTI KEADILAN RAKYAT (PKR)": "PH"
}

def standardize_coalition(party_name, year):
    # Try to find a direct map, otherwise return "Others"
    name = str(party_name).strip().upper()
    return STRICT_COALITION_MAP.get(name, "Others")

def get_seat_num(seat_id):
    """Extracts the integer number from SeatID formats like 'P.001' or '001'."""
    try:
        return int(str(seat_id).split(".")[1]) if "." in str(seat_id) else int(seat_id)
    except (IndexError, ValueError):
        return 0
    
def prepare_and_clean(results_path, turnout_path, year):
    df_results = pd.read_csv(results_path)
    df_turnout_master = pd.read_csv(turnout_path)
    
    # 1. Clean headers
    df_results.columns = df_results.columns.str.strip().str.replace(u'\ufeff', '')
    df_turnout_master.columns = df_turnout_master.columns.str.strip().str.replace(u'\ufeff', '')

    # 2. Rename candidate columns
    df_results = df_results.rename(columns=COLUMN_MAPS[year])

    if 'candidate' not in df_results.columns:
    # Look for any column that contains 'name' or 'nama' (case insensitive)
        potential_name_cols = [c for c in df_results.columns if 'name' in c.lower() or 'nama' in c.lower()]
        if potential_name_cols:
            df_results = df_results.rename(columns={potential_name_cols[0]: 'candidate'})

    # 3. Extract seat_id from candidate file
    if 'parlimen_raw' in df_results.columns:
        split_data = df_results['parlimen_raw'].str.split(' ', n=1, expand=True)
        df_results['seat_id'] = split_data[0].str.strip()
        df_results['seat_name'] = split_data[1].str.strip()
    
    # 4. Rename turnout columns using your TURNOUT_MAPS
    year_map = TURNOUT_MAPS[year]
    df_turnout_slice = df_turnout_master.rename(columns=year_map)

    # 5. Safety check (using lowercase 'seat_id' as defined in your map)
    if 'seat_id' not in df_turnout_slice.columns:
        available = df_turnout_master.columns.tolist()
        raise ValueError(f"Still can't find 'seat_id'. Available: {available}")

    # 6. Merge (Both now have 'seat_id' lowercase)
    final_df = pd.merge(df_results, df_turnout_slice, on="seat_id", how="left")
    
    return clean_dataset(final_df)

def clean_dataset(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    df.columns = [c.lower() for c in df.columns]
    
    def assign_coalition(row):
        year = str(row.get('election_year', 'GE14'))
        party = str(row.get('party', '')).strip()
        return COALITION_LIST.get(year, {}).get(party, "Others")

    df['standardized_coalition'] = df.apply(assign_coalition, axis=1)
    #df['coalition'] = df.apply(assign_coalition, axis=1)

    # Clean numeric columns
    # We treat 'turnout_rate' as float, but votes as int
    cols_to_fix = ["votes_for_candidate", "total_votes", "turnout_rate"]
    for col in cols_to_fix:
        if col in df.columns:
            # If col is a DataFrame (due to duplicates), take the first one
            series_data = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
            
            s = series_data.astype(str).replace(r'[%, ]', '', regex=True)
            s = s.replace(['Uncontested', '#VALUE!', 'nan', 'Tiada', 'None'], '0')
            
            df[col] = pd.to_numeric(s, errors='coerce').fillna(0)
            
            if col in ["votes_for_candidate", "total_votes"]:
                df[col] = df[col].astype(int)
    
    if "seat_id" in df.columns:
        seat_nums = df["seat_id"].apply(get_seat_num)

        df["state"] = "Unknown"
        for state, s_range in state_seatID.items():
            df.loc[seat_nums.isin(s_range), "state"] = state
            
        df["region"] = "Unknown"
        for region, s_set in region_seatID.items():
            df.loc[seat_nums.isin(s_set), "region"] = region

        # Math Features
        if 'total_votes' in df.columns and 'votes_for_candidate' in df.columns:
            opponent_votes = df['total_votes'] - df['votes_for_candidate']
            df['relative_vote_margin'] = np.where(df['total_votes'] > 0,
                                                 (df['votes_for_candidate'] - opponent_votes) / df['total_votes'],
                                                 0)
            df['log_total_voters'] = np.log1p(df['total_votes'].astype(float))

    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df