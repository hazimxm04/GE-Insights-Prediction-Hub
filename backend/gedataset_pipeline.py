import pandas as pd
import numpy as np


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


def get_seat_num(seat_id):
    """Extracts the integer number from SeatID formats like 'P.001' or '001'."""
    try:
        return int(str(seat_id).split(".")[1]) if "." in str(seat_id) else int(seat_id)
    except (IndexError, ValueError):
        return 0
    
def clean_dataset(df):
    # 1. Create a fresh copy to avoid 'SettingWithCopy' or Slice/View errors
    df = df.copy()
    
    # 2. Clean symbols and convert to numeric safely
    cols_to_fix = ["Votes for Candidate", "Total Votes", "Candidate Vote Share"]
    for col in cols_to_fix:
        if col in df.columns:
            # If there are duplicate columns, df[col] is a DataFrame. 
            # We pick only the first one to ensure it's a Series.
            target_col = df[col].iloc[:, 0] if isinstance(df[col], pd.DataFrame) else df[col]
            
            # Now we clean and convert the single Series
            cleaned_values = target_col.astype(str).replace(r'[%,]', '', regex=True)
            df[col] = pd.to_numeric(cleaned_values, errors='coerce').fillna(0)
    
    if "Jobs" in df.columns:
        df["Jobs"] = df["Jobs"].replace(["Tiada", "TIADA", "0", "-", np.nan], "Tiada Pekerjaan")

    # 3. Feature Engineering
    if "Seat ID" in df.columns:
        seat_nums = df["Seat ID"].apply(get_seat_num)

        # Assign State
        df["State"] = "Unknown"
        for state, s_range in state_seatID.items():
            df.loc[seat_nums.isin(s_range), "State"] = state
            
        # Assign Region
        df["Region"] = "Unknown"
        for region, s_set in region_seatID.items():
            df.loc[seat_nums.isin(s_set), "Region"] = region

        # 4. Math Features (Required for model.feature_names_in_)
        opponent_votes = df['Total Votes'] - df['Votes for Candidate']
        df['Relative Vote Margin'] = np.where(df['Total Votes'] > 0,
                                             (df['Votes for Candidate'] - opponent_votes) / df['Total Votes'],
                                             0)
        
        df['Log Total Votes'] = np.log1p(df['Total Votes'])
        
        # Safe groupby transform to avoid errors on single-row inputs
        state_means = df.groupby('State')['Log Total Votes'].transform('mean')
        df['Log Votes vs State Avg'] = df['Log Total Votes'] - state_means

    return df


