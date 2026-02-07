import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder #represent columns in binary values
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import joblib

from keras.models import Sequential
from keras.layers import Dense

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


df_parliment = pd.read_csv("Parliment_GE14.csv")
df_state = pd.read_csv("State_2018.csv")
df_votes = pd.read_csv("TotalVotes_2018.csv")

df_parliment = df_parliment.rename(columns = {"Pekerjaan" : "Jobs"})
df_parliment["Jobs"] = df_parliment["Jobs"].replace(["Tiada", "TIADA", "0", '-'], "Tiada Pekerjaan")
df_parliment["Status"] = df_parliment["Status"].replace({"KLH": "Lost", "MNG": "Win", "HD": "Deposit Loss"})
df_parliment["Jobs"] = df_parliment["Jobs"].replace(["Tiada", "TIADA", "0", '-'], "Tiada Pekerjaan")
df_parliment["Status"] = df_parliment["Status"].replace({"KLH": "Lost", "MNG": "Win", "HD": "Deposit Loss"})
    
#remove any special symbols into just integers in numerical_features inputs (, %)
df_parliment['Total Votes'] = (
    df_parliment['Total Votes']
    .astype(str)
    .str.replace(',', '', regex=False) # Removes the comma
    .astype(float)
)

df_parliment['Turnout Rate'] = (
    df_parliment['Turnout Rate']
    .astype(str)
    .str.replace('%', '', regex=False) # Removes the percentage sign
    .astype(float)
)



#new 'states' column in parliment -> via f string for seat ID
state_seatID = {
    "Perlis": range(1,4),
    "Kedah" : range(4, 18),
    "Kelantan" : range(18, 32),
    "Terengganu" : range(32, 40),
    "Pulau Pinang" : range(40, 53),
    "Perak" : range(53, 77),
    "Pahang" : range(77, 91),
    "Selangor" : range(91, 114),
    "Kuala Lumpur" : range(114, 125),
    "Putrajaya" : range(125, 126),
    "Negeri Sembilan" : range(126, 134),
    "Melaka" : range(134, 140),
    "Johor" : range(140, 166),
    "Labuan" : range(166, 167),
    "Sabah" : range(167, 193),
    "Sarawak" : range(193, 223),
}

region_seatID = {
    "North": set(state_seatID["Perlis"]) | set(state_seatID["Kedah"])|set(state_seatID["Pulau Pinang"]),
    "West Coast": set(state_seatID["Selangor"]) | set(state_seatID["Perak"])|set(state_seatID["Negeri Sembilan"])| set(state_seatID["Melaka"]) | set(state_seatID["Kuala Lumpur"]) | set(state_seatID["Putrajaya"]),
    "East Coast": set(state_seatID["Kelantan"]) | set(state_seatID["Pahang"])|set(state_seatID["Terengganu"]),
    "South": set(state_seatID["Johor"]),
    "East Malaysia (Sabah)": set(state_seatID["Sabah"]),
    "East Malaysia (Sarawak)": set(state_seatID["Sarawak"]),
    "East Malaysia (Labuan)": set(state_seatID["Labuan"]),
}

#func to map seatId into State
def get_state(seat_id):
    seat_num = int(seat_id.split(".")[1])
    # "P.054".split(".") → ["P", "054"] -> 54
    #for key, value in state_seatID.items():items() -> loop within keys & values
    for state, seat_range in state_seatID.items():
        if seat_num in seat_range:
            return state
    return None #no match acquired

def get_region(seat_id):
    seat_num = int(seat_id.split(".")[1])
    for region, set_state in region_seatID.items():
        if seat_num in set_state:
            return region
    return None

def eng_features(df):
    df_parliment["State"] = df_parliment["Seat ID"].map(get_state)
    df_parliment["Region"] = df_parliment["Seat ID"].map(get_region)

    return df

def main():
    # 1. Load the data
    df = pd.read_csv("Parliment_GE14.csv")
    
    # 2. Clean numeric columns (Remove commas and % signs)
    df['Total Votes'] = df['Total Votes'].astype(str).str.replace(',', '').astype(float)
    df['Turnout Rate'] = df['Turnout Rate'].astype(str).str.replace('%', '').astype(float)
    df['Votes for Candidate'] = df['Votes for Candidate'].astype(str).str.replace(',', '').astype(float)

    # 3. Feature Engineering (Margin & Logs)
    opponent_votes = df['Total Votes'] - df['Votes for Candidate']
    df['Relative Vote Margin'] = (df['Votes for Candidate'] - opponent_votes) / df['Total Votes']
    df['Log Total Votes'] = np.log1p(df['Total Votes'])
    
    # Calculate state average for log votes
    state_avg = df.groupby('State')['Log Total Votes'].transform('mean')
    df['Log Votes vs State Avg'] = df['Log Total Votes'] - state_avg

    # 4. Define Target (y) and Features (X)
    # Your CSV 'Result' column is already 0 and 1. Perfect.
    y = df['Result'].astype(int)
    
    # DROP identity columns to keep the model Agnostic (Party Blind)
    columns_to_drop = [
        'Seat ID', 'Seat Name', 'Candidate Name', 'Votes for Candidate',
        'Result', 'Status', 'Coalition', 'Candidate Party', 'Gender'
    ]
    X = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 6. Build the Pipeline
    num_features = ['Total Votes', 'Turnout Rate', 'Relative Vote Margin', 'Log Total Votes', 'Log Votes vs State Avg']
    cat_features = ['State', 'Region', 'Jobs']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

    baseline_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, C=0.1))
    ])

    # 7. Train and Save
    print(f"Training on {len(X_train)} rows...")
    baseline_model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
    print(f"Success! Model Accuracy: {accuracy:.2%}")
    
    joblib.dump(baseline_model, 'stable_election_model.joblib')
    print("Model saved as stable_election_model.joblib")


def feature_analyzer(baseline_model, preprocessor):
    #feature diagnosis to read weights assigned by LR models to each feature
    #to know how influencial a feature is in predicting

    #access classifier aka LR from pipeline
    log_reg = baseline_model.named_steps['classifier']
    coefficients = log_reg.coef_[0] 

    #obtain efature names after preprocessings
    feature_names = preprocessor.get_feature_names_out()

    #create DF to easy viewing & sorting
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    #sort functions, higher absolute coefficient means stronger influence
    feature_importance_df['Absolute_Coefficient'] = np.abs(feature_importance_df['Coefficient'])
    feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

    print("\n--- Top 15 Logistic Regression Feature Importances (Coefficients) ---")
    print(feature_importance_df.head(15).to_markdown(index=False))
    print("-------------------------------------------------------------------\n")

#analyze each coalition's performance in each state
def coalition_analyzer(df, coalition_name):
    coalition_df = df[df['Coalition'] == coalition_name].copy()
    #why double df here??

    if coalition_df.empty:
        return pd.DataFrame()
    
    #aggregate by state
    state_results = (
        coalition_df.groupby('State').agg(Total_Seats_Contested = ('Seat Name', 'nunique'),
                                          Total_Seats_Won = ('Result', 'sum')
                                          )
                                          .reset_index()
    )

    #calculate win percentage
    state_results['Win_Percentage'] = (
        state_results['Total_Seats_Won'] / state_results['Total_Seats_Contested']
    ) *100

    return state_results

def winner_analyzer(df):
    overall_results = (
        df.groupby('Coalition')['Result']
        .sum()
        .reset_index(name = 'Total_Seats_Won')
    )

    winning_coalition_name = overall_results.loc[overall_results['Total_Seats_Won'].idxmax(), 'Coalition']

    print(f"\n--- Dynamic Analysis: Winner Identified as '{winning_coalition_name}' ---")
    winner_report = coalition_analyzer(df, winning_coalition_name)

    if not winner_report.empty:
        winner_report['Win_Percentage'] = winner_report['Win_Percentage'].round(2).astype(str) + '%'
        print(winner_report.to_markdown(index=False))
        return winner_report
    else:
        print(f"Error: Could not generate report for {winning_coalition_name}.")
        return None



if __name__ == "__main__":
    X_train_processed, X_test, y_train, y_test, preprocessor, model, history, baseline_model, df=main()

    feature_analyzer(baseline_model, preprocessor)

    winner_analyzer(df)