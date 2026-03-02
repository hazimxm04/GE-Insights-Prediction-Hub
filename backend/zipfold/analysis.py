import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from database import supabase

# 1. GLOBAL CONFIG: Change once to update everywhere
FEATURES_SINGLE = ['turnout_rate', 'relative_vote_margin', 'log_total_voters', 'state', 'region', 'standardized_coalition']
FEATURES_CROSS = ['turnout_rate', 'relative_vote_margin', 'log_total_voters', 'standardized_coalition']

def create_pipeline(features, use_categorical=True):
    """Helper to create a standard RF pipeline for any function."""
    if use_categorical:
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['state', 'region', 'standardized_coalition']),
            ('num', 'passthrough', ['turnout_rate', 'relative_vote_margin', 'log_total_voters'])
        ])
    else:
        preprocessor = 'passthrough' 

    return Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def get_database(year=None):
    """Fetches data from Supabase with GE14-safe filtering."""
    query = supabase.table('historical_results').select('*')
    response = query.eq('election_year', year).execute() if year else query.limit(5000).execute()
    
    df = pd.DataFrame(response.data)
    if df.empty: return df

    df['is_winner'] = (df.groupby(['election_year', 'seat_id'])['votes_for_candidate']
                         .transform('max') == df['votes_for_candidate']).astype(int)
    return df

def get_year_analysis(df, year):
    """Trains a model for a single year with full feature set."""
    from gedataset_pipeline import clean_dataset

    # Run clean_dataset so standardized_coalition, state, region,
    # relative_vote_margin, log_total_voters are all present
    df = clean_dataset(df)

    data = df[df['election_year'].astype(str).str.upper() == year.upper()].copy()
    if data.empty: return None

    # Verify all required features exist after cleaning
    missing = [f for f in FEATURES_SINGLE if f not in data.columns]
    if missing:
        print(f"  ⚠️  Missing features for {year}: {missing}")
        return None

    # Recalculate is_winner if not present
    if 'is_winner' not in data.columns:
        max_v = data.groupby('seat_id')['votes_for_candidate'].transform('max')
        data['is_winner'] = ((data['votes_for_candidate'] == max_v) & (max_v > 0)).astype(int)

    X, y = data[FEATURES_SINGLE], data['is_winner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_pipeline(FEATURES_SINGLE)
    model.fit(X_train, y_train)
    return model

def get_cross_comparison(df, train_year, test_year):
    train_df = df[df['election_year'] == train_year]
    test_df = df[df['election_year'] == test_year]

    # Use the features common to both
    features = ['turnout_rate', 'relative_vote_margin', 'log_total_voters']
    
    X_train, y_train = train_df[features], train_df['is_winner']
    X_test, y_test = test_df[features], test_df['is_winner']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    
    # --- DATA TO SEND BACK ---
    acc = accuracy_score(y_test, predictions)
    report_dict = classification_report(y_test, predictions, output_dict=True)
    
    # Logic to find which regions the model struggled with
    test_df = test_df.copy()
    test_df['correct'] = (predictions == y_test)
    regional_breakdown = test_df.groupby('region')['correct'].mean().to_dict()

    return {
        "accuracy": acc,
        "regional_accuracy": regional_breakdown,
        "detailed_metrics": report_dict
    }

def evaluate_performance(model, df, year):
    """Generates the detailed GE performance scoreboard for the user."""
    X, y = df[FEATURES_SINGLE], df['is_winner']
    df['prediction'] = model.predict(X)
    df['correct'] = (df['prediction'] == y)

    # Calculate metrics
    overall_acc = accuracy_score(y, df['prediction'])
    regional_acc = df.groupby('region')['correct'].mean().sort_values(ascending=False)
    report = classification_report(y, df['prediction'])
    
    print(f"\n================ GE{year} PERFORMANCE REPORT ================")
    print(f"NATIONAL ACCURACY: {overall_acc:.2%}")
    print("\nREGIONAL PATTERNS:\n", regional_acc)
    print("\nDETAILED CLASSIFICATION:\n", report)
    
    return regional_acc.to_dict()