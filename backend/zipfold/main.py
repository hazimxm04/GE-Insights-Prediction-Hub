from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os
import anthropic

import supabase
from database import get_database, supabase
import gedataset_pipeline as gep
from predictor import get_model
from predictor import prediction_engine
from analysis import get_cross_comparison

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_cache = {}

#use class when request input from user (Request Body)
class SeatInput(BaseModel):
    seat_id: str
    state: str
    party: str
    coalition: str
    total_votes: int         # Standardized with underscores
    votes_for_candidate: int # Standardized with underscores
    turnout_rate: float      # Standardized with underscores
    election_year: str 

class StateInput(BaseModel):
    state_filter: str = "Nationwide"   # State or Region name, or "Nationwide"
    seat_filter: str = ""              # Optional specific seat e.g. "P.001" or "P.001 Padang Besar"
    turnout_rate: float = 80.0
    swing: float = 0.0
    base_voter_rate: float = 50.0      # Base % of votes going to the coalition
    coalition: str = "All"
    election_year: str = "GE14"

@app.get("/seats")
def get_seat_state_list():
    df = get_database()

    menu_data = df[['seat_id', 'seat_name', 'state' ]].to_dict(orient="records")
    return menu_data


@app.post("/predict")
def predict_seat(data: SeatInput):
    # Convert Pydantic model to a dictionary
    input_dict = data.dict()
    
    # Use the refined engine from predictor.py
    verdict, probability = prediction_engine(input_dict, electionyear=data.election_year)
    
    return {
        "seat_id": data.seat_id,
        "verdict": verdict,
        "probability": round(probability, 4),
        "status": "success" if verdict != "ERROR" else "failed"
    }

@app.post("/predict-summary")
def state_prediction(data: StateInput):
    try:
        # 1. Standardize Input
        target_year = str(data.election_year).strip().upper()
        df = get_database(year=target_year)
        
        if df.empty:
            return {"error": f"No data found for election year: {target_year}"}
        
        # 2. Pipeline Cleaning
        df = gep.clean_dataset(df)

        # Deduplicate columns (common cause of errors)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # 3. Filtering
        search_state = data.state_filter.strip().lower()
        search_coalition = data.coalition.strip().lower()
        search_seat = data.seat_filter.strip().lower() if data.seat_filter else ""

        # Seat filter takes priority (most specific)
        if search_seat:
            # seat_filter can be "P.001" or "P.001 Padang Besar" — match on seat_id prefix
            seat_id_part = search_seat.split(" ")[0]  # extract just "P.001"
            df = df[df["seat_id"].str.lower() == seat_id_part].copy()
        elif search_state != "nationwide":
            df = df[(df["state"].str.lower() == search_state) | (df["region"].str.lower() == search_state)].copy()

        # FIX: Use standardized_coalition (created by clean_dataset) not raw coalition column
        if search_coalition != "all":
            coalition_col = "standardized_coalition" if "standardized_coalition" in df.columns else "coalition"
            df = df[df[coalition_col].str.lower() == search_coalition].copy()

        if df.empty:
            return {"summary": {"wins": 0, "total_seats": 0}, "message": "No data found."}

        # 4. Math Logic (Now 100% safe from the "arg must be a list" error)
        # Force these specific columns to be numeric
        def get_1d_col(col_name):
            if col_name not in df.columns:
                return pd.Series(0, index=df.index)
            val = df[col_name]
            # Force 1D
            if isinstance(val, pd.DataFrame):
                val = val.iloc[:, 0]
            return pd.to_numeric(val, errors='coerce').fillna(0)
        # 4. PERFORM CALCULATIONS ON GUARANTEED 1D OBJECTS
        votes = get_1d_col('votes_for_candidate')
        total = get_1d_col('total_votes')

        df['hist_share'] = (votes / total.replace(0, 1)) * 100
        # base_voter_rate scales the historical share (50 = no change, 60 = boost, 40 = reduce)
        effective_share = df['hist_share'] * (data.base_voter_rate / 50.0) if data.base_voter_rate != 50 else df['hist_share']
        # Apply swing on top, then clamp to valid range [0, 100] so model gets clean inputs
        swung_share = (effective_share + float(data.swing)).clip(0, 100)
        df['sim_turnout'] = total * (data.turnout_rate / 100)
        # sim_votes feeds into clean_dataset() which recomputes relative_vote_margin
        # THAT is how swing reaches the model — it never sees 'swing' directly
        df['sim_votes'] = df['sim_turnout'] * (swung_share / 100)

        # 5. Prediction Compilation
        df_pipe = df.rename(columns={'sim_votes': 'votes_for_candidate', 'sim_turnout': 'total_votes'}).copy()
        processed_df = gep.clean_dataset(df_pipe)
        
        current_model = get_model(target_year)
        features = processed_df[current_model.feature_names_in_]
        
        # Predict probability for every individual seat
        processed_df['win_probability'] = current_model.predict_proba(features)[:, 1]
        processed_df['verdict_num'] = (processed_df['win_probability'] > 0.5).astype(int)

        # 6. Final State Summary
        total_wins = int(processed_df['verdict_num'].sum())
        
        return {
            "summary": {
                "view": data.state_filter,
                "coalition": data.coalition,
                "total_seats": len(processed_df),
                "wins": total_wins,
                "losses": len(processed_df) - total_wins
            },
            "seats": processed_df[['seat_id', 'seat_name', 'verdict_num', 'win_probability']].to_dict(orient="records")
        }

    except Exception as e:
        print(f"CRASH ERROR: {e}")
        return {"error": "Internal Server Error", "details": str(e)}

@app.get("/api/options")
def get_filter_options(year: Optional[str] = None):
    all_years = []
    try:
        # STEP A: Get all available years
        year_check = supabase.table('historical_results').select('election_year').execute()
        all_years = sorted(list(set([r['election_year'] for r in year_check.data if r.get('election_year')])), reverse=True)

        # STEP B: Fetch data for the requested year
        df = get_database(year=year)

        if df.empty:
            return {"states": [], "regions": [], "years": all_years, "seats": [], "coalitions": []}

        # STEP C: Run through pipeline to get standardized columns
        df_clean = gep.clean_dataset(df)

        # Build seat list safely — handle both 'seat_name' and 'seat_id' only
        if 'seat_name' in df_clean.columns and 'seat_id' in df_clean.columns:
            seat_list = sorted((df_clean['seat_id'] + " " + df_clean['seat_name']).unique().tolist())
        elif 'seat_id' in df_clean.columns:
            seat_list = sorted(df_clean['seat_id'].unique().tolist())
        else:
            seat_list = []

        # Coalitions from standardized_coalition (created by clean_dataset)
        coalition_col = "standardized_coalition" if "standardized_coalition" in df_clean.columns else "coalition"
        coalitions = sorted(df_clean[coalition_col].dropna().unique().tolist()) if coalition_col in df_clean.columns else []

        # Filter out Unknown from states/regions
        states = sorted([s for s in df_clean['state'].unique().tolist() if s != 'Unknown']) if 'state' in df_clean.columns else []
        regions = sorted([r for r in df_clean['region'].unique().tolist() if r != 'Unknown']) if 'region' in df_clean.columns else []

        return {
            "states": states,
            "regions": regions,
            "years": all_years,
            "seats": seat_list,
            "coalitions": coalitions
        }
    except Exception as e:
        print(f"OPTIONS ERROR: {e}")
        # Always return valid shape even on error so frontend doesnt break
        return {"states": [], "regions": [], "years": all_years, "seats": [], "coalitions": [], "error": str(e)}

@app.get("/api/compare different datasets")
def compare_election_years(train_year: str, test_year: str):
    try:
        raw_df = get_database()
        df = get_database(raw_df) # This gets the full 5000 rows
        
        # Get the detailed dictionary from analysis.py
        report_data = get_cross_comparison(df, train_year, test_year)
        
        return {
            "status": "success",
            "train_year": train_year,
            "test_year": test_year,
            "report": report_data  # <--- THIS puts the report in your browser
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/available-years")
def get_years():
    try:
        df = get_database() # Fetch your data
        # Get unique years, sort them, and return as a list
        years = sorted(df['election_year'].unique().tolist())
        return {"years": years}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/compare")
async def compare_years(
    train_year:  str,
    test_year:   str,
    coalition_a: str = "All",   # coalition filter for Dataset A (train_year)
    coalition_b: str = "All",   # coalition filter for Dataset B (test_year)
    scope:       str = "Nationwide",
    target:      str = ""
):
    try:
        train_year  = train_year.strip().upper()
        test_year   = test_year.strip().upper()
        coalition_a = coalition_a.strip()
        coalition_b = coalition_b.strip()
        scope       = scope.strip()
        target      = target.strip().lower()

        # 1. Fetch + clean both datasets
        train_df = gep.clean_dataset(get_database(year=train_year))
        test_df  = gep.clean_dataset(get_database(year=test_year))

        if train_df.empty:
            return {"status": "error", "message": f"No data for {train_year}"}
        if test_df.empty:
            return {"status": "error", "message": f"No data for {test_year}"}

        def apply_scope_filter(df):
            """Apply only scope/target filter — coalition filtered separately per dataset."""
            if scope.lower() == "state" and target:
                df = df[df["state"].str.lower() == target].copy()
            elif scope.lower() == "region" and target:
                df = df[df["region"].str.lower() == target].copy()
            elif scope.lower() == "seat" and target:
                seat_id = target.split(" ")[0]
                df = df[df["seat_id"].str.lower() == seat_id].copy()
            return df

        def apply_coalition_filter(df, coalition):
            if coalition.lower() != "all":
                col = "standardized_coalition" if "standardized_coalition" in df.columns else "coalition"
                df = df[df[col].str.lower() == coalition.lower()].copy()
            return df

        # Apply scope first (same for both), then each dataset gets its own coalition filter
        train_scoped = apply_scope_filter(train_df)
        test_scoped  = apply_scope_filter(test_df)

        train_filtered = apply_coalition_filter(train_scoped, coalition_a)
        test_filtered  = apply_coalition_filter(test_scoped,  coalition_b)

        if train_filtered.empty and coalition_a.lower() != "all":
            return {"status": "error", "message": f"No data for {coalition_a} in {train_year}"}
        if test_filtered.empty and coalition_b.lower() != "all":
            return {"status": "error", "message": f"No data for {coalition_b} in {test_year}"}

        # 2. Full seat summary (unfiltered by coalition, for regional/flipped analysis)
        def seat_summary(df):
            col = "standardized_coalition" if "standardized_coalition" in df.columns else "coalition"
            rows = []
            for (sid, sname, state, region), grp in df.groupby(["seat_id", "seat_name", "state", "region"]):
                winner_row = grp.loc[grp["votes_for_candidate"].idxmax()]
                rows.append({
                    "seat_id":          sid,
                    "seat_name":        sname,
                    "state":            state,
                    "region":           region,
                    "winner_coalition": winner_row[col] if col in winner_row.index else "Unknown",
                    "votes":            int(winner_row["votes_for_candidate"]),
                    "total":            int(winner_row["total_votes"]) if "total_votes" in winner_row.index else 0,
                })
            return pd.DataFrame(rows)

        train_summary = seat_summary(train_scoped)
        test_summary  = seat_summary(test_scoped)

        # 3. Coalition-specific stats (respects coalition_a/b filter)
        def coalition_stats(filtered_df, full_summary_df, year_label, coalition_filter):
            col_field = "winner_coalition"
            total_seats = len(full_summary_df)

            if coalition_filter.lower() == "all":
                # Show all coalitions
                wins_by_coalition = full_summary_df[col_field].value_counts().to_dict()
            else:
                # Count wins for filtered coalition only
                wins = int((full_summary_df[col_field].str.lower() == coalition_filter.lower()).sum())
                wins_by_coalition = {coalition_filter: wins}

            return {
                "year":               year_label,
                "total_seats":        total_seats,
                "coalition_filter":   coalition_filter,
                "wins_by_coalition":  wins_by_coalition,
                "win_rate_by_coalition": {
                    k: round(v / total_seats * 100, 1) if total_seats > 0 else 0
                    for k, v in wins_by_coalition.items()
                }
            }

        train_stats = coalition_stats(train_filtered, train_summary, train_year, coalition_a)
        test_stats  = coalition_stats(test_filtered,  test_summary,  test_year,  coalition_b)

        # 4. Head-to-head: coalition_a in train vs coalition_b in test
        a_wins  = train_stats["wins_by_coalition"].get(coalition_a, sum(train_stats["wins_by_coalition"].values()) if coalition_a == "All" else 0)
        b_wins  = test_stats["wins_by_coalition"].get(coalition_b,  sum(test_stats["wins_by_coalition"].values())  if coalition_b == "All" else 0)
        a_rate  = train_stats["win_rate_by_coalition"].get(coalition_a, 0)
        b_rate  = test_stats["win_rate_by_coalition"].get(coalition_b, 0)

        head_to_head = {
            "a": {"year": train_year, "coalition": coalition_a, "wins": a_wins, "win_rate": a_rate},
            "b": {"year": test_year,  "coalition": coalition_b, "wins": b_wins, "win_rate": b_rate},
            "delta_wins": b_wins - a_wins,
            "delta_rate": round(b_rate - a_rate, 1)
        }

        # 5. Regional breakdown (full, unfiltered by coalition)
        def regional_breakdown(summary_df, coalition_filter):
            result = {}
            for region, grp in summary_df.groupby("region"):
                all_wins = grp["winner_coalition"].value_counts().to_dict()
                if coalition_filter.lower() != "all":
                    focused = int((grp["winner_coalition"].str.lower() == coalition_filter.lower()).sum())
                    result[region] = {"total": len(grp), "wins": all_wins, "focused_wins": focused}
                else:
                    result[region] = {"total": len(grp), "wins": all_wins, "focused_wins": None}
            return result

        # 6. Seats that changed hands (scope-filtered, not coalition-filtered)
        if not train_summary.empty and not test_summary.empty:
            merged = pd.merge(
                train_summary[["seat_id", "seat_name", "state", "winner_coalition"]].rename(columns={"winner_coalition": "winner_" + train_year}),
                test_summary[["seat_id", "winner_coalition"]].rename(columns={"winner_coalition": "winner_" + test_year}),
                on="seat_id", how="inner"
            )
            merged["changed"] = merged["winner_" + train_year] != merged["winner_" + test_year]
            flipped_seats = merged[merged["changed"]][["seat_id", "seat_name", "state", "winner_" + train_year, "winner_" + test_year]].to_dict(orient="records")
        else:
            flipped_seats = []

        return {
            "status":     "success",
            "train_year": train_year,
            "test_year":  test_year,
            "coalition_a": coalition_a,
            "coalition_b": coalition_b,
            "scope":      scope,
            "target":     target,
            "report": {
                "head_to_head":   head_to_head,
                "train":          train_stats,
                "test":           test_stats,
                "regional_train": regional_breakdown(train_summary, coalition_a),
                "regional_test":  regional_breakdown(test_summary,  coalition_b),
                "flipped_seats":  flipped_seats,
                "flipped_count":  len(flipped_seats)
            }
        }
    except Exception as e:
        print(f"COMPARE ERROR: {e}")
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ── LLM EXPLANATION LAYER ──────────────────────────────────────────────────

class ExplainInput(BaseModel):
    mode: str                    # "predict" or "compare"
    year: str                    # e.g. "GE14"
    scope: str                   # "Seat", "State", "Region"
    target: str                  # e.g. "P.106 Damansara" or "Selangor"
    coalition: str               # e.g. "Harapan"
    outcome: str                 # e.g. "WIN" or "14 / 22 Seats Won"
    probability: float           # 0-100
    swing: float = 0.0
    turnout: float = 80.0
    # Compare mode extras
    coalition_b: str = ""
    year_b: str = ""
    delta_wins: int = 0
    delta_rate: float = 0.0
    flipped_count: int = 0

@app.post("/api/explain")
def explain_prediction(data: ExplainInput):
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return {"status": "error", "explanation": "ANTHROPIC_API_KEY not set.", "details": "Add it to your environment"}
        client = anthropic.Anthropic(api_key=api_key)

        if data.mode == "predict":
            prompt = f"""You are an expert Malaysian political analyst. A machine learning model just made this election prediction:

- Dataset: {data.year} Malaysian General Election
- Scope: {data.scope} — {data.target}
- Coalition analysed: {data.coalition}
- Outcome: {data.outcome}
- Win probability: {data.probability:.1f}%
- Swing rate applied: {data.swing:+.1f}%
- Turnout rate: {data.turnout:.0f}%

Write a concise 3-sentence plain-English analysis of this prediction result. 
Sentence 1: summarise the outcome and what it means.
Sentence 2: comment on what the probability level suggests about competitiveness.
Sentence 3: note how the swing/turnout settings may have influenced this result.
Be specific, analytical, and direct. Do not use bullet points."""

        else:  # compare mode
            direction = "gained" if data.delta_wins > 0 else "lost" if data.delta_wins < 0 else "held steady at"
            prompt = f"""You are an expert Malaysian political analyst. A cross-election comparison was just run:

- Comparing: {data.year} vs {data.year_b}
- Coalition A ({data.year}): {data.coalition} — {data.outcome.split('|')[0].strip()}
- Coalition B ({data.year_b}): {data.coalition_b} — {data.outcome.split('|')[1].strip() if '|' in data.outcome else 'N/A'}
- Seat change: {data.coalition_b} {direction} {abs(data.delta_wins)} seats ({data.delta_rate:+.1f}%)
- Seats that changed hands: {data.flipped_count}

Write a concise 3-sentence political analysis of this cross-election comparison.
Sentence 1: describe what changed between the two elections for these coalitions.
Sentence 2: interpret what the seat swing and flipped count reveals about voter movement.
Sentence 3: give one possible real-world political factor that could explain this shift.
Be specific, analytical, and direct. Do not use bullet points."""

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "status": "success",
            "explanation": message.content[0].text
        }

    except Exception as e:
        print(f"EXPLAIN ERROR: {e}")
        return {"status": "error", "explanation": "Analysis unavailable.", "details": str(e)}


@app.get("/api/model-report")
def get_model_report(year: str = "GE14"):
    """Returns the saved evaluation metrics for a trained model year."""
    import json
    year_upper = year.strip().upper()
    report_path = os.path.join(BASE_DIR, f"model_report_{year_upper}.json")
    print(f"Looking for report at: {report_path}")
    if not os.path.exists(report_path):
        # List what IS in the directory to help debug
        existing = [f for f in os.listdir(BASE_DIR) if f.startswith("model_report")]
        print(f"Existing reports: {existing}")
        return {
            "status": "error",
            "message": f"No report found for {year_upper}. Run model_training.py first.",
            "looked_for": report_path,
            "existing_reports": existing
        }
    with open(report_path) as f:
        return {"status": "success", "year": year_upper, "report": json.load(f)}