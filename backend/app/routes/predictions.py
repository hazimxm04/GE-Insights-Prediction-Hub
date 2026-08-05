import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.core.models.state_predictor import StatePredictor

router = APIRouter(prefix="/predict", tags=["predictions"])

# Cache predictors (load once, reuse)
_predictors: dict = {}

def get_predictor(state: str) -> StatePredictor:
    if state not in _predictors:
        try:
            _predictors[state] = StatePredictor(state)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load {state} model: {str(e)}"
            )
    return _predictors[state]

# ── Request / Response models ─────────────────────────────────────

class SeatPredictionRequest(BaseModel):
    seat_name:            str
    majority_change:      float = 0.0
    turnout_change:       float = 0.0
    incumbent_held:       int   = 0
    log_voters:           float = 10.5
    majority_perc_change: float = 0.0
    n_candidates_b:       int   = 3

class SeatPredictionResponse(BaseModel):
    seat_name:       str
    prediction:      str
    probability:     float
    confidence:      str
    is_ood:          bool
    ood_score:       float
    fallback_used:   bool
    warning:         str | None

class StatePredictionResponse(BaseModel):
    state:        str
    total_seats:  int
    predicted_bn:    int
    predicted_non_bn: int
    ood_flagged:  int
    predictions:  list

# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/states")
async def get_valid_states():
    """List available states"""
    return {
        "states": [
            {"id": "johor",        "name": "Johor",           "seats": 56},
            {"id": "neg_sembilan", "name": "Negeri Sembilan", "seats": 36},
            {"id": "melaka",       "name": "Melaka",          "seats": 28},
        ]
    }

@router.post("/seat/{state}", response_model=SeatPredictionResponse)
async def predict_single_seat(state: str, request: SeatPredictionRequest):
    """Predict outcome for a single DUN seat"""
    if state not in ["johor", "neg_sembilan", "melaka"]:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    predictor = get_predictor(state)

    features = {
        "majority_change":      request.majority_change,
        "turnout_change":       request.turnout_change,
        "incumbent_held":       request.incumbent_held,
        "log_voters":           request.log_voters,
        "majority_perc_change": request.majority_perc_change,
        "n_candidates_b":       request.n_candidates_b,
    }

    result = predictor.predict_seat(request.seat_name, features)

    return SeatPredictionResponse(**{
        k: result[k] for k in SeatPredictionResponse.model_fields
    })

@router.get("/all/{state}", response_model=StatePredictionResponse)
async def predict_all_seats(state: str):
    """Predict all DUN seats for a state using 2026 validation data"""
    if state not in ["johor", "neg_sembilan", "melaka"]:
        raise HTTPException(status_code=400, detail=f"Invalid state: {state}")

    predictor = get_predictor(state)
    df = predictor.predict_all()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No validation data for {state}"
        )

    predictions = df.to_dict(orient="records")

    return StatePredictionResponse(
        state=state,
        total_seats=len(df),
        predicted_bn=int((df["prediction"] == "BN").sum()),
        predicted_non_bn=int((df["prediction"] == "non-BN").sum()),
        ood_flagged=int(df["is_ood"].sum()),
        predictions=predictions,
    )