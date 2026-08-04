from fastapi import APIRouter
from pydantic import BaseModel
from core.models.state_predictor import StatePredictor

router = APIRouter(prefix="/predict")

class PredictionRequest(BaseModel):
    swing: float = 0.0
    turnout: float = 1.0

@router.post("/{state}")
async def predict_state(state: str, request: PredictionRequest):
    predictor = StatePredictor(state)
    # ... logic
    return {"predictions": [...]}