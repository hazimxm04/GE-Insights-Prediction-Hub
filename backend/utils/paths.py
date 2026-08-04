from pathlib import Path
from app.config import settings

def get_model_path(state: str, model_type: str):
    return Path(settings.MODELS_PATH) / state / f"{model_type}_model.pkl"

def get_data_path(filename: str):
    return Path(settings.DATA_PATH) / "raw" / filename