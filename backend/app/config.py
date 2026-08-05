import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

class Settings:
    # API
    API_TITLE:   str = "GE-Insights State Election Predictor"
    API_VERSION: str = "1.0.0"
    DEBUG:       bool = os.getenv("DEBUG", "True") == "True"

    # Paths
    MODELS_DIR: Path = ROOT / "backend" / "models"
    DATA_DIR:   Path = ROOT / "data"

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://yourdomain.com",
    ]

    # States
    VALID_STATES: list = ["johor", "neg_sembilan", "melaka"]

settings = Settings()