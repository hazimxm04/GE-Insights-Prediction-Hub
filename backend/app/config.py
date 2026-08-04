from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    MODELS_PATH: str = "./backend/models"
    DATA_PATH: str = "./data"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env.local"

settings = Settings()
