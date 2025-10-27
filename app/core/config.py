from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "mlops-fastapi"
    ENV: str = "dev"
    MODEL_PATH: str = "app/models/model.joblib"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    LOG_LEVEL: str = "INFO"
    PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_DIR / "data"
    MODELS_DIR: Path = PROJECT_DIR / "models"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()