from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    APP_NAME: str = "mlops-fastapi"
    ENV: str = "dev"
    MODEL_PATH: str = "app/models/model.joblib"
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    LOG_LEVEL: str = "INFO"
    PROJECT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_DIR / "data"
    MODELS_DIR: Path = PROJECT_DIR / "models"

settings = Settings()
