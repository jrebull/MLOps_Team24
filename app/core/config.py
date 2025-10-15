from pydantic import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "mlops-fastapi"
    ENV: str = "dev"
    MODEL_PATH: str = "app/models/model.joblib"
    MLFLOW_TRACKING_URI: str | None = None
    LOG_LEVEL: str = "INFO"
    # add DB, S3 creds, API keys...
    class Config:
        env_file = ".env"

settings = Settings()
