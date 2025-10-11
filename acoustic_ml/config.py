"""
Configuración del proyecto
"""
import os
from pathlib import Path

# Directorios base
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# Subdirectorios de datos
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Configuración de MLflow
MLFLOW_TRACKING_URI = "file:///" + str(PROJECT_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "acoustic_ml_experiments"

# Configuración de DVC
DVC_REMOTE_NAME = "local"
DVC_REMOTE_URL = str(PROJECT_DIR / "dvcstore")

# Datasets Turcos
TURKISH_ORIGINAL = "turkis_music_emotion_original.csv"
TURKISH_MODIFIED = "turkish_music_emotion_modified.csv"

# Seed para reproducibilidad
RANDOM_STATE = 42
