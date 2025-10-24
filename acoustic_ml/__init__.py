"""
Acoustic ML - MLOps Team 24
Proyecto de Machine Learning para análisis de características acústicas

Refactorizado con POO + SOLID principles - Fase 2
"""
__version__ = "0.2.0"

# Dataset Manager (POO)
from acoustic_ml.dataset import DatasetManager

# Feature Engineering (POO)
from acoustic_ml.features import (
    FeatureTransformer,
    NumericFeatureSelector,
    PowerFeatureTransformer,
    FeaturePipelineBuilder,
    create_preprocessing_pipeline,
)

# Configuración
from acoustic_ml.config import (
    PROJECT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TURKISH_ORIGINAL,
    TURKISH_MODIFIED,
    RANDOM_STATE,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)

__all__ = [
    "DatasetManager",
    "FeatureTransformer",
    "NumericFeatureSelector",
    "PowerFeatureTransformer",
    "FeaturePipelineBuilder",
    "create_preprocessing_pipeline",
    "PROJECT_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "TURKISH_ORIGINAL",
    "TURKISH_MODIFIED",
    "RANDOM_STATE",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
]
