"""
Acoustic ML - MLOps Team 24
Proyecto de Machine Learning para análisis de características acústicas
"""

__version__ = "0.1.0"

from .dataset import (
    load_raw_data,
    save_processed_data,
    load_turkish_original,
    load_turkish_modified,
    get_dataset_info,
)

from .config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TURKISH_ORIGINAL,
    TURKISH_MODIFIED,
)

__all__ = [
    "load_raw_data",
    "save_processed_data",
    "load_turkish_original",
    "load_turkish_modified",
    "get_dataset_info",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "RANDOM_STATE",
    "TURKISH_ORIGINAL",
    "TURKISH_MODIFIED",
]
