import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from contextlib import contextmanager
import threading
from acoustic_ml.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, TURKISH_ORIGINAL, CLEANED_FILENAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatasetManager(metaclass=SingletonMeta):
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("DataFrame vacío")
        logger.info(f"Loaded {path.name} with shape {df.shape}")
        return df

    def load_original(self) -> pd.DataFrame:
        return self._load_csv(self.raw_dir / TURKISH_ORIGINAL)

    def save(self, df: pd.DataFrame, filename: str) -> Path:
        path = self.processed_dir / filename
        df.to_csv(path, index=False)
        logger.info(f"Saved {filename}")
        return path

    def load_processed(self, filename: str = CLEANED_FILENAME) -> pd.DataFrame:
        return self._load_csv(self.processed_dir / filename)

    def get_train_test_split(
        self, target_column: str = "Class", test_size: float = 0.2, random_state: int = 42
    ):
        df = self.load_processed()
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} no existe")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
