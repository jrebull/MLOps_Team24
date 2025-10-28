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

    def get_train_test_split(
        self, target_column: str = "Class", test_size: float = 0.2, random_state: int = 42
    ):
        df = self.load_processed()
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} no existe")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def load_train_test_split(self, validate: bool = False):
        """
        Load pre-saved train-test split files from PROCESSED_DATA_DIR.
        
        This method loads the four CSV files created by a previous split:
        - X_train.csv, X_test.csv, y_train.csv, y_test.csv
        
        Args:
            validate: If True, performs integrity checks on loaded data
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        
        Raises:
            FileNotFoundError: If any split file is missing
            ValueError: If validation fails (shapes mismatch, empty data, etc.)
        
        MLOps Best Practice:
            Loads existing splits for reproducibility across experiments.
            Use get_train_test_split() to generate NEW splits.
        """
        split_files = {
            'X_train': self.processed_dir / 'X_train.csv',
            'X_test': self.processed_dir / 'X_test.csv',
            'y_train': self.processed_dir / 'y_train.csv',
            'y_test': self.processed_dir / 'y_test.csv'
        }
        
        # Check all files exist
        missing = [name for name, path in split_files.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing split files: {missing}. "
                f"Run get_train_test_split() first to generate splits."
            )
        
        # Load all splits
        X_train = pd.read_csv(split_files['X_train'])
        X_test = pd.read_csv(split_files['X_test'])
        y_train = pd.read_csv(split_files['y_train']).squeeze()  # Series
        y_test = pd.read_csv(split_files['y_test']).squeeze()    # Series
        
        logger.info(f"Loaded splits: X_train{X_train.shape}, X_test{X_test.shape}, "
                    f"y_train{y_train.shape}, y_test{y_test.shape}")
        
        # Validation checks
        if validate:
            # Check no empty datasets
            if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                raise ValueError("One or more splits are empty")
            
            # Check X and y have matching samples
            if len(X_train) != len(y_train):
                raise ValueError(f"X_train ({len(X_train)}) and y_train ({len(y_train)}) size mismatch")
            if len(X_test) != len(y_test):
                raise ValueError(f"X_test ({len(X_test)}) and y_test ({len(y_test)}) size mismatch")
            
            # Check feature consistency
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(f"Feature mismatch: X_train has {X_train.shape[1]} features, "
                               f"X_test has {X_test.shape[1]} features")
            
            logger.info("✅ Validation passed: All splits are consistent")
        
        return X_train, X_test, y_train, y_test