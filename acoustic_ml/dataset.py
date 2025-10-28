from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from contextlib import contextmanager
import threading
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from acoustic_ml.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TURKISH_MODIFIED,
    TURKISH_ORIGINAL,
    CLEANED_FILENAME,
    PROCESSED_PATTERN
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetConfig:
    RAW_DIR = RAW_DATA_DIR
    PROCESSED_DIR = PROCESSED_DATA_DIR
    TURKISH_ORIGINAL = TURKISH_ORIGINAL
    TURKISH_MODIFIED = TURKISH_MODIFIED
    CLEANED_FILENAME = CLEANED_FILENAME
    PROCESSED_PATTERN = PROCESSED_PATTERN

    @classmethod
    def validate_dirs(cls) -> None:
        for name, path in [("RAW_DIR", cls.RAW_DIR), ("PROCESSED_DIR", cls.PROCESSED_DIR)]:
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        logger.info("Data directories validated.")

    @classmethod
    def list_files(cls) -> Dict[str, List[Path]]:
        raw = list(cls.RAW_DIR.glob("*.csv")) if cls.RAW_DIR.exists() else []
        processed = list(cls.PROCESSED_DIR.glob("*.csv")) if cls.PROCESSED_DIR.exists() else []
        return {"raw": raw, "processed": processed, "total": len(raw)+len(processed)}


class SingletonMeta(type):
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
                    logger.debug(f"Singleton created: {cls.__name__}")
        return cls._instances[cls]

    @classmethod
    def clear_instances(cls) -> None:
        with cls._lock:
            cls._instances.clear()
            logger.warning("All singleton instances cleared.")


class DatasetValidator:

    @staticmethod
    def validate_df(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1, check_nulls: bool = True) -> bool:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty DataFrame")
        if df.shape[0] < min_rows:
            raise ValueError(f"DataFrame rows {df.shape[0]} < {min_rows}")
        if df.shape[1] < min_cols:
            raise ValueError(f"DataFrame columns {df.shape[1]} < {min_cols}")
        if check_nulls:
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                logger.warning(f"{null_count} nulls detected ({null_count/df.size*100:.2f}%)")
        logger.info(f"DataFrame validated: {df.shape}")
        return True

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        logger.info(f"Required columns present: {required_cols}")
        return True

    @staticmethod
    def validate_target(y: pd.Series, expected_classes: Optional[List[str]] = None) -> bool:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Target DataFrame must have 1 column")
            y = y.iloc[:, 0]
        unique_classes = y.unique()
        if expected_classes:
            missing = set(expected_classes) - set(unique_classes)
            extra = set(unique_classes) - set(expected_classes)
            if missing: logger.warning(f"Missing expected classes: {missing}")
            if extra: logger.warning(f"Unexpected classes: {extra}")
        logger.info(f"Target classes: {unique_classes}")
        return True

    @staticmethod
    def validate_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> bool:
        if X_train.shape[0] != len(y_train) or X_test.shape[0] != len(y_test):
            raise ValueError("Mismatch in rows between X and y")
        if not X_train.columns.equals(X_test.columns):
            raise ValueError("Train/Test columns mismatch")
        logger.info("Train/Test split validated")
        return True


class DatasetStatistics:

    @staticmethod
    def summary(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum()/1024/1024,
            "null_count": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum()
        }

    @staticmethod
    def correlation(df: pd.DataFrame, method: str = "pearson", threshold: float = 0.8) -> Tuple[pd.DataFrame, List[Tuple[str,str,float]]]:
        numeric = df.select_dtypes(include=[np.number])
        corr_matrix = numeric.corr(method=method)
        high_corr = [(i,j,corr_matrix.loc[i,j]) for i in numeric.columns for j in numeric.columns if i<j and abs(corr_matrix.loc[i,j])>=threshold]
        logger.info(f"{len(high_corr)} high correlation pairs found")
        return corr_matrix, high_corr

    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5) -> Dict[str, pd.Series]:
        numeric = df.select_dtypes(include=[np.number])
        outliers = {}
        for col in numeric.columns:
            if method == "iqr":
                q1, q3 = numeric[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                mask = (numeric[col] < q1 - threshold*iqr) | (numeric[col] > q3 + threshold*iqr)
            elif method == "zscore":
                mask = np.abs((numeric[col] - numeric[col].mean()) / numeric[col].std()) > threshold
            else:
                raise ValueError(f"Invalid method {method}")
            outliers[col] = mask
            if mask.sum() > 0:
                logger.info(f"{mask.sum()} outliers in {col}")
        return outliers


class DatasetManager(metaclass=SingletonMeta):

    def __init__(self, config: type = DatasetConfig):
        self.config = config
        self.validator = DatasetValidator()
        self.statistics = DatasetStatistics()
        try:
            self.config.validate_dirs()
        except FileNotFoundError as e:
            logger.warning(e)

    def _load_csv(self, path: Path, validate: bool = True) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if validate: self.validator.validate_df(df)
        logger.info(f"Loaded {path.name} ({df.shape})")
        return df

    def load_original(self) -> pd.DataFrame:
        return self._load_csv(self.config.RAW_DIR / self.config.TURKISH_ORIGINAL)

    def load_modified(self) -> pd.DataFrame:
        return self._load_csv(self.config.RAW_DIR / self.config.TURKISH_MODIFIED)

    def load_cleaned(self) -> pd.DataFrame:
        return self._load_csv(self.config.PROCESSED_DIR / self.config.CLEANED_FILENAME)

    def load_processed(self, version: str = "v2") -> pd.DataFrame:
        filename = self.config.PROCESSED_PATTERN.format(version=version)
        return self._load_csv(self.config.PROCESSED_DIR / filename)

    def save(self, df: pd.DataFrame, filename: str, backup: bool = False) -> Path:
        path = self.config.PROCESSED_DIR / filename
        if backup and path.exists():
            backup_path = path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            path.rename(backup_path)
            logger.info(f"Backup created: {backup_path.name}")
        df.to_csv(path, index=False)
        logger.info(f"Saved {filename}")
        return path

    @contextmanager
    def load_context(self, path: Path):
        df = self._load_csv(path)
        try:
            yield df
        finally:
            del df
            logger.debug("DataFrame released from memory")


    def get_train_test_split(
        self,
        version: str = "v2",
        target_column: str = "label",  # Ajusta al nombre de tu target
        test_size: float = 0.2,
        random_state: int = 42,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Retorna X_train, X_test, y_train, y_test para MLflow y pipelines.
        """
        df = self.load_processed(version=version)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )

        if validate:
            self.validator.validate_train_test(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test