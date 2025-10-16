#REFACTORIZADO, NO USAR
#SE VALIDA QUE TENGA EL MISMO RESULTADO QUE dataset.py ORIGINAL

from pathlib import Path
import pandas as pd
import logging
from acoustic_ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TURKISH_MODIFIED, TURKISH_ORIGINAL, CLEANED_FILENAME, PROCESSED_PATTERN


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetConfig:
    RAW_DIR = RAW_DATA_DIR
    PROCESSED_DIR = PROCESSED_DATA_DIR
    TURKISH_ORIGINAL = TURKISH_ORIGINAL
    TURKISH_MODIFIED = TURKISH_MODIFIED
    CLEANED_FILENAME = CLEANED_FILENAME
    PROCESSED_PATTERN = PROCESSED_PATTERN

class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class DatasetManager(metaclass=SingletonMeta):
    def __init__(self, config: DatasetConfig = DatasetConfig):
        self.config = config

    def _load_csv(self, filepath: Path, lazy: bool = False) -> pd.DataFrame | None:
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        if lazy:
            logger.info(f"CSV pendiente de carga (lazy): {filepath.name}")
            return None
        logger.info(f"Cargando: {filepath.name}")
        df = pd.read_csv(filepath)
        logger.info(f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df

    def load_original(self, lazy: bool = False) -> pd.DataFrame | None:
        return self._load_csv(self.config.RAW_DIR / self.config.TURKISH_ORIGINAL, lazy)

    def load_modified(self, lazy: bool = False) -> pd.DataFrame | None:
        return self._load_csv(self.config.RAW_DIR / self.config.TURKISH_MODIFIED, lazy)

    def load_cleaned(self, lazy: bool = False) -> pd.DataFrame | None:
        return self._load_csv(self.config.PROCESSED_DIR / self.config.CLEANED_FILENAME, lazy)

    def load_processed(self, version: str = "v2", lazy: bool = False) -> pd.DataFrame | None:
        filename = self.config.PROCESSED_PATTERN.format(version=version)
        return self._load_csv(self.config.PROCESSED_DIR / filename, lazy)

    def save(self, df: pd.DataFrame, filename: str) -> None:
        path = self.config.PROCESSED_DIR / filename
        df.to_csv(path, index=False)
        logger.info(f"Datos guardados en {path}")


    @staticmethod
    def dataset_info(df: pd.DataFrame) -> None:
        logger.info("Información del Dataset")
        logger.info("=" * 60)
        logger.info(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        logger.info(f"Memoria: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        logger.info(f"Nulos: {df.isnull().sum().sum():,}")
        for col in df.columns:
            dtype = df[col].dtype
            nulls = df[col].isnull().sum()
            logger.info(f"   • {col:30s} | {str(dtype):10s} | Nulls: {nulls:,}")


    def load_train_test_split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        paths = {
            "X_train": self.config.PROCESSED_DIR / "X_train.csv",
            "X_test": self.config.PROCESSED_DIR / "X_test.csv",
            "y_train": self.config.PROCESSED_DIR / "y_train.csv",
            "y_test": self.config.PROCESSED_DIR / "y_test.csv",
        }

        missing = [p for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Archivos faltantes: {missing}")

        X_train = pd.read_csv(paths["X_train"])
        X_test = pd.read_csv(paths["X_test"])
        y_train = pd.read_csv(paths["y_train"])['Class']
        y_test = pd.read_csv(paths["y_test"])['Class']

        logger.info("Conjuntos train/test cargados correctamente")
        return X_train, X_test, y_train, y_test
