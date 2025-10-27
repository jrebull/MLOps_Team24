from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any
from contextlib import contextmanager
import threading
import warnings
from datetime import datetime

from acoustic_ml.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TURKISH_MODIFIED,
    TURKISH_ORIGINAL,
    CLEANED_FILENAME,
    PROCESSED_PATTERN
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetConfig:
    """
    Configuración centralizada para gestión de datasets.
    
    Almacena todas las rutas y nombres de archivos utilizados
    en el proyecto. Implementa validación de existencia de directorios.
    
    Attributes:
        RAW_DIR: Directorio de datos raw
        PROCESSED_DIR: Directorio de datos procesados
        TURKISH_ORIGINAL: Nombre del archivo original
        TURKISH_MODIFIED: Nombre del archivo modificado
        CLEANED_FILENAME: Nombre del archivo limpio
        PROCESSED_PATTERN: Patrón para archivos procesados versionados
    
    Ejemplo:
        >>> config = DatasetConfig()
        >>> config.validate_directories()
        >>> print(config.RAW_DIR)
    """
    
    RAW_DIR = RAW_DATA_DIR
    PROCESSED_DIR = PROCESSED_DATA_DIR
    TURKISH_ORIGINAL = TURKISH_ORIGINAL
    TURKISH_MODIFIED = TURKISH_MODIFIED
    CLEANED_FILENAME = CLEANED_FILENAME
    PROCESSED_PATTERN = PROCESSED_PATTERN
    
    @classmethod
    def validate_directories(cls) -> None:
        """
        Valida que los directorios necesarios existan.
        
        Raises:
            FileNotFoundError: Si algún directorio no existe
        """
        for dir_name, dir_path in [("RAW_DIR", cls.RAW_DIR), ("PROCESSED_DIR", cls.PROCESSED_DIR)]:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Directorio {dir_name} no encontrado: {dir_path}. "
                    f"Crear directorio o verificar configuración."
                )
        logger.info("Directorios de datos validados correctamente")
    
    @classmethod
    def get_all_available_files(cls) -> Dict[str, List[Path]]:
        """
        Obtiene lista de todos los archivos disponibles.
        
        Returns:
            Diccionario con archivos raw y procesados
        """
        raw_files = list(cls.RAW_DIR.glob("*.csv")) if cls.RAW_DIR.exists() else []
        processed_files = list(cls.PROCESSED_DIR.glob("*.csv")) if cls.PROCESSED_DIR.exists() else []
        
        return {
            "raw": raw_files,
            "processed": processed_files,
            "total": len(raw_files) + len(processed_files)
        }
    
    @classmethod
    def get_config_summary(cls) -> str:
        """
        Genera resumen de la configuración.
        
        Returns:
            String con información de configuración
        """
        files = cls.get_all_available_files()
        summary = f"""
DatasetConfig Summary:
─────────────────────────────────────
RAW_DIR:           {cls.RAW_DIR}
PROCESSED_DIR:     {cls.PROCESSED_DIR}
─────────────────────────────────────
Archivos raw:      {len(files['raw'])}
Archivos procesados: {len(files['processed'])}
Total archivos:    {files['total']}
─────────────────────────────────────
"""
        return summary


class SingletonMeta(type):
    """
    Metaclase thread-safe para implementar patrón Singleton.
    
    Asegura que solo exista una instancia de la clase y que
    múltiples threads no creen instancias duplicadas.
    
    Attributes:
        _instances: Diccionario de instancias singleton
        _lock: Lock para thread safety
    
    Ejemplo:
        >>> class MyClass(metaclass=SingletonMeta):
        ...     pass
        >>> obj1 = MyClass()
        >>> obj2 = MyClass()
        >>> assert obj1 is obj2  # Misma instancia
    """
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Controla la creación de instancias.
        
        Thread-safe: Usa lock para prevenir race conditions.
        """
        # Double-checked locking pattern
        if cls not in cls._instances:
            with cls._lock:
                # Verificar nuevamente dentro del lock
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
                    logger.debug(f"Nueva instancia Singleton creada: {cls.__name__}")
        
        return cls._instances[cls]
    
    @classmethod
    def clear_instances(cls) -> None:
        """
        Limpia todas las instancias singleton (útil para testing).
        
        Warning:
            Solo usar en tests. En producción, el singleton debe persistir.
        """
        with cls._lock:
            cls._instances.clear()
            logger.warning("Todas las instancias Singleton han sido limpiadas")


class DatasetValidator:
    """
    Validador robusto para datasets.
    
    Proporciona métodos para validar integridad, estructura y
    contenido de datasets antes del procesamiento.
    
    Ejemplo:
        >>> validator = DatasetValidator()
        >>> validator.validate_dataframe(df, min_rows=100)
        >>> validator.validate_required_columns(df, ['Class', 'mfcc_1'])
    """
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        min_rows: int = 1,
        min_cols: int = 1,
        check_nulls: bool = True
    ) -> bool:
        """
        Valida estructura básica del DataFrame.
        
        Args:
            df: DataFrame a validar
            min_rows: Número mínimo de filas
            min_cols: Número mínimo de columnas
            check_nulls: Si True, advertir sobre valores nulos
        
        Returns:
            True si válido
        
        Raises:
            ValueError: Si la validación falla
        """
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Input debe ser un pandas DataFrame")
        
        if df.empty:
            raise ValueError("DataFrame está vacío")
        
        if df.shape[0] < min_rows:
            raise ValueError(
                f"DataFrame tiene {df.shape[0]} filas, "
                f"mínimo requerido: {min_rows}"
            )
        
        if df.shape[1] < min_cols:
            raise ValueError(
                f"DataFrame tiene {df.shape[1]} columnas, "
                f"mínimo requerido: {min_cols}"
            )
        
        if check_nulls:
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                logger.warning(
                    f"DataFrame contiene {null_count} valores nulos "
                    f"({null_count/df.size*100:.2f}%)"
                )
        
        logger.info(f"DataFrame validado: {df.shape[0]} filas × {df.shape[1]} columnas")
        return True
    
    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame,
        required_cols: List[str]
    ) -> bool:
        """
        Valida que existan columnas requeridas.
        
        Args:
            df: DataFrame a validar
            required_cols: Lista de columnas requeridas
        
        Returns:
            True si todas las columnas existen
        
        Raises:
            ValueError: Si faltan columnas
        """
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(
                f"Columnas requeridas faltantes: {missing_cols}\n"
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        logger.info(f"Todas las columnas requeridas presentes: {required_cols}")
        return True
    
    @staticmethod
    def validate_target_variable(
        y: pd.Series,
        expected_classes: Optional[List[str]] = None
    ) -> bool:
        """
        Valida variable target.
        
        Args:
            y: Serie con target variable
            expected_classes: Clases esperadas (opcional)
        
        Returns:
            True si válido
        
        Raises:
            ValueError: Si la validación falla
        """
        if y is None or not isinstance(y, (pd.Series, pd.DataFrame)):
            raise ValueError("Target debe ser pandas Series o DataFrame")
        
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"Target DataFrame debe tener 1 columna, tiene {y.shape[1]}")
            y = y.iloc[:, 0]
        
        unique_classes = y.unique()
        logger.info(f"Target tiene {len(unique_classes)} clases únicas: {unique_classes}")
        
        if expected_classes:
            missing = set(expected_classes) - set(unique_classes)
            extra = set(unique_classes) - set(expected_classes)
            
            if missing:
                logger.warning(f"Clases esperadas faltantes: {missing}")
            if extra:
                logger.warning(f"Clases inesperadas encontradas: {extra}")
        
        return True
    
    @staticmethod
    def validate_train_test_split(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> bool:
        """
        Valida conjuntos de train/test.
        
        Args:
            X_train, X_test: Features de train y test
            y_train, y_test: Targets de train y test
        
        Returns:
            True si válido
        
        Raises:
            ValueError: Si la validación falla
        """
        # Validar shapes
        if X_train.shape[0] != len(y_train):
            raise ValueError(
                f"X_train ({X_train.shape[0]}) y y_train ({len(y_train)}) "
                "tienen diferente número de filas"
            )
        
        if X_test.shape[0] != len(y_test):
            raise ValueError(
                f"X_test ({X_test.shape[0]}) y y_test ({len(y_test)}) "
                "tienen diferente número de filas"
            )
        
        # Validar columnas
        if not X_train.columns.equals(X_test.columns):
            raise ValueError("X_train y X_test tienen columnas diferentes")
        
        # Validar distribución de clases
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        logger.info("Distribución de clases en train:")
        for cls, prop in train_dist.items():
            logger.info(f"  {cls}: {prop*100:.2f}%")
        
        logger.info("Distribución de clases en test:")
        for cls, prop in test_dist.items():
            logger.info(f"  {cls}: {prop*100:.2f}%")
        
        logger.info("Train/test split validado correctamente")
        return True


class DatasetStatistics:
    """
    Generador de estadísticas para datasets.
    
    Proporciona análisis estadístico comprehensivo de datasets
    incluyendo distribuciones, correlaciones, y detección de anomalías.
    
    Ejemplo:
        >>> stats = DatasetStatistics()
        >>> summary = stats.get_summary(df)
        >>> correlations = stats.get_correlation_matrix(df)
    """
    
    @staticmethod
    def get_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera resumen estadístico completo.
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            Diccionario con estadísticas
        """
        summary = {
            "shape": df.shape,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "null_count": df.isnull().sum().sum(),
            "null_percentage": (df.isnull().sum().sum() / df.size) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=[np.number]).columns.tolist(),
            "column_types": df.dtypes.to_dict()
        }
        
        return summary
    
    @staticmethod
    def get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Obtiene estadísticas de columnas numéricas.
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            DataFrame con estadísticas descriptivas
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No hay columnas numéricas para analizar")
            return pd.DataFrame()
        
        stats = numeric_df.describe().T
        stats['null_count'] = numeric_df.isnull().sum()
        stats['null_pct'] = (numeric_df.isnull().sum() / len(df)) * 100
        
        return stats
    
    @staticmethod
    def get_correlation_matrix(
        df: pd.DataFrame,
        method: str = 'pearson',
        threshold: float = 0.8
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Calcula matriz de correlación y encuentra pares altamente correlacionados.
        
        Args:
            df: DataFrame a analizar
            method: Método de correlación ('pearson', 'spearman', 'kendall')
            threshold: Umbral para correlación alta
        
        Returns:
            Tupla (matriz_correlacion, lista_pares_correlacionados)
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            logger.warning("Necesita al menos 2 columnas numéricas para correlación")
            return pd.DataFrame(), []
        
        corr_matrix = numeric_df.corr(method=method)
        
        # Encontrar pares altamente correlacionados
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            logger.info(
                f"Encontrados {len(high_corr_pairs)} pares con "
                f"|correlación| >= {threshold}"
            )
        
        return corr_matrix, high_corr_pairs
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, pd.Series]:
        """
        Detecta outliers en columnas numéricas.
        
        Args:
            df: DataFrame a analizar
            method: Método de detección ('iqr' o 'zscore')
            threshold: Umbral (1.5 para IQR, 3 para Z-score)
        
        Returns:
            Diccionario {columna: mask_outliers}
        """
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_df.columns:
            if method == 'iqr':
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (
                    (numeric_df[col] < Q1 - threshold * IQR) |
                    (numeric_df[col] > Q3 + threshold * IQR)
                )
            elif method == 'zscore':
                z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
                outlier_mask = z_scores > threshold
            else:
                raise ValueError(f"Método inválido: {method}. Usar 'iqr' o 'zscore'")
            
            outliers[col] = outlier_mask
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                logger.info(
                    f"Columna '{col}': {n_outliers} outliers detectados "
                    f"({n_outliers/len(df)*100:.2f}%)"
                )
        
        return outliers


class DatasetManager(metaclass=SingletonMeta):
    """
    Gestor principal de datasets con patrón Singleton thread-safe.
    
    Proporciona interfaz unificada para:
    - Carga/guardado de datos en múltiples formatos
    - Validación automática
    - Análisis estadístico
    - Gestión de splits train/test
    - Context managers para operaciones seguras
    
    Attributes:
        config: Configuración de datasets
        validator: Validador de datos
        statistics: Generador de estadísticas
    
    Ejemplo:
        >>> manager = DatasetManager()
        >>> df = manager.load_processed(version='v2')
        >>> manager.validate_dataset(df)
        >>> stats = manager.get_statistics(df)
    """
    
    def __init__(self, config: type = DatasetConfig):
        """
        Inicializa el gestor de datasets.
        
        Args:
            config: Clase de configuración (default: DatasetConfig)
        """
        self.config = config
        self.validator = DatasetValidator()
        self.statistics = DatasetStatistics()
        
        # Validar directorios al inicializar
        try:
            self.config.validate_directories()
        except FileNotFoundError as e:
            logger.warning(f"Advertencia en inicialización: {e}")
        
        logger.info("DatasetManager inicializado (Singleton)")
    
    def _load_csv(
        self,
        filepath: Path,
        lazy: bool = False,
        validate: bool = True,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Carga archivo CSV con validación opcional.
        
        Args:
            filepath: Ruta del archivo
            lazy: Si True, solo valida existencia sin cargar
            validate: Si True, valida DataFrame después de cargar
            **kwargs: Argumentos adicionales para pd.read_csv
        
        Returns:
            DataFrame cargado o None si lazy=True
        
        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        
        if lazy:
            logger.info(f"CSV pendiente de carga (lazy): {filepath.name}")
            return None
        
        logger.info(f"Cargando: {filepath.name}")
        df = pd.read_csv(filepath, **kwargs)
        
        logger.info(
            f"Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas "
            f"({df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB)"
        )
        
        if validate:
            try:
                self.validator.validate_dataframe(df)
            except ValueError as e:
                logger.warning(f"Validación falló: {e}")
        
        return df
    
    def load_original(
        self,
        lazy: bool = False,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Carga dataset original turco.
        
        Args:
            lazy: Modo lazy loading
            validate: Validar después de cargar
        
        Returns:
            DataFrame con datos originales
        """
        return self._load_csv(
            self.config.RAW_DIR / self.config.TURKISH_ORIGINAL,
            lazy=lazy,
            validate=validate
        )
    
    def load_modified(
        self,
        lazy: bool = False,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Carga dataset modificado turco.
        
        Args:
            lazy: Modo lazy loading
            validate: Validar después de cargar
        
        Returns:
            DataFrame con datos modificados
        """
        return self._load_csv(
            self.config.RAW_DIR / self.config.TURKISH_MODIFIED,
            lazy=lazy,
            validate=validate
        )
    
    def load_cleaned(
        self,
        lazy: bool = False,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Carga dataset limpio.
        
        Args:
            lazy: Modo lazy loading
            validate: Validar después de cargar
        
        Returns:
            DataFrame con datos limpios
        """
        return self._load_csv(
            self.config.PROCESSED_DIR / self.config.CLEANED_FILENAME,
            lazy=lazy,
            validate=validate
        )
    
    def load_processed(
        self,
        version: str = "v2",
        lazy: bool = False,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Carga dataset procesado por versión.
        
        Args:
            version: Versión del dataset (ej. 'v1', 'v2', 'v3')
            lazy: Modo lazy loading
            validate: Validar después de cargar
        
        Returns:
            DataFrame con datos procesados
        """
        filename = self.config.PROCESSED_PATTERN.format(version=version)
        return self._load_csv(
            self.config.PROCESSED_DIR / filename,
            lazy=lazy,
            validate=validate
        )
    
    def save(
        self,
        df: pd.DataFrame,
        filename: str,
        validate: bool = True,
        backup: bool = False
    ) -> Path:
        """
        Guarda DataFrame en formato CSV.
        
        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo
            validate: Validar antes de guardar
            backup: Si True, crear backup si el archivo existe
        
        Returns:
            Path del archivo guardado
        """
        if validate:
            self.validator.validate_dataframe(df)
        
        path = self.config.PROCESSED_DIR / filename
        
        # Crear backup si existe y se solicita
        if backup and path.exists():
            backup_path = path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            path.rename(backup_path)
            logger.info(f"Backup creado: {backup_path.name}")
        
        df.to_csv(path, index=False)
        logger.info(f"Datos guardados en {path}")
        
        return path
    
    def load_train_test_split(
        self,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Carga splits de train/test.
        
        Args:
            validate: Validar splits después de cargar
        
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        
        Raises:
            FileNotFoundError: Si faltan archivos
        """
        paths = {
            "X_train": self.config.PROCESSED_DIR / "X_train.csv",
            "X_test": self.config.PROCESSED_DIR / "X_test.csv",
            "y_train": self.config.PROCESSED_DIR / "y_train.csv",
            "y_test": self.config.PROCESSED_DIR / "y_test.csv",
        }
        
        missing = [p.name for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Archivos faltantes: {missing}\n"
                f"Verificar que los splits estén en {self.config.PROCESSED_DIR}"
            )
        
        X_train = pd.read_csv(paths["X_train"])
        X_test = pd.read_csv(paths["X_test"])
        y_train = pd.read_csv(paths["y_train"])['Class']
        y_test = pd.read_csv(paths["y_test"])['Class']
        
        logger.info("Conjuntos train/test cargados correctamente")
        logger.info(f"  Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        logger.info(f"  Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
        
        if validate:
            self.validator.validate_train_test_split(X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test
    
    def save_train_test_split(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        validate: bool = True
    ) -> Dict[str, Path]:
        """
        Guarda splits de train/test.
        
        Args:
            X_train, X_test: Features
            y_train, y_test: Targets
            validate: Validar antes de guardar
        
        Returns:
            Diccionario con paths guardados
        """
        if validate:
            self.validator.validate_train_test_split(X_train, X_test, y_train, y_test)
        
        paths = {}
        paths['X_train'] = self.save(X_train, "X_train.csv", validate=False)
        paths['X_test'] = self.save(X_test, "X_test.csv", validate=False)
        
        # Guardar y como DataFrame con columna 'Class'
        y_train_df = pd.DataFrame({'Class': y_train})
        y_test_df = pd.DataFrame({'Class': y_test})
        
        paths['y_train'] = self.save(y_train_df, "y_train.csv", validate=False)
        paths['y_test'] = self.save(y_test_df, "y_test.csv", validate=False)
        
        logger.info("Train/test splits guardados exitosamente")
        return paths
    
    @staticmethod
    def dataset_info(df: pd.DataFrame, detailed: bool = False) -> None:
        """
        Muestra información del dataset.
        
        Args:
            df: DataFrame a analizar
            detailed: Si True, muestra información detallada
        """
        logger.info("="*60)
        logger.info("INFORMACIÓN DEL DATASET")
        logger.info("="*60)
        logger.info(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        logger.info(f"Memoria: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        logger.info(f"Nulos totales: {df.isnull().sum().sum():,}")
        
        if detailed:
            logger.info("\nColumnas:")
            for col in df.columns:
                dtype = df[col].dtype
                nulls = df[col].isnull().sum()
                unique = df[col].nunique()
                logger.info(
                    f"   • {col:30s} | {str(dtype):10s} | "
                    f"Nulls: {nulls:,} | Unique: {unique:,}"
                )
        else:
            logger.info(f"\nColumnas ({len(df.columns)}):")
            for col in df.columns:
                dtype = df[col].dtype
                nulls = df[col].isnull().sum()
                logger.info(f"   • {col:30s} | {str(dtype):10s} | Nulls: {nulls:,}")
        
        logger.info("="*60)
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
        min_rows: int = 1
    ) -> bool:
        """
        Valida dataset comprehensivamente.
        
        Args:
            df: DataFrame a validar
            required_cols: Columnas requeridas
            min_rows: Número mínimo de filas
        
        Returns:
            True si pasa todas las validaciones
        """
        self.validator.validate_dataframe(df, min_rows=min_rows)
        
        if required_cols:
            self.validator.validate_required_columns(df, required_cols)
        
        return True
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene estadísticas del dataset.
        
        Args:
            df: DataFrame a analizar
        
        Returns:
            Diccionario con estadísticas
        """
        return self.statistics.get_summary(df)
    
    @contextmanager
    def load_context(self, filepath: Path):
        """
        Context manager para carga segura de datos.
        
        Args:
            filepath: Path del archivo a cargar
        
        Yields:
            DataFrame cargado
        
        Ejemplo:
            >>> with manager.load_context(path) as df:
            ...     # Procesar df
            ...     pass
        """
        df = None
        try:
            df = self._load_csv(filepath)
            yield df
        except Exception as e:
            logger.error(f"Error en load_context: {e}")
            raise
        finally:
            if df is not None:
                del df
                logger.debug("DataFrame liberado de memoria")
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """
        Lista todos los datasets disponibles.
        
        Returns:
            Diccionario con archivos disponibles
        """
        files = self.config.get_all_available_files()
        
        available = {
            "raw": [f.name for f in files['raw']],
            "processed": [f.name for f in files['processed']],
        }
        
        logger.info(f"Datasets disponibles:")
        logger.info(f"  Raw: {len(available['raw'])} archivos")
        logger.info(f"  Processed: {len(available['processed'])} archivos")
        
        return available


# ============================================================================
# FUNCIONES LEGACY - Compatibilidad backward
# ============================================================================

def get_dataset_manager() -> DatasetManager:
    """
    Obtiene instancia singleton de DatasetManager.
    
    .. deprecated:: 2024.10
        Usar directamente DatasetManager() en su lugar.
    
    Returns:
        Instancia de DatasetManager
    """
    warnings.warn(
        "get_dataset_manager() está deprecated. "
        "Usar DatasetManager() directamente en su lugar.",
        DeprecationWarning,
        stacklevel=2
    )
    return DatasetManager()
