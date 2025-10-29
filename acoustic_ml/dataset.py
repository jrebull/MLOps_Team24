"""
Dataset Module - Professional MLOps Implementation
===================================================

Este módulo proporciona una infraestructura robusta para la gestión de datasets
en proyectos de Machine Learning, siguiendo las mejores prácticas de MLOps.

Componentes principales:
-----------------------
1. DatasetConfig: Configuración centralizada con validación automática
2. SingletonMeta: Metaclase thread-safe para el patrón Singleton
3. DatasetValidator: Validación comprehensiva de datos
4. DatasetStatistics: Análisis estadístico y detección de anomalías
5. DatasetManager: Gestor principal con operaciones seguras

Características:
---------------
- Thread-safe Singleton pattern
- Validación automática en todas las operaciones
- Context managers para operaciones seguras
- Logging comprehensivo
- Manejo robusto de errores
- Type hints completos
- Documentación en español

Uso:
----
>>> from acoustic_ml.dataset import DatasetManager
>>> manager = DatasetManager()
>>> df = manager.load_processed()
>>> manager.dataset_info(df, detailed=True)

Autor: MLOps Team24
Fecha: 2024
Versión: 2.0.0 (Refactorizado)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
from contextlib import contextmanager
import threading
from typing import Tuple, Optional, List, Dict, Any, Union
import warnings

# Importar configuración centralizada
from acoustic_ml.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    TURKISH_ORIGINAL,
    TURKISH_MODIFIED,
    CLEANED_FILENAME
)

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CLASE 1: DATASETCONFIG - Configuración Centralizada
# ============================================================================

class DatasetConfig:
    """
    Configuración centralizada para el manejo de datasets.
    
    Esta clase proporciona acceso centralizado a rutas, nombres de archivos
    y parámetros de configuración del proyecto, con validación automática.
    
    Atributos:
    ---------
    RAW_DIR : Path
        Directorio de datos raw
    INTERIM_DIR : Path
        Directorio de datos intermedios
    PROCESSED_DIR : Path
        Directorio de datos procesados
    TURKISH_ORIGINAL : str
        Nombre del archivo original turco
    TURKISH_MODIFIED : str
        Nombre del archivo modificado turco
    CLEANED_FILENAME : str
        Nombre del archivo limpio procesado
    
    Métodos:
    -------
    validate_directories()
        Valida que todos los directorios existan
    get_all_available_files()
        Lista todos los archivos disponibles
    get_config_summary()
        Retorna resumen de la configuración
    """
    
    # Directorios del proyecto
    RAW_DIR = RAW_DATA_DIR
    INTERIM_DIR = INTERIM_DATA_DIR
    PROCESSED_DIR = PROCESSED_DATA_DIR
    
    # Nombres de archivos estándar
    TURKISH_ORIGINAL = TURKISH_ORIGINAL
    TURKISH_MODIFIED = TURKISH_MODIFIED
    CLEANED_FILENAME = CLEANED_FILENAME
    
    # Parámetros por defecto
    DEFAULT_TARGET = "Class"
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_RANDOM_STATE = 42
    EXPECTED_CLASSES = ["Happy", "Sad", "Angry", "Relax"]
    
    @classmethod
    def validate_directories(cls) -> bool:
        """
        Valida que todos los directorios de configuración existan.
        
        Returns:
        -------
        bool
            True si todos los directorios existen
            
        Raises:
        ------
        FileNotFoundError
            Si algún directorio no existe
        """
        dirs_to_check = [cls.RAW_DIR, cls.INTERIM_DIR, cls.PROCESSED_DIR]
        
        for directory in dirs_to_check:
            if not directory.exists():
                raise FileNotFoundError(
                    f"Directorio no encontrado: {directory}\n"
                    f"Asegúrate de que la estructura del proyecto esté completa."
                )
        
        logger.info("✅ Validación de directorios exitosa")
        return True
    
    @classmethod
    def get_all_available_files(cls) -> Dict[str, List[str]]:
        """
        Lista todos los archivos CSV disponibles en raw y processed.
        
        Returns:
        -------
        dict
            Diccionario con listas de archivos por tipo
        """
        raw_files = list(cls.RAW_DIR.glob("*.csv"))
        processed_files = list(cls.PROCESSED_DIR.glob("*.csv"))
        
        return {
            'raw': [f.name for f in raw_files],
            'processed': [f.name for f in processed_files],
            'total': len(raw_files) + len(processed_files)
        }
    
    @classmethod
    def get_config_summary(cls) -> str:
        """
        Genera un resumen legible de la configuración.
        
        Returns:
        -------
        str
            Resumen formateado de la configuración
        """
        files = cls.get_all_available_files()
        
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║              DATASET CONFIGURATION SUMMARY                      ║
╠════════════════════════════════════════════════════════════════╣
║ Directorios:                                                   ║
║   • RAW:       {str(cls.RAW_DIR):<46}║
║   • INTERIM:   {str(cls.INTERIM_DIR):<46}║
║   • PROCESSED: {str(cls.PROCESSED_DIR):<46}║
╠════════════════════════════════════════════════════════════════╣
║ Archivos Disponibles:                                          ║
║   • Raw:       {len(files['raw']):>2} archivos                                      ║
║   • Processed: {len(files['processed']):>2} archivos                                      ║
╠════════════════════════════════════════════════════════════════╣
║ Configuración ML:                                              ║
║   • Target:       {cls.DEFAULT_TARGET:<45}║
║   • Test Size:    {cls.DEFAULT_TEST_SIZE:<45}║
║   • Random State: {cls.DEFAULT_RANDOM_STATE:<45}║
║   • Classes:      {len(cls.EXPECTED_CLASSES)} emociones                                    ║
╚════════════════════════════════════════════════════════════════╝
"""
        return summary


# ============================================================================
# CLASE 2: SINGLETONMETA - Metaclase Thread-Safe
# ============================================================================

class SingletonMeta(type):
    """
    Metaclase thread-safe para implementar el patrón Singleton.
    
    Esta implementación garantiza que solo exista una instancia de la clase
    que la utilice, incluso en entornos multi-threaded.
    
    Características:
    ---------------
    - Thread-safe mediante Lock
    - Double-checked locking pattern
    - Lazy initialization
    - Método clear_instances() para testing
    
    Uso:
    ----
    >>> class MiClase(metaclass=SingletonMeta):
    ...     pass
    >>> obj1 = MiClase()
    >>> obj2 = MiClase()
    >>> assert obj1 is obj2  # True
    """
    
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """
        Controla la creación de instancias garantizando Singleton.
        
        Usa double-checked locking para eficiencia en multi-threading.
        """
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        
        return cls._instances[cls]
    
    @classmethod
    def clear_instances(cls) -> None:
        """
        Limpia todas las instancias singleton.
        
        Útil principalmente para testing, donde necesitas
        resetear el estado del singleton entre tests.
        
        Warning:
        -------
        Solo usar en entornos de testing. No usar en producción.
        """
        with cls._lock:
            cls._instances.clear()
            logger.debug("Instancias Singleton limpiadas")


# ============================================================================
# CLASE 3: DATASETVALIDATOR - Validación Comprehensiva
# ============================================================================

class DatasetValidator:
    """
    Validador comprehensivo para DataFrames y operaciones de ML.
    
    Proporciona métodos de validación robustos para asegurar la calidad
    e integridad de los datos en todas las etapas del pipeline.
    
    Métodos:
    -------
    validate_dataframe(df, min_rows, min_cols)
        Valida estructura básica del DataFrame
    validate_required_columns(df, required_cols)
        Valida presencia de columnas requeridas
    validate_target_variable(y, expected_classes)
        Valida variable objetivo
    validate_train_test_split(X_train, X_test, y_train, y_test)
        Valida splits de entrenamiento/prueba
    """
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        min_rows: int = 1,
        min_cols: int = 1,
        check_nulls: bool = False
    ) -> bool:
        """
        Valida la estructura básica de un DataFrame.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a validar
        min_rows : int, default=1
            Número mínimo de filas requeridas
        min_cols : int, default=1
            Número mínimo de columnas requeridas
        check_nulls : bool, default=False
            Si True, verifica que no haya valores nulos
            
        Returns:
        -------
        bool
            True si todas las validaciones pasan
            
        Raises:
        ------
        ValueError
            Si alguna validación falla
        """
        # Check 1: Not None
        if df is None:
            raise ValueError("DataFrame no puede ser None")
        
        # Check 2: Es DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Se esperaba pd.DataFrame, se recibió {type(df)}")
        
        # Check 3: No está vacío
        if df.empty:
            raise ValueError("DataFrame está vacío")
        
        # Check 4: Mínimo de filas
        if len(df) < min_rows:
            raise ValueError(
                f"DataFrame tiene {len(df)} filas, se requieren al menos {min_rows}"
            )
        
        # Check 5: Mínimo de columnas
        if len(df.columns) < min_cols:
            raise ValueError(
                f"DataFrame tiene {len(df.columns)} columnas, "
                f"se requieren al menos {min_cols}"
            )
        
        # Check 6: Valores nulos (opcional)
        if check_nulls:
            null_count = df.isnull().sum().sum()
            if null_count > 0:
                raise ValueError(
                    f"DataFrame contiene {null_count} valores nulos"
                )
        
        logger.debug(f"✅ DataFrame válido: {df.shape}")
        return True
    
    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame,
        required_cols: List[str]
    ) -> bool:
        """
        Valida que todas las columnas requeridas estén presentes.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a validar
        required_cols : List[str]
            Lista de nombres de columnas requeridas
            
        Returns:
        -------
        bool
            True si todas las columnas están presentes
            
        Raises:
        ------
        ValueError
            Si faltan columnas requeridas
        """
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise ValueError(
                f"Columnas faltantes: {missing_cols}\n"
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        logger.debug(f"✅ Todas las columnas requeridas presentes: {required_cols}")
        return True
    
    @staticmethod
    def validate_target_variable(
        y: Union[pd.Series, np.ndarray],
        expected_classes: Optional[List[str]] = None,
        min_samples_per_class: int = 1
    ) -> bool:
        """
        Valida la variable objetivo (target).
        
        Parameters:
        ----------
        y : pd.Series or np.ndarray
            Variable objetivo a validar
        expected_classes : List[str], optional
            Clases esperadas en la variable objetivo
        min_samples_per_class : int, default=1
            Número mínimo de muestras por clase
            
        Returns:
        -------
        bool
            True si todas las validaciones pasan
            
        Raises:
        ------
        ValueError
            Si alguna validación falla
        """
        # Check 1: No es None o vacío
        if y is None or len(y) == 0:
            raise ValueError("Variable objetivo está vacía o es None")
        
        # Convertir a Series si es necesario
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Check 2: Verificar clases esperadas
        if expected_classes is not None:
            actual_classes = set(y.unique())
            expected_set = set(expected_classes)
            
            missing = expected_set - actual_classes
            extra = actual_classes - expected_set
            
            if missing:
                logger.warning(f"⚠️  Clases faltantes: {missing}")
            
            if extra:
                raise ValueError(f"Clases inesperadas: {extra}")
        
        # Check 3: Mínimo de muestras por clase
        class_counts = y.value_counts()
        insufficient = class_counts[class_counts < min_samples_per_class]
        
        if not insufficient.empty:
            raise ValueError(
                f"Clases con muestras insuficientes (<{min_samples_per_class}):\n"
                f"{insufficient.to_dict()}"
            )
        
        logger.debug(f"✅ Variable objetivo válida. Distribución:\n{class_counts}")
        return True
    
    @staticmethod
    def validate_train_test_split(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray]
    ) -> bool:
        """
        Valida la integridad de un train-test split.
        
        Parameters:
        ----------
        X_train, X_test : pd.DataFrame
            Features de entrenamiento y prueba
        y_train, y_test : pd.Series or np.ndarray
            Targets de entrenamiento y prueba
            
        Returns:
        -------
        bool
            True si todas las validaciones pasan
            
        Raises:
        ------
        ValueError
            Si alguna validación falla
        """
        # Check 1: Tamaños consistentes
        if len(X_train) != len(y_train):
            raise ValueError(
                f"Inconsistencia en train: X_train={len(X_train)}, y_train={len(y_train)}"
            )
        
        if len(X_test) != len(y_test):
            raise ValueError(
                f"Inconsistencia en test: X_test={len(X_test)}, y_test={len(y_test)}"
            )
        
        # Check 2: Features consistentes
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"Número de features inconsistente: "
                f"train={X_train.shape[1]}, test={X_test.shape[1]}"
            )
        
        # Check 3: Columnas idénticas
        if not all(X_train.columns == X_test.columns):
            raise ValueError("Columnas de features no coinciden entre train y test")
        
        # Check 4: No hay overlap de índices
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        overlap = train_idx & test_idx
        
        if overlap:
            logger.warning(f"⚠️  Índices compartidos entre train/test: {len(overlap)}")
        
        logger.info(
            f"✅ Split válido - Train: {X_train.shape}, Test: {X_test.shape}"
        )
        return True


# ============================================================================
# CLASE 4: DATASETSTATISTICS - Análisis Estadístico
# ============================================================================

class DatasetStatistics:
    """
    Análisis estadístico comprehensivo de datasets.
    
    Proporciona métodos para calcular estadísticas descriptivas,
    detectar anomalías, y analizar la calidad de los datos.
    
    Métodos:
    -------
    get_summary(df)
        Resumen general del dataset
    get_numeric_stats(df)
        Estadísticas de variables numéricas
    get_correlation_matrix(df, threshold)
        Matriz de correlación con pares altamente correlacionados
    detect_outliers(df, method, threshold)
        Detección de outliers usando diferentes métodos
    """
    
    @staticmethod
    def get_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un resumen comprehensivo del dataset.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a analizar
            
        Returns:
        -------
        dict
            Diccionario con estadísticas del dataset
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'shape': df.shape,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'n_numeric': len(numeric_cols),
            'n_categorical': len(categorical_cols),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'null_counts': df.isnull().sum().to_dict(),
            'total_nulls': df.isnull().sum().sum(),
            'null_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'dtypes': df.dtypes.value_counts().to_dict()
        }
        
        return summary
    
    @staticmethod
    def get_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadísticas descriptivas para columnas numéricas.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a analizar
            
        Returns:
        -------
        pd.DataFrame
            DataFrame con estadísticas descriptivas
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("⚠️  No hay columnas numéricas para analizar")
            return pd.DataFrame()
        
        stats = numeric_df.describe().T
        
        # Agregar estadísticas adicionales
        stats['missing'] = numeric_df.isnull().sum()
        stats['missing_pct'] = (numeric_df.isnull().sum() / len(df)) * 100
        stats['zeros'] = (numeric_df == 0).sum()
        stats['zeros_pct'] = ((numeric_df == 0).sum() / len(df)) * 100
        stats['skewness'] = numeric_df.skew()
        stats['kurtosis'] = numeric_df.kurtosis()
        
        return stats
    
    @staticmethod
    def get_correlation_matrix(
        df: pd.DataFrame,
        threshold: float = 0.8
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Calcula matriz de correlación e identifica pares altamente correlacionados.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a analizar
        threshold : float, default=0.8
            Umbral para considerar correlación alta
            
        Returns:
        -------
        tuple
            (matriz_correlacion, lista_pares_alta_correlacion)
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("⚠️  No hay columnas numéricas para correlación")
            return pd.DataFrame(), []
        
        corr_matrix = numeric_df.corr()
        
        # Encontrar pares con alta correlación
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append((col1, col2, corr_val))
        
        if high_corr_pairs:
            logger.info(
                f"🔍 Encontrados {len(high_corr_pairs)} pares con "
                f"correlación ≥ {threshold}"
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
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a analizar
        method : str, default='iqr'
            Método de detección: 'iqr' o 'zscore'
        threshold : float, default=1.5
            Umbral para el método seleccionado
            - IQR: multiplicador del rango intercuartílico (default 1.5)
            - Z-score: número de desviaciones estándar (default 3.0)
            
        Returns:
        -------
        dict
            Diccionario con máscaras booleanas de outliers por columna
        """
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        
        if method == 'iqr':
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers[col] = (
                    (numeric_df[col] < lower_bound) |
                    (numeric_df[col] > upper_bound)
                )
        
        elif method == 'zscore':
            for col in numeric_df.columns:
                z_scores = np.abs(
                    (numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std()
                )
                outliers[col] = z_scores > threshold
        
        else:
            raise ValueError(f"Método no soportado: {method}. Use 'iqr' o 'zscore'")
        
        total_outliers = sum(mask.sum() for mask in outliers.values())
        logger.info(f"🔍 Detectados {total_outliers} outliers usando método '{method}'")
        
        return outliers


# ============================================================================
# CLASE 5: DATASETMANAGER - Gestor Principal (Singleton)
# ============================================================================

class DatasetManager(metaclass=SingletonMeta):
    """
    Gestor principal para todas las operaciones con datasets.
    
    Implementa el patrón Singleton para garantizar una única instancia
    de gestión de datos en todo el proyecto.
    
    Características:
    ---------------
    - Singleton thread-safe
    - Validación automática en operaciones
    - Context managers para operaciones seguras
    - Logging comprehensivo
    - Integración con DVC/MLflow
    
    Atributos:
    ---------
    config : DatasetConfig
        Configuración del dataset
    validator : DatasetValidator
        Validador de datos
    stats : DatasetStatistics
        Analizador estadístico
    
    Métodos principales:
    -------------------
    load_original()
        Carga datos raw originales
    load_processed(filename)
        Carga datos procesados
    save(df, filename, validate)
        Guarda DataFrame con validación
    get_train_test_split(...)
        Genera split entrenamiento/prueba
    load_train_test_split(validate)
        Carga split guardado previamente
    dataset_info(df, detailed)
        Muestra información del dataset
    validate_dataset(df, ...)
        Valida dataset con múltiples checks
    get_statistics(df)
        Obtiene estadísticas del dataset
    """
    
    def __init__(self, config: Optional[type] = None):
        """
        Inicializa el DatasetManager.
        
        Parameters:
        ----------
        config : type, optional
            Clase de configuración a usar (default: DatasetConfig)
        """
        # Solo inicializar una vez (Singleton)
        if hasattr(self, '_initialized'):
            return
        
        self.config = config or DatasetConfig
        self.validator = DatasetValidator()
        self.stats = DatasetStatistics()
        
        # Validar directorios al inicializar
        try:
            self.config.validate_directories()
        except FileNotFoundError as e:
            logger.warning(f"⚠️  {e}")
        
        self._initialized = True
        logger.info("✅ DatasetManager inicializado (Singleton)")
    
    # ========================================================================
    # MÉTODOS DE CARGA
    # ========================================================================
    
    def _load_csv(
        self,
        path: Path,
        validate: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Carga un archivo CSV con validación opcional.
        
        Parameters:
        ----------
        path : Path
            Ruta al archivo CSV
        validate : bool, default=True
            Si True, valida el DataFrame después de cargar
        **kwargs
            Argumentos adicionales para pd.read_csv()
            
        Returns:
        -------
        pd.DataFrame
            DataFrame cargado
            
        Raises:
        ------
        FileNotFoundError
            Si el archivo no existe
        ValueError
            Si el archivo está vacío o la validación falla
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Archivo no encontrado: {path}\n"
                f"Directorio: {path.parent}\n"
                f"Archivos disponibles: {list(path.parent.glob('*.csv'))}"
            )
        
        try:
            df = pd.read_csv(path, **kwargs)
            
            if validate:
                self.validator.validate_dataframe(df)
            
            logger.info(f"✅ Cargado: {path.name} - Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error cargando {path.name}: {e}")
            raise
    
    def load_original(self) -> pd.DataFrame:
        """
        Carga el dataset original desde raw data.
        
        Returns:
        -------
        pd.DataFrame
            Dataset original
        """
        path = self.config.RAW_DIR / self.config.TURKISH_ORIGINAL
        return self._load_csv(path, validate=True)
    
    def load_processed(
        self,
        filename: str = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Carga un dataset procesado.
        
        Parameters:
        ----------
        filename : str, optional
            Nombre del archivo (default: CLEANED_FILENAME de config)
        validate : bool, default=True
            Si True, valida el DataFrame
            
        Returns:
        -------
        pd.DataFrame
            Dataset procesado
        """
        filename = filename or self.config.CLEANED_FILENAME
        path = self.config.PROCESSED_DIR / filename
        return self._load_csv(path, validate=validate)
    
    # ========================================================================
    # MÉTODOS DE GUARDADO
    # ========================================================================
    
    def save(
        self,
        df: pd.DataFrame,
        filename: str,
        validate: bool = True,
        **kwargs
    ) -> Path:
        """
        Guarda un DataFrame como CSV con validación opcional.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a guardar
        filename : str
            Nombre del archivo de salida
        validate : bool, default=True
            Si True, valida el DataFrame antes de guardar
        **kwargs
            Argumentos adicionales para df.to_csv()
            
        Returns:
        -------
        Path
            Ruta del archivo guardado
        """
        if validate:
            self.validator.validate_dataframe(df)
        
        path = self.config.PROCESSED_DIR / filename
        
        # Argumentos por defecto
        save_kwargs = {'index': False}
        save_kwargs.update(kwargs)
        
        df.to_csv(path, **save_kwargs)
        logger.info(f"💾 Guardado: {filename} - Shape: {df.shape}")
        
        return path
    
    # ========================================================================
    # TRAIN-TEST SPLIT
    # ========================================================================
    
    def get_train_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        target_column: str = None,
        test_size: float = None,
        random_state: int = None,
        stratify: bool = True,
        save_splits: bool = False,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Genera train-test split con validación y opción de guardado.
        
        Parameters:
        ----------
        df : pd.DataFrame, optional
            DataFrame a dividir (default: carga processed)
        target_column : str, optional
            Nombre de la columna objetivo (default: config.DEFAULT_TARGET)
        test_size : float, optional
            Proporción del test set (default: config.DEFAULT_TEST_SIZE)
        random_state : int, optional
            Semilla aleatoria (default: config.DEFAULT_RANDOM_STATE)
        stratify : bool, default=True
            Si True, hace split estratificado
        save_splits : bool, default=False
            Si True, guarda los splits en processed/
        validate : bool, default=True
            Si True, valida los splits resultantes
            
        Returns:
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Usar valores por defecto de config
        target_column = target_column or self.config.DEFAULT_TARGET
        test_size = test_size or self.config.DEFAULT_TEST_SIZE
        random_state = random_state or self.config.DEFAULT_RANDOM_STATE
        
        # Cargar datos si no se proporcionaron
        if df is None:
            df = self.load_processed()
        
        # Validar columna objetivo
        if target_column not in df.columns:
            raise ValueError(
                f"Columna objetivo '{target_column}' no encontrada.\n"
                f"Columnas disponibles: {list(df.columns)}"
            )
        
        # Separar features y target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Validar target
        if validate:
            self.validator.validate_target_variable(
                y,
                expected_classes=self.config.EXPECTED_CLASSES
            )
        
        # Realizar split
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Validar splits
        if validate:
            self.validator.validate_train_test_split(
                X_train, X_test, y_train, y_test
            )
        
        logger.info(
            f"✂️  Split creado - Train: {len(X_train)}, Test: {len(X_test)} "
            f"({test_size*100:.1f}% test)"
        )
        
        # Guardar splits si se solicita
        if save_splits:
            self._save_splits(X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test
    
    def _save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> None:
        """
        Guarda los splits de train/test en archivos CSV.
        
        Parameters:
        ----------
        X_train, X_test : pd.DataFrame
            Features de train y test
        y_train, y_test : pd.Series
            Targets de train y test
        """
        self.save(X_train, 'X_train.csv', validate=False)
        self.save(X_test, 'X_test.csv', validate=False)
        
        # Guardar y como Series con nombre
        y_train_df = y_train.to_frame()
        y_test_df = y_test.to_frame()
        
        self.save(y_train_df, 'y_train.csv', validate=False)
        self.save(y_test_df, 'y_test.csv', validate=False)
        
        logger.info("💾 Splits guardados en processed/")
    
    def load_train_test_split(
        self,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Carga splits guardados previamente.
        
        Parameters:
        ----------
        validate : bool, default=True
            Si True, valida la integridad de los splits
            
        Returns:
        -------
        tuple
            (X_train, X_test, y_train, y_test)
            
        Raises:
        ------
        FileNotFoundError
            Si algún archivo de split no existe
        ValueError
            Si la validación falla
        """
        split_files = {
            'X_train': self.config.PROCESSED_DIR / 'X_train.csv',
            'X_test': self.config.PROCESSED_DIR / 'X_test.csv',
            'y_train': self.config.PROCESSED_DIR / 'y_train.csv',
            'y_test': self.config.PROCESSED_DIR / 'y_test.csv'
        }
        
        # Verificar que todos los archivos existan
        missing = [name for name, path in split_files.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Archivos de split faltantes: {missing}\n"
                f"Ejecuta get_train_test_split(save_splits=True) primero."
            )
        
        # Cargar X splits con index_col=0
        X_train = pd.read_csv(split_files['X_train'], index_col=0)
        X_test = pd.read_csv(split_files['X_test'], index_col=0)
        
        # Cargar y splits - manejar diferentes formatos
        try:
            y_train = pd.read_csv(split_files['y_train'], index_col=0)
            if isinstance(y_train, pd.DataFrame) and len(y_train.columns) == 1:
                y_train = y_train.iloc[:, 0]
            elif isinstance(y_train, pd.DataFrame) and len(y_train.columns) == 0:
                y_train = pd.read_csv(
                    split_files['y_train'],
                    header=None,
                    index_col=0,
                    squeeze=True
                )
        except Exception:
            y_train = pd.read_csv(split_files['y_train']).squeeze()
        
        try:
            y_test = pd.read_csv(split_files['y_test'], index_col=0)
            if isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 1:
                y_test = y_test.iloc[:, 0]
            elif isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 0:
                y_test = pd.read_csv(
                    split_files['y_test'],
                    header=None,
                    index_col=0,
                    squeeze=True
                )
        except Exception:
            y_test = pd.read_csv(split_files['y_test']).squeeze()
        
        # Convertir a numpy arrays y luego a Series
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values.ravel()
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.ravel()
        
        y_train = pd.Series(y_train, name='Class')
        y_test = pd.Series(y_test, name='Class')
        
        logger.info(
            f"📂 Splits cargados - Train: {X_train.shape}, Test: {X_test.shape}"
        )
        
        # Validar splits
        if validate:
            self.validator.validate_train_test_split(
                X_train, X_test, y_train, y_test
            )
        
        return X_train, X_test, y_train, y_test
    
    # ========================================================================
    # MÉTODOS DE ANÁLISIS
    # ========================================================================
    
    def dataset_info(
        self,
        df: pd.DataFrame,
        detailed: bool = False
    ) -> None:
        """
        Muestra información comprehensiva del dataset.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Dataset a analizar
        detailed : bool, default=False
            Si True, muestra información detallada adicional
        """
        print("\n" + "="*70)
        print("📊 DATASET INFORMATION")
        print("="*70)
        
        # Información básica
        summary = self.stats.get_summary(df)
        
        print(f"\n📏 Dimensiones:")
        print(f"   • Shape: {summary['shape']}")
        print(f"   • Filas: {summary['n_rows']:,}")
        print(f"   • Columnas: {summary['n_cols']}")
        print(f"   • Memoria: {summary['memory_mb']:.2f} MB")
        
        print(f"\n🔢 Tipos de datos:")
        print(f"   • Numéricas: {summary['n_numeric']}")
        print(f"   • Categóricas: {summary['n_categorical']}")
        
        print(f"\n❓ Valores nulos:")
        print(f"   • Total: {summary['total_nulls']}")
        print(f"   • Porcentaje: {summary['null_percentage']:.2f}%")
        
        if summary['total_nulls'] > 0:
            print("\n   Por columna:")
            for col, count in summary['null_counts'].items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    print(f"      - {col}: {count} ({pct:.1f}%)")
        
        # Información detallada
        if detailed:
            print("\n" + "="*70)
            print("📈 ESTADÍSTICAS DETALLADAS")
            print("="*70)
            
            numeric_stats = self.stats.get_numeric_stats(df)
            if not numeric_stats.empty:
                print("\n🔢 Variables numéricas:")
                print(numeric_stats.to_string())
            
            # Correlaciones altas
            _, high_corr = self.stats.get_correlation_matrix(df, threshold=0.8)
            if high_corr:
                print("\n🔗 Correlaciones altas (≥0.8):")
                for col1, col2, corr in high_corr:
                    print(f"   • {col1} ↔ {col2}: {corr:.3f}")
        
        print("\n" + "="*70)
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
        min_rows: int = 1,
        check_nulls: bool = False
    ) -> bool:
        """
        Valida un dataset con múltiples checks.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Dataset a validar
        required_cols : List[str], optional
            Columnas requeridas
        min_rows : int, default=1
            Número mínimo de filas
        check_nulls : bool, default=False
            Si True, rechaza datasets con nulos
            
        Returns:
        -------
        bool
            True si todas las validaciones pasan
        """
        try:
            # Validación básica
            self.validator.validate_dataframe(
                df,
                min_rows=min_rows,
                check_nulls=check_nulls
            )
            
            # Validar columnas requeridas
            if required_cols:
                self.validator.validate_required_columns(df, required_cols)
            
            logger.info("✅ Dataset válido - Todas las validaciones pasaron")
            return True
            
        except ValueError as e:
            logger.error(f"❌ Validación fallida: {e}")
            raise
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Obtiene estadísticas comprehensivas del dataset.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Dataset a analizar
            
        Returns:
        -------
        dict
            Diccionario con todas las estadísticas
        """
        return self.stats.get_summary(df)
    
    # ========================================================================
    # CONTEXT MANAGERS
    # ========================================================================
    
    @contextmanager
    def temporary_dataset(self, df: pd.DataFrame, filename: str):
        """
        Context manager para trabajar con un dataset temporal.
        
        Guarda el DataFrame temporalmente y lo elimina al finalizar.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame a guardar temporalmente
        filename : str
            Nombre del archivo temporal
            
        Yields:
        ------
        Path
            Ruta del archivo temporal
        
        Example:
        -------
        >>> with manager.temporary_dataset(df, 'temp.csv') as temp_path:
        ...     # Trabajar con el archivo temporal
        ...     loaded = pd.read_csv(temp_path)
        ... # El archivo se elimina automáticamente
        """
        temp_path = self.save(df, filename)
        
        try:
            yield temp_path
        finally:
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"🗑️  Archivo temporal eliminado: {filename}")


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def print_dataset_config():
    """Imprime la configuración actual del dataset."""
    print(DatasetConfig.get_config_summary())


def quick_load(filename: str = None) -> pd.DataFrame:
    """
    Atajo para cargar rápidamente un dataset procesado.
    
    Parameters:
    ----------
    filename : str, optional
        Nombre del archivo (default: CLEANED_FILENAME)
        
    Returns:
    -------
    pd.DataFrame
        Dataset cargado
    """
    manager = DatasetManager()
    return manager.load_processed(filename)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'DatasetConfig',
    'SingletonMeta',
    'DatasetValidator',
    'DatasetStatistics',
    'DatasetManager',
    'print_dataset_config',
    'quick_load'
]


# ============================================================================
# TESTING CODE (solo si se ejecuta directamente)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING DATASET MODULE")
    print("="*70)
    
    # Test configuración
    print("\n1. Testing DatasetConfig...")
    print(DatasetConfig.get_config_summary())
    
    # Test manager
    print("\n2. Testing DatasetManager (Singleton)...")
    manager1 = DatasetManager()
    manager2 = DatasetManager()
    print(f"   Singleton OK: {manager1 is manager2}")
    
    # Test carga
    print("\n3. Testing load operations...")
    try:
        df = manager1.load_processed()
        print(f"   ✅ Dataset cargado: {df.shape}")
        
        # Test info
        print("\n4. Testing dataset_info...")
        manager1.dataset_info(df, detailed=False)
        
    except FileNotFoundError:
        print("   ⚠️  Archivo procesado no encontrado (esperado en testing)")
    
    print("\n" + "="*70)
    print("✅ MODULE TEST COMPLETED")
    print("="*70)
