"""
Módulo de ingeniería de características para clasificación de emociones en música.

Este módulo implementa transformadores compatibles con scikit-learn para
procesamiento de features acústicas en el proyecto de Turkish Music Emotion
Recognition.

Implementa principios SOLID:
- Single Responsibility: Cada transformer tiene una responsabilidad específica
- Open/Closed: Extensible mediante herencia, cerrado a modificación
- Liskov Substitution: Todos los transformers son intercambiables
- Interface Segregation: Interfaces mínimas y específicas
- Dependency Inversion: Depende de abstracciones sklearn

Arquitectura:
    FeatureTransformer: Clase base abstracta con validación
    │
    ├── NumericFeatureSelector: Selección de columnas numéricas
    ├── PowerFeatureTransformer: Transformación de potencia (normalización)
    ├── OutlierRemover: Eliminación de outliers con IQR
    ├── FeatureScaler: Escalado con múltiples métodos
    ├── CorrelationFilter: Filtrado por correlación
    └── VarianceThresholdSelector: Selección por varianza

    FeaturePipelineBuilder: Constructor de pipelines con patrón Builder
    
    Factory Functions: Funciones de conveniencia para crear pipelines

Ejemplo básico:
    >>> from acoustic_ml.features import create_preprocessing_pipeline
    >>> pipeline = create_preprocessing_pipeline()
    >>> X_transformed = pipeline.fit_transform(X_train)

Ejemplo avanzado:
    >>> from acoustic_ml.features import FeaturePipelineBuilder
    >>> pipeline = (FeaturePipelineBuilder()
    ...     .add_numeric_selector()
    ...     .add_outlier_remover()
    ...     .add_power_transformer()
    ...     .add_feature_scaler(method='standard')
    ...     .build())

Author: MLOps Team 24
Date: 2024-10
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Tuple
import logging
import warnings

# Configuración de logging
logger = logging.getLogger(__name__)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Clase base abstracta para transformadores de features.
    
    Proporciona funcionalidad común para todos los transformadores:
    - Validación de datos de entrada
    - Logging automático de operaciones
    - Preservación de nombres de features
    - Compatibilidad con pandas DataFrames y numpy arrays
    
    Todos los transformadores personalizados deben heredar de esta clase
    e implementar los métodos fit() y transform().
    
    Attributes:
        feature_names_in_: Nombres de features de entrada (si disponible)
        n_features_in_: Número de features de entrada
        is_fitted_: Flag indicando si el transformer está entrenado
    
    Ejemplo:
        >>> class MyTransformer(FeatureTransformer):
        ...     def fit(self, X, y=None):
        ...         self._validate_data(X)
        ...         # Tu lógica de fit aquí
        ...         self.is_fitted_ = True
        ...         return self
        ...     def transform(self, X):
        ...         self._check_is_fitted()
        ...         return X * 2  # Ejemplo simple
    """
    
    def __init__(self):
        """Inicializa el transformador base."""
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y=None):
        """
        Método base de entrenamiento.
        
        Subclases deben override este método con su lógica específica,
        pero deben llamar a super().fit(X, y) al inicio.
        
        Args:
            X: Datos de entrada (DataFrame o array)
            y: Target (opcional, para compatibilidad sklearn)
        
        Returns:
            self: Instancia entrenada
        """
        self._validate_data(X)
        self.is_fitted_ = True
        logger.debug(f"{self.__class__.__name__} entrenado con {self.n_features_in_} features")
        return self
    
    def transform(self, X):
        """
        Método base de transformación.
        
        Subclases DEBEN implementar este método con su lógica específica.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_transformed: Datos transformados
        
        Raises:
            NotImplementedError: Si la subclase no implementa este método
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} debe implementar el método transform()"
        )
    
    def _validate_data(self, X, reset: bool = True) -> None:
        """
        Valida los datos de entrada y almacena metadata.
        
        Args:
            X: Datos a validar
            reset: Si True, reinicia metadata de features
        
        Raises:
            ValueError: Si X es None o vacío
            TypeError: Si X no es DataFrame ni array
        """
        if X is None:
            raise ValueError("X no puede ser None")
        
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("DataFrame está vacío")
            if reset:
                self.feature_names_in_ = X.columns.tolist()
                self.n_features_in_ = X.shape[1]
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("Array está vacío")
            if reset:
                self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
                self.feature_names_in_ = None
        else:
            raise TypeError(
                f"X debe ser pandas DataFrame o numpy array, "
                f"recibido: {type(X)}"
            )
    
    def _check_is_fitted(self) -> None:
        """
        Verifica que el transformer esté entrenado.
        
        Raises:
            RuntimeError: Si el transformer no está entrenado
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} no está entrenado. "
                "Ejecutar fit() primero."
            )
    
    def _preserve_dataframe_format(
        self,
        X_original: Union[pd.DataFrame, np.ndarray],
        X_transformed: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Preserva el formato DataFrame si el input original era DataFrame.
        
        Args:
            X_original: Datos originales de entrada
            X_transformed: Datos transformados (array)
            feature_names: Nombres de features para el output (opcional)
        
        Returns:
            X_transformed en el mismo formato que X_original
        """
        if isinstance(X_original, pd.DataFrame):
            if feature_names is None:
                feature_names = X_original.columns.tolist()
            return pd.DataFrame(
                X_transformed,
                columns=feature_names,
                index=X_original.index
            )
        return X_transformed
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Obtiene los nombres de las features de salida.
        
        Args:
            input_features: Nombres de features de entrada (opcional)
        
        Returns:
            Lista de nombres de features de salida
        """
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        elif input_features is not None:
            return input_features
        else:
            return [f"feature_{i}" for i in range(self.n_features_in_)]


class NumericFeatureSelector(FeatureTransformer):
    """
    Selecciona solo columnas numéricas de un DataFrame.
    
    Útil para filtrar automáticamente columnas categóricas o no numéricas
    antes del procesamiento. Permite excluir columnas específicas como IDs.
    
    Attributes:
        exclude_cols: Lista de columnas a excluir
        feature_names_: Nombres de features seleccionadas
    
    Ejemplo:
        >>> selector = NumericFeatureSelector(exclude_cols=['id', 'filename'])
        >>> X_numeric = selector.fit_transform(X)
        >>> print(selector.feature_names_)
        ['mfcc_1', 'mfcc_2', 'spectral_centroid', ...]
    """
    
    def __init__(self, exclude_cols: Optional[List[str]] = None):
        """
        Inicializa el selector de features numéricas.
        
        Args:
            exclude_cols: Lista de nombres de columnas a excluir
        """
        super().__init__()
        self.exclude_cols = exclude_cols or []
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """
        Identifica las columnas numéricas a seleccionar.
        
        Args:
            X: DataFrame o array de entrada
            y: Target (ignorado, para compatibilidad sklearn)
        
        Returns:
            self: Instancia entrenada
        
        Raises:
            ValueError: Si X no es DataFrame y exclude_cols no está vacío
        """
        self._validate_data(X)
        
        if isinstance(X, pd.DataFrame):
            # Seleccionar columnas numéricas
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Excluir columnas especificadas
            self.feature_names_ = [
                col for col in numeric_cols 
                if col not in self.exclude_cols
            ]
            
            logger.info(
                f"NumericFeatureSelector: {len(self.feature_names_)} "
                f"columnas numéricas seleccionadas de {X.shape[1]} totales"
            )
            
            if self.exclude_cols:
                logger.debug(f"Columnas excluidas: {self.exclude_cols}")
        
        elif isinstance(X, np.ndarray):
            if self.exclude_cols:
                raise ValueError(
                    "exclude_cols solo funciona con DataFrames. "
                    "Para arrays, todas las columnas se seleccionan."
                )
            self.feature_names_ = list(range(X.shape[1]))
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transforma el input seleccionando solo columnas numéricas.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_transformed: Datos con solo columnas numéricas seleccionadas
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            X_transformed = X[self.feature_names_]
            logger.debug(f"Features seleccionadas: {X_transformed.shape[1]}")
            return X_transformed
        
        return X
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Retorna nombres de features de salida."""
        return self.feature_names_


class PowerFeatureTransformer(FeatureTransformer):
    """
    Aplica transformación de potencia para normalizar distribuciones.
    
    Wrapper alrededor de sklearn.preprocessing.PowerTransformer que:
    - Maneja tanto Yeo-Johnson como Box-Cox
    - Preserva formato DataFrame
    - Proporciona logging detallado
    
    El método Yeo-Johnson funciona con valores positivos y negativos.
    Box-Cox requiere valores estrictamente positivos.
    
    Attributes:
        method: Método de transformación ('yeo-johnson' o 'box-cox')
        transformer: Instancia de PowerTransformer de sklearn
    
    Ejemplo:
        >>> transformer = PowerFeatureTransformer(method='yeo-johnson')
        >>> X_normalized = transformer.fit_transform(X_train)
        >>> X_test_normalized = transformer.transform(X_test)
    """
    
    def __init__(self, method: str = "yeo-johnson", standardize: bool = True):
        """
        Inicializa el transformador de potencia.
        
        Args:
            method: 'yeo-johnson' o 'box-cox'
            standardize: Si True, estandariza después de transformar
        
        Raises:
            ValueError: Si method no es válido
        """
        super().__init__()
        
        valid_methods = ['yeo-johnson', 'box-cox']
        if method not in valid_methods:
            raise ValueError(
                f"method debe ser uno de {valid_methods}, "
                f"recibido: {method}"
            )
        
        self.method = method
        self.standardize = standardize
        self.transformer = None
    
    def fit(self, X, y=None):
        """
        Entrena el transformador de potencia.
        
        Args:
            X: Datos de entrada
            y: Target (ignorado)
        
        Returns:
            self: Instancia entrenada
        """
        self._validate_data(X)
        
        # Crear y entrenar el transformer de sklearn
        self.transformer = PowerTransformer(
            method=self.method,
            standardize=self.standardize
        )
        
        # Convertir a array si es DataFrame
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        self.transformer.fit(X_array)
        
        logger.info(
            f"PowerFeatureTransformer entrenado con método '{self.method}' "
            f"en {self.n_features_in_} features"
        )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Aplica la transformación de potencia.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_transformed: Datos transformados
        """
        self._check_is_fitted()
        
        # Guardar referencia al formato original
        X_original = X
        
        # Convertir a array si es necesario
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Transformar
        X_transformed = self.transformer.transform(X_array)
        
        # Preservar formato DataFrame si corresponde
        X_transformed = self._preserve_dataframe_format(
            X_original,
            X_transformed
        )
        
        logger.debug(f"PowerTransform aplicado a {X_transformed.shape[1]} features")
        
        return X_transformed


class OutlierRemover(FeatureTransformer):
    """
    Elimina outliers usando el método IQR (Interquartile Range).
    
    Detecta valores atípicos calculando Q1, Q3 y el rango intercuartil (IQR),
    y marca como outliers los valores fuera de [Q1 - factor*IQR, Q3 + factor*IQR].
    
    IMPORTANTE: Este transformer elimina FILAS completas, no valores individuales.
    Por lo tanto, el número de samples puede reducirse.
    
    Attributes:
        factor: Factor multiplicador del IQR (típicamente 1.5 o 3.0)
        lower_bounds_: Límites inferiores por feature
        upper_bounds_: Límites superiores por feature
        n_outliers_removed_: Número de outliers removidos en fit
    
    Ejemplo:
        >>> remover = OutlierRemover(factor=1.5)
        >>> X_clean = remover.fit_transform(X_train)
        >>> print(f"Outliers removidos: {remover.n_outliers_removed_}")
    """
    
    def __init__(self, factor: float = 1.5):
        """
        Inicializa el removedor de outliers.
        
        Args:
            factor: Factor para calcular límites (1.5 = moderado, 3.0 = extremo)
        """
        super().__init__()
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.n_outliers_removed_ = 0
    
    def fit(self, X, y=None):
        """
        Calcula los límites de outliers basados en IQR.
        
        Args:
            X: Datos de entrada
            y: Target (ignorado)
        
        Returns:
            self: Instancia entrenada
        """
        self._validate_data(X)
        
        # Convertir a array para cálculos
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Calcular percentiles
        Q1 = np.percentile(X_array, 25, axis=0)
        Q3 = np.percentile(X_array, 75, axis=0)
        IQR = Q3 - Q1
        
        # Calcular límites
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        
        logger.info(
            f"OutlierRemover: límites calculados con factor {self.factor}"
        )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Remueve filas con outliers.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_clean: Datos sin outliers
        
        Note:
            El número de filas puede reducirse. Usar solo en training.
        """
        self._check_is_fitted()
        
        # Convertir a array para comparaciones
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Identificar filas sin outliers
        mask = np.all(
            (X_array >= self.lower_bounds_) & (X_array <= self.upper_bounds_),
            axis=1
        )
        
        self.n_outliers_removed_ = np.sum(~mask)
        
        # Filtrar
        if isinstance(X, pd.DataFrame):
            X_clean = X[mask]
        else:
            X_clean = X_array[mask]
        
        if self.n_outliers_removed_ > 0:
            logger.info(
                f"OutlierRemover: {self.n_outliers_removed_} outliers removidos "
                f"({self.n_outliers_removed_/len(X)*100:.2f}%)"
            )
        
        return X_clean


class FeatureScaler(FeatureTransformer):
    """
    Escalador versátil con múltiples métodos de escalado.
    
    Soporta:
    - 'standard': StandardScaler (media=0, std=1)
    - 'minmax': MinMaxScaler (rango [0,1])
    - 'robust': RobustScaler (usa mediana e IQR, robusto a outliers)
    
    Attributes:
        method: Método de escalado
        scaler: Instancia del scaler de sklearn
    
    Ejemplo:
        >>> scaler = FeatureScaler(method='robust')
        >>> X_scaled = scaler.fit_transform(X_train)
    """
    
    VALID_METHODS = ['standard', 'minmax', 'robust']
    
    def __init__(self, method: str = 'standard'):
        """
        Inicializa el escalador.
        
        Args:
            method: Método de escalado ('standard', 'minmax', 'robust')
        
        Raises:
            ValueError: Si method no es válido
        """
        super().__init__()
        
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method debe ser uno de {self.VALID_METHODS}, "
                f"recibido: {method}"
            )
        
        self.method = method
        self.scaler = None
    
    def fit(self, X, y=None):
        """
        Entrena el escalador.
        
        Args:
            X: Datos de entrada
            y: Target (ignorado)
        
        Returns:
            self: Instancia entrenada
        """
        self._validate_data(X)
        
        # Seleccionar scaler según método
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        
        # Entrenar
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.scaler.fit(X_array)
        
        logger.info(
            f"FeatureScaler entrenado con método '{self.method}' "
            f"en {self.n_features_in_} features"
        )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Escala los datos.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_scaled: Datos escalados
        """
        self._check_is_fitted()
        
        X_original = X
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Escalar
        X_scaled = self.scaler.transform(X_array)
        
        # Preservar formato
        X_scaled = self._preserve_dataframe_format(X_original, X_scaled)
        
        logger.debug(f"Escalado '{self.method}' aplicado a {X_scaled.shape[1]} features")
        
        return X_scaled


class CorrelationFilter(FeatureTransformer):
    """
    Filtra features altamente correlacionadas.
    
    Elimina una de cada par de features con correlación absoluta mayor
    al threshold especificado. Útil para reducir multicolinealidad.
    
    Attributes:
        threshold: Umbral de correlación (0-1)
        features_to_keep_: Índices de features a mantener
        correlation_matrix_: Matriz de correlación calculada
    
    Ejemplo:
        >>> filter = CorrelationFilter(threshold=0.95)
        >>> X_filtered = filter.fit_transform(X_train)
        >>> print(f"Features retenidas: {len(filter.features_to_keep_)}")
    """
    
    def __init__(self, threshold: float = 0.95):
        """
        Inicializa el filtro de correlación.
        
        Args:
            threshold: Umbral de correlación (valores mayores se filtran)
        
        Raises:
            ValueError: Si threshold no está en (0, 1)
        """
        super().__init__()
        
        if not 0 < threshold < 1:
            raise ValueError(f"threshold debe estar entre 0 y 1, recibido: {threshold}")
        
        self.threshold = threshold
        self.features_to_keep_ = None
        self.correlation_matrix_ = None
    
    def fit(self, X, y=None):
        """
        Identifica features correlacionadas a remover.
        
        Args:
            X: Datos de entrada
            y: Target (ignorado)
        
        Returns:
            self: Instancia entrenada
        """
        self._validate_data(X)
        
        # Convertir a DataFrame si es necesario
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        # Calcular matriz de correlación
        self.correlation_matrix_ = X_df.corr().abs()
        
        # Identificar features a mantener
        upper_triangle = np.triu(np.ones(self.correlation_matrix_.shape), k=1).astype(bool)
        to_drop = [
            column for column in range(len(self.correlation_matrix_.columns))
            if any(self.correlation_matrix_.iloc[column, upper_triangle[column]] > self.threshold)
        ]
        
        self.features_to_keep_ = [
            i for i in range(self.n_features_in_)
            if i not in to_drop
        ]
        
        n_removed = self.n_features_in_ - len(self.features_to_keep_)
        logger.info(
            f"CorrelationFilter: {n_removed} features removidas "
            f"(correlación > {self.threshold})"
        )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Filtra features correlacionadas.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_filtered: Datos con features no correlacionadas
        """
        self._check_is_fitted()
        
        if isinstance(X, pd.DataFrame):
            cols_to_keep = [X.columns[i] for i in self.features_to_keep_]
            return X[cols_to_keep]
        else:
            return X[:, self.features_to_keep_]


class VarianceThresholdSelector(FeatureTransformer):
    """
    Elimina features con varianza menor al umbral.
    
    Wrapper alrededor de sklearn.feature_selection.VarianceThreshold
    que preserva formato DataFrame y proporciona logging.
    
    Attributes:
        threshold: Umbral de varianza mínima
        selector: Instancia de VarianceThreshold de sklearn
    
    Ejemplo:
        >>> selector = VarianceThresholdSelector(threshold=0.01)
        >>> X_selected = selector.fit_transform(X_train)
    """
    
    def __init__(self, threshold: float = 0.0):
        """
        Inicializa el selector por varianza.
        
        Args:
            threshold: Umbral de varianza mínima (default 0.0 = remover constantes)
        """
        super().__init__()
        self.threshold = threshold
        self.selector = None
    
    def fit(self, X, y=None):
        """
        Identifica features con varianza suficiente.
        
        Args:
            X: Datos de entrada
            y: Target (ignorado)
        
        Returns:
            self: Instancia entrenada
        """
        self._validate_data(X)
        
        self.selector = VarianceThreshold(threshold=self.threshold)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.selector.fit(X_array)
        
        n_removed = self.n_features_in_ - self.selector.transform(X_array).shape[1]
        logger.info(
            f"VarianceThresholdSelector: {n_removed} features removidas "
            f"(varianza < {self.threshold})"
        )
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        """
        Filtra features con baja varianza.
        
        Args:
            X: Datos de entrada
        
        Returns:
            X_selected: Features con varianza suficiente
        """
        self._check_is_fitted()
        
        X_original = X
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_selected = self.selector.transform(X_array)
        
        # Obtener nombres de features seleccionadas
        if isinstance(X, pd.DataFrame):
            selected_features = [
                X.columns[i] for i in range(len(X.columns))
                if self.selector.get_support()[i]
            ]
            return pd.DataFrame(
                X_selected,
                columns=selected_features,
                index=X.index
            )
        
        return X_selected


# ============================================================================
# BUILDER PATTERN - Construcción de pipelines
# ============================================================================

class FeaturePipelineBuilder:
    """
    Constructor de pipelines de features usando patrón Builder.
    
    Permite construcción fluida de pipelines complejos:
    - Métodos encadenables
    - Validación automática
    - Nomenclatura clara
    
    Ejemplo:
        >>> pipeline = (FeaturePipelineBuilder()
        ...     .add_numeric_selector()
        ...     .add_outlier_remover(factor=1.5)
        ...     .add_power_transformer()
        ...     .add_feature_scaler(method='robust')
        ...     .build())
        >>> X_processed = pipeline.fit_transform(X_train)
    """
    
    def __init__(self):
        """Inicializa el builder con lista vacía de steps."""
        self.steps = []
        logger.debug("FeaturePipelineBuilder inicializado")
    
    def add_numeric_selector(
        self,
        exclude_cols: Optional[List[str]] = None,
        name: str = "numeric_selector"
    ):
        """
        Agrega selector de columnas numéricas.
        
        Args:
            exclude_cols: Columnas a excluir
            name: Nombre del step en el pipeline
        
        Returns:
            self: Para encadenamiento
        """
        self.steps.append((name, NumericFeatureSelector(exclude_cols)))
        logger.debug(f"Step agregado: {name}")
        return self
    
    def add_power_transformer(
        self,
        method: str = "yeo-johnson",
        name: str = "power_transformer"
    ):
        """
        Agrega transformador de potencia.
        
        Args:
            method: Método ('yeo-johnson' o 'box-cox')
            name: Nombre del step
        
        Returns:
            self: Para encadenamiento
        """
        self.steps.append((name, PowerFeatureTransformer(method)))
        logger.debug(f"Step agregado: {name}")
        return self
    
    def add_outlier_remover(
        self,
        factor: float = 1.5,
        name: str = "outlier_remover"
    ):
        """
        Agrega removedor de outliers.
        
        Args:
            factor: Factor IQR
            name: Nombre del step
        
        Returns:
            self: Para encadenamiento
        """
        self.steps.append((name, OutlierRemover(factor)))
        logger.debug(f"Step agregado: {name}")
        return self
    
    def add_feature_scaler(
        self,
        method: str = 'standard',
        name: str = "scaler"
    ):
        """
        Agrega escalador de features.
        
        Args:
            method: Método de escalado
            name: Nombre del step
        
        Returns:
            self: Para encadenamiento
        """
        self.steps.append((name, FeatureScaler(method)))
        logger.debug(f"Step agregado: {name}")
        return self
    
    def add_correlation_filter(
        self,
        threshold: float = 0.95,
        name: str = "correlation_filter"
    ):
        """
        Agrega filtro de correlación.
        
        Args:
            threshold: Umbral de correlación
            name: Nombre del step
        
        Returns:
            self: Para encadenamiento
        """
        self.steps.append((name, CorrelationFilter(threshold)))
        logger.debug(f"Step agregado: {name}")
        return self
    
    def add_variance_selector(
        self,
        threshold: float = 0.0,
        name: str = "variance_selector"
    ):
        """
        Agrega selector por varianza.
        
        Args:
            threshold: Umbral de varianza
            name: Nombre del step
        
        Returns:
            self: Para encadenamiento
        """
        self.steps.append((name, VarianceThresholdSelector(threshold)))
        logger.debug(f"Step agregado: {name}")
        return self
    
    def build(self):
        """
        Construye el Pipeline de sklearn.
        
        Returns:
            Pipeline: Pipeline de sklearn con todos los steps
        
        Raises:
            ValueError: Si no hay steps agregados
        """
        if not self.steps:
            raise ValueError("No se agregaron steps al pipeline")
        
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(self.steps)
        
        logger.info(f"Pipeline construido con {len(self.steps)} steps")
        return pipeline


# ============================================================================
# FACTORY FUNCTIONS - Pipelines pre-configurados
# ============================================================================

def create_preprocessing_pipeline(
    exclude_cols: Optional[List[str]] = None,
    power_transform: bool = True
) -> 'Pipeline':
    """
    Crea pipeline de preprocessing básico.
    
    Pipeline estándar:
    1. Selección de columnas numéricas
    2. Transformación de potencia (opcional)
    
    Args:
        exclude_cols: Columnas a excluir
        power_transform: Si True, aplica PowerTransformer
    
    Returns:
        Pipeline configurado
    
    Ejemplo:
        >>> pipeline = create_preprocessing_pipeline(exclude_cols=['id'])
        >>> X_processed = pipeline.fit_transform(X_train)
    """
    builder = FeaturePipelineBuilder().add_numeric_selector(exclude_cols)
    
    if power_transform:
        builder.add_power_transformer()
    
    return builder.build()


def create_feature_selection_pipeline(
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95
) -> 'Pipeline':
    """
    Crea pipeline de selección de features.
    
    Pipeline:
    1. Filtro por varianza
    2. Filtro por correlación
    
    Args:
        variance_threshold: Umbral de varianza mínima
        correlation_threshold: Umbral de correlación máxima
    
    Returns:
        Pipeline configurado
    
    Ejemplo:
        >>> pipeline = create_feature_selection_pipeline()
        >>> X_selected = pipeline.fit_transform(X_train)
    """
    return (FeaturePipelineBuilder()
            .add_variance_selector(variance_threshold)
            .add_correlation_filter(correlation_threshold)
            .build())


def create_full_pipeline(
    exclude_cols: Optional[List[str]] = None,
    remove_outliers: bool = True,
    scale_method: str = 'standard'
) -> 'Pipeline':
    """
    Crea pipeline completo de procesamiento.
    
    Pipeline robusto:
    1. Selección numérica
    2. Remoción de outliers (opcional)
    3. Power Transform
    4. Escalado
    
    Args:
        exclude_cols: Columnas a excluir
        remove_outliers: Si True, remueve outliers
        scale_method: Método de escalado ('standard', 'minmax', 'robust')
    
    Returns:
        Pipeline configurado
    
    Ejemplo:
        >>> pipeline = create_full_pipeline(
        ...     exclude_cols=['id'],
        ...     remove_outliers=True,
        ...     scale_method='robust'
        ... )
        >>> X_ready = pipeline.fit_transform(X_train)
    """
    builder = FeaturePipelineBuilder().add_numeric_selector(exclude_cols)
    
    if remove_outliers:
        builder.add_outlier_remover(factor=1.5)
    
    builder.add_power_transformer()
    builder.add_feature_scaler(method=scale_method)
    
    return builder.build()


# ============================================================================
# FUNCIONES LEGACY - Compatibilidad backward
# ============================================================================

def create_preprocessing_pipeline_legacy(exclude_cols=None):
    """
    Función legacy para compatibilidad.
    
    .. deprecated:: 2024.10
        Usar :func:`create_preprocessing_pipeline` en su lugar.
    """
    warnings.warn(
        "create_preprocessing_pipeline_legacy() está deprecated. "
        "Usar create_preprocessing_pipeline() en su lugar.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_preprocessing_pipeline(exclude_cols)
