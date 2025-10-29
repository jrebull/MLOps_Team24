import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Base Transformer
# -------------------------------------------------------------------------
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self._validate_data(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        raise NotImplementedError("Subclase debe implementar transform()")

    def _validate_data(self, X, reset=True):
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
        else:
            raise TypeError(f"X debe ser DataFrame o ndarray, recibido: {type(X)}")

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} no está entrenado")

    def _preserve_dataframe_format(self, X_original, X_transformed, feature_names=None):
        if isinstance(X_original, pd.DataFrame):
            if feature_names is None:
                feature_names = X_original.columns.tolist()
            return pd.DataFrame(X_transformed, columns=feature_names, index=X_original.index)
        return X_transformed

# -------------------------------------------------------------------------
# Transformers
# -------------------------------------------------------------------------
class NumericFeatureSelector(FeatureTransformer):
    def __init__(self, exclude_cols: Optional[List[str]] = None):
        super().__init__()
        self.exclude_cols = exclude_cols or []
        self.feature_names_ = []

    def fit(self, X, y=None):
        self._validate_data(X)
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_names_ = [c for c in numeric_cols if c not in self.exclude_cols]
        elif isinstance(X, np.ndarray) and self.exclude_cols:
            raise ValueError("exclude_cols solo funciona con DataFrames")
        else:
            self.feature_names_ = list(range(X.shape[1]))
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names_]
        return X

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_

class PowerFeatureTransformer(FeatureTransformer):
    def __init__(self, method: str = "yeo-johnson", standardize: bool = True):
        super().__init__()
        self.method = method
        self.standardize = standardize
        self.transformer = None

    def fit(self, X, y=None):
        self._validate_data(X)
        self.transformer = PowerTransformer(method=self.method, standardize=self.standardize)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.transformer.fit(X_array)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_transformed = self.transformer.transform(X_array)
        return self._preserve_dataframe_format(X, X_transformed)

class OutlierRemover(FeatureTransformer):
    def __init__(self, factor: float = 1.5):
        super().__init__()
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self._validate_data(X)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        Q1 = np.percentile(X_array, 25, axis=0)
        Q3 = np.percentile(X_array, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bounds_ = Q1 - self.factor * IQR
        self.upper_bounds_ = Q3 + self.factor * IQR
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        mask = np.all((X_array >= self.lower_bounds_) & (X_array <= self.upper_bounds_), axis=1)
        self.n_outliers_removed_ = len(X) - mask.sum()
        return X[mask] if isinstance(X, pd.DataFrame) else X_array[mask]

class FeatureScaler(FeatureTransformer):
    def __init__(self, method: str = 'standard'):
        super().__init__()
        self.method = method
        valid_methods = ["standard", "minmax", "robust"]
        if method not in valid_methods:
            raise ValueError(f"Método inválido: {method}. Usa: {valid_methods}")
        self.scaler = None

    def fit(self, X, y=None):
        self._validate_data(X)
        self.scaler = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}[self.method]()
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.scaler.fit(X_array)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)
        return self._preserve_dataframe_format(X, X_scaled)

class CorrelationFilter(FeatureTransformer):
    def __init__(self, threshold: float = 0.95):
        super().__init__()
        self.threshold = threshold
        self.features_to_keep_ = None

    def fit(self, X, y=None):
        self._validate_data(X)
        X_df = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        corr = X_df.corr().abs()
        upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
        to_drop = [i for i in range(len(corr.columns)) if any(corr.iloc[i, upper[i]] > self.threshold)]
        self.features_to_keep_ = [i for i in range(X_df.shape[1]) if i not in to_drop]
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.features_to_keep_]
        return X[:, self.features_to_keep_]

class VarianceThresholdSelector(FeatureTransformer):
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.selector = None

    def fit(self, X, y=None):
        self._validate_data(X)
        self.selector = VarianceThreshold(threshold=self.threshold)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        self.selector.fit(X_array)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_selected = self.selector.transform(X_array)
        if isinstance(X, pd.DataFrame):
            selected_cols = [X.columns[i] for i in range(X.shape[1]) if self.selector.get_support()[i]]
            return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)
        return X_selected

# -------------------------------------------------------------------------
# Pipeline Builder
# -------------------------------------------------------------------------
class FeaturePipelineBuilder:
    def __init__(self):
        self.steps = []

    def add_numeric_selector(self, exclude_cols=None, name="numeric_selector"):
        self.steps.append((name, NumericFeatureSelector(exclude_cols)))
        return self

    def add_power_transformer(self, method="yeo-johnson", name="power_transformer"):
        self.steps.append((name, PowerFeatureTransformer(method)))
        return self

    def add_outlier_remover(self, factor=1.5, name="outlier_remover"):
        self.steps.append((name, OutlierRemover(factor)))
        return self

    def add_feature_scaler(self, method='standard', name="scaler"):
        self.steps.append((name, FeatureScaler(method)))
        return self

    def add_correlation_filter(self, threshold=0.95, name="correlation_filter"):
        self.steps.append((name, CorrelationFilter(threshold)))
        return self

    def add_variance_selector(self, threshold=0.0, name="variance_selector"):
        self.steps.append((name, VarianceThresholdSelector(threshold)))
        return self

    def build(self):
        if not self.steps:
            raise ValueError("No se agregaron steps al pipeline")
        return Pipeline(self.steps)
# -------------------------------------------------------------------------
# Factory Functions
# -------------------------------------------------------------------------
def create_preprocessing_pipeline(exclude_cols: Optional[List[str]] = None,
                                  power_transform: bool = True) -> Pipeline:
    builder = FeaturePipelineBuilder().add_numeric_selector(exclude_cols)
    if power_transform:
        builder.add_power_transformer()
    return builder.build()


def create_feature_selection_pipeline(variance_threshold: float = 0.01,
                                      correlation_threshold: float = 0.95) -> Pipeline:
    return (FeaturePipelineBuilder()
            .add_variance_selector(variance_threshold)
            .add_correlation_filter(correlation_threshold)
            .build())


def create_full_pipeline(exclude_cols: Optional[List[str]] = None,
                         remove_outliers: bool = True,
                         scale_method: str = 'standard') -> Pipeline:
    builder = FeaturePipelineBuilder().add_numeric_selector(exclude_cols)
    if remove_outliers:
        builder.add_outlier_remover(factor=1.5)
    builder.add_power_transformer()
    builder.add_feature_scaler(method=scale_method)
    return builder.build()
