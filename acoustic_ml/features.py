"""
Feature engineering con sklearn-compatible transformers.
Implementa SOLID principles y POO.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Base class para transformadores de features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        raise NotImplementedError("Subclases deben implementar transform()")


class NumericFeatureSelector(FeatureTransformer):
    """Selecciona solo columnas numéricas."""
    
    def __init__(self, exclude_cols=None):
        self.exclude_cols = exclude_cols or []
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_names_ = [col for col in numeric_cols if col not in self.exclude_cols]
        else:
            self.feature_names_ = list(range(X.shape[1]))
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names_]
        return X


class PowerFeatureTransformer(FeatureTransformer):
    """Aplica PowerTransformer de sklearn."""
    
    def __init__(self, method="yeo-johnson"):
        self.method = method
        self.transformer = None
    
    def fit(self, X, y=None):
        self.transformer = PowerTransformer(method=self.method, standardize=True)
        self.transformer.fit(X)
        return self
    
    def transform(self, X):
        X_transformed = self.transformer.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_transformed, columns=X.columns, index=X.index)
        return X_transformed


class FeaturePipelineBuilder:
    """Builder Pattern para construir pipelines."""
    
    def __init__(self):
        self.steps = []
    
    def add_numeric_selector(self, exclude_cols=None):
        self.steps.append(("numeric_selector", NumericFeatureSelector(exclude_cols)))
        return self
    
    def add_power_transformer(self, method="yeo-johnson"):
        self.steps.append(("power_transformer", PowerFeatureTransformer(method)))
        return self
    
    def build(self):
        from sklearn.pipeline import Pipeline
        return Pipeline(self.steps)


def create_preprocessing_pipeline(exclude_cols=None):
    """Factory para pipeline de preprocessing estándar."""
    return (FeaturePipelineBuilder()
            .add_numeric_selector(exclude_cols=exclude_cols)
            .add_power_transformer()
            .build())
