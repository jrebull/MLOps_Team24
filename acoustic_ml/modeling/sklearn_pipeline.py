"""
Sklearn Pipeline Integration Module

Este módulo integra todos los componentes de acoustic_ml en un pipeline
end-to-end compatible con scikit-learn.

Autor: MLOps Team 24
Fecha: Octubre 2025
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Imports de nuestro módulo
from acoustic_ml.features import create_full_pipeline
from acoustic_ml.modeling.train import BaseModelTrainer, ModelConfig, TrainingConfig
from acoustic_ml.modeling.evaluate import ModelEvaluator


class SklearnMLPipeline(BaseEstimator):
    """
    Pipeline end-to-end que integra preprocesamiento y entrenamiento.
    
    Compatible con scikit-learn para poder usar en GridSearchCV, cross_val_score, etc.
    
    Attributes:
        feature_pipeline: Pipeline de preprocesamiento de features
        model_trainer: Instancia de ModelTrainer para entrenar modelos
        model_evaluator: Instancia de ModelEvaluator para evaluación
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        model_params: Optional[Dict[str, Any]] = None,
        feature_config: Optional[Dict[str, Any]] = None,
        scale_method: str = "robust"  # ✅ CAMBIO 1: Añadir parámetro
    ):
        """
        Inicializa el pipeline end-to-end.
        
        Args:
            model_type: Tipo de modelo ('random_forest', 'svm', etc.)
            model_params: Parámetros para el modelo
            feature_config: Configuración para el pipeline de features
            scale_method: Método de escalado ('robust', 'standard', 'minmax')
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.feature_config = feature_config or {}
        self.scale_method = scale_method  # ✅ CAMBIO 2: Guardar parámetro
        
        # Se inicializarán en fit()
        self.feature_pipeline = None
        self.model_trainer = None
        self.model_evaluator = None
        self.is_fitted_ = False
    
    def _get_model_class(self, model_type: str):
        """
        Obtiene la clase del modelo y parámetros por defecto.
        
        Args:
            model_type: Tipo de modelo
            
        Returns:
            Tuple con (clase_modelo, params_default)
        """
        models = {
            'random_forest': (
                RandomForestClassifier,
                {'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'n_jobs': -1}
            ),
            'gradient_boosting': (
                GradientBoostingClassifier,
                {'n_estimators': 100, 'max_depth': 3, 'random_state': 42}
            ),
            'svm': (
                SVC,
                {'kernel': 'rbf', 'C': 1.0, 'random_state': 42}
            ),
            'logistic_regression': (
                LogisticRegression,
                {'max_iter': 1000, 'random_state': 42}
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Model type '{model_type}' no soportado. Use: {list(models.keys())}")
        
        return models[model_type]
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el pipeline completo.
        
        Args:
            X: DataFrame con features
            y: Series con target
            
        Returns:
            self: Instancia entrenada
        """
        print("🚀 Iniciando entrenamiento del pipeline...")
        
        # 1. Crear y entrenar pipeline de features
        print("  → Creando pipeline de features...")
        # Pipeline completo SIN OutlierRemover (no elimina filas)
        # Usando RobustScaler basado en análisis de outliers
        # Decisión: RobustScaler es más robusto a outliers (usa mediana e IQR)
        self.feature_pipeline = create_full_pipeline(
            exclude_cols=None,
            remove_outliers=False,  # ⚠️ Crítico: False para no eliminar filas
            scale_method=self.scale_method  # ✅ CAMBIO 3: Usar parámetro en lugar de hardcoded 'robust'
        )
        
        print("  → Transformando features de entrenamiento...")
        X_transformed = self.feature_pipeline.fit_transform(X)
        
        # 2. Configurar modelo
        print(f"  → Configurando modelo {self.model_type}...")
        model_class, default_params = self._get_model_class(self.model_type)
        
        # Merge default params with user params
        hyperparameters = {**default_params, **self.model_params}
        
        # Crear configs
        model_config = ModelConfig(
            name=f"{self.model_type}_pipeline",
            model_class=model_class,
            hyperparameters=hyperparameters
        )
        
        training_config = TrainingConfig(
            cv_folds=5,
            scoring="accuracy",
            mlflow_tracking_uri=None,  # ❌ Deshabilitado - logging manejado externamente
            mlflow_experiment="Equipo24-MER"  # Experimento del equipo
        )
        
        # 3. Entrenar modelo
        print(f"  → Entrenando modelo {self.model_type}...")
        self.model_trainer = BaseModelTrainer(model_config, training_config)
        self.model_trainer.train(X_transformed, y)
        
        # 4. Marcar como entrenado
        self.is_fitted_ = True
        
        print("✅ Pipeline entrenado exitosamente")
        print(f"   Features transformadas: {X_transformed.shape}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el pipeline entrenado.
        
        IMPORTANTE: Devuelve labels decodificados (strings), no números.
        
        Args:
            X: DataFrame con features RAW (sin transformar)
            
        Returns:
            Array con predicciones como strings (e.g., 'happy', 'sad')
        """
        if not self.is_fitted_:
            raise ValueError("El pipeline debe ser entrenado primero (llamar fit)")
        
        # 1. Transformar features usando el pipeline entrenado
        X_transformed = self.feature_pipeline.transform(X)
        
        # 2. Predecir usando el modelo entrenado (devuelve números encoded)
        predictions_encoded = self.model_trainer.model.predict(X_transformed)
        
        # 3. ✅ DECODIFICAR a strings usando label_encoder
        if hasattr(self.model_trainer, 'label_encoder') and self.model_trainer.label_encoder is not None:
            predictions_decoded = self.model_trainer.label_encoder.inverse_transform(predictions_encoded)
            return predictions_decoded
        else:
            # Si no hay label_encoder, devolver números
            return predictions_encoded
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice probabilidades por clase.
        
        Args:
            X: DataFrame con features RAW (sin transformar)
            
        Returns:
            Array con probabilidades [n_samples, n_classes]
        """
        if not self.is_fitted_:
            raise ValueError("El pipeline debe ser entrenado primero")
        
        X_transformed = self.feature_pipeline.transform(X)
        probas = self.model_trainer.model.predict_proba(X_transformed)
        
        return probas
    
    def get_classes(self) -> np.ndarray:
        """
        Retorna las clases del modelo (labels decodificados).
        
        Returns:
            Array con nombres de clases
        """
        if not self.is_fitted_:
            raise ValueError("El pipeline debe ser entrenado primero")
        
        if hasattr(self.model_trainer, 'label_encoder') and self.model_trainer.label_encoder is not None:
            return self.model_trainer.label_encoder.classes_
        else:
            return self.model_trainer.model.classes_

        
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calcula el accuracy del pipeline.
        
        Args:
            X: DataFrame con features
            y: Series con target
            
        Returns:
            Accuracy score
        """
        if not self.is_fitted_:
            raise ValueError("El pipeline debe ser entrenado primero (llamar fit)")
        
        # 1. Hacer predicciones
        predictions = self.predict(X)
        
        # 2. Calcular accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, predictions)
        
        return accuracy


def create_sklearn_pipeline(
    model_type: str = "random_forest",
    model_params: Optional[Dict[str, Any]] = None,
    scale_method: str = "robust"  # ✅ CAMBIO 4: Añadir parámetro a función factory
) -> SklearnMLPipeline:
    """
    Factory function para crear un pipeline sklearn end-to-end.
    
    Args:
        model_type: Tipo de modelo a usar
        model_params: Parámetros del modelo
        scale_method: Método de escalado ('robust', 'standard', 'minmax')
        
    Returns:
        Pipeline configurado listo para fit()
        
    Example:
        >>> pipeline = create_sklearn_pipeline("random_forest")
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict(X_test)
    """
    return SklearnMLPipeline(
        model_type=model_type,
        model_params=model_params,
        scale_method=scale_method  # ✅ CAMBIO 5: Pasar parámetro al constructor
    )


if __name__ == "__main__":
    print("🧪 Testing sklearn_pipeline module...")
    print("✅ Module structure created")
