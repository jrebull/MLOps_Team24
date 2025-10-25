"""
Entrenamiento de modelos con principios SOLID y patrones de diseño.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración de un modelo."""
    name: str
    model_class: type
    hyperparameters: Dict[str, Any]
    random_state: int = 42


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    cv_folds: int = 5
    scoring: str = "accuracy"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "Equipo24-MER"


class ModelTrainer(ABC):
    """Interfaz para entrenadores."""
    
    @abstractmethod
    def train(self, X, y) -> BaseEstimator:
        pass
    
    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, float]:
        pass


class BaseModelTrainer(ModelTrainer):
    """Entrenador base con patrón Template Method."""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.label_encoder = None
    
    def train(self, X, y):
        logger.info(f"Entrenando: {self.model_config.name}")
        y_encoded = self._encode_labels(y)
        self.model = self._initialize_model()
        self.model.fit(X, y_encoded)
        if self.training_config.mlflow_tracking_uri:
            self._log_to_mlflow(X, y_encoded)
        return self.model
    
    def evaluate(self, X, y):
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        y_encoded = self._encode_labels(y)
        scores = cross_val_score(
            self.model, X, y_encoded,
            cv=self.training_config.cv_folds,
            scoring=self.training_config.scoring
        )
        return {
            f"{self.training_config.scoring}_mean": float(scores.mean()),
            f"{self.training_config.scoring}_std": float(scores.std()),
        }
    
    def _initialize_model(self):
        return self.model_config.model_class(**self.model_config.hyperparameters)
    
    def _encode_labels(self, y):
        if isinstance(y, pd.Series):
            y = y.values
        if y.dtype == object:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                return self.label_encoder.fit_transform(y)
            return self.label_encoder.transform(y)
        return y
    
    def _log_to_mlflow(self, X, y):
        import warnings
        # Suprimir warning deprecation cosmético de artifact_path
        warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')
        
        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.training_config.mlflow_experiment)
        with mlflow.start_run(run_name=self.model_config.name):
            # Log parameters
            for k, v in self.model_config.hyperparameters.items():
                mlflow.log_param(k, v if v is not None else "None")
            
            # Log metrics
            metrics = self.evaluate(X, y)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log model as artifact (con signature y example)
            import numpy as np
            from mlflow.models import infer_signature
            
            # Preparar muestra de datos (5 filas)
            sample_size = min(5, len(X))
            if hasattr(X, 'iloc'):  # DataFrame
                X_sample = X.iloc[:sample_size]
            else:  # numpy array
                X_sample = X[:sample_size]
            
            # Crear input example (primeras 5 filas)
            input_example = X_sample
            
            # Inferir signature del modelo (usando la misma muestra)
            predictions = self.model.predict(X_sample)
            signature = infer_signature(X_sample, predictions)
            
            # Suprimir warning específicamente durante log_model
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*artifact_path.*')
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    signature=signature,
                    input_example=input_example
                )
            
            logger.info(f"MLflow: {metrics}")
            logger.info(f"Model logged to MLflow artifacts")


def train_baseline_model(X_train, y_train):
    """Función factory para entrenar modelo baseline."""
    model_config = ModelConfig(
        name="RandomForest_Baseline",
        model_class=RandomForestClassifier,
        hyperparameters={
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1
        }
    )
    training_config = TrainingConfig()
    trainer = BaseModelTrainer(model_config, training_config)
    model = trainer.train(X_train, y_train)
    return model, trainer
