from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import logging
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models import infer_signature


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class ModelConfig:
    """Configuración de un modelo ML."""
    name: str
    model_class: type
    hyperparameters: Dict[str, Any]
    random_state: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Configuración de entrenamiento y MLflow."""
    cv_folds: int = 5
    scoring: str = "accuracy"
    mlflow_tracking_uri: str = "file:///mlruns"
    mlflow_experiment: str = "acoustic_ml_experiments"


class ModelTrainer(ABC):
    @abstractmethod
    def train(self, X, y) -> BaseEstimator:
        pass

    @abstractmethod
    def evaluate(self, X, y) -> Dict[str, float]:
        pass


class BaseModelTrainer(ModelTrainer):

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.model: BaseEstimator | None = None
        self.label_encoder: LabelEncoder | None = None

    def train(self, X, y) -> BaseEstimator:
        logger.info(f"Entrenando modelo: {self.model_config.name}")
        y_encoded = self._encode_labels(y)
        self.model = self._initialize_model()
        self.model.fit(X, y_encoded)

        if self.training_config.mlflow_tracking_uri:
            self._log_to_mlflow(X, y_encoded)

        return self.model

    def evaluate(self, X, y) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado.")
        y_encoded = self._encode_labels(y)
        scores = cross_val_score(
            self.model,
            X,
            y_encoded,
            cv=self.training_config.cv_folds,
            scoring=self.training_config.scoring
        )
        return {
            f"{self.training_config.scoring}_mean": float(scores.mean()),
            f"{self.training_config.scoring}_std": float(scores.std())
        }


    def _initialize_model(self) -> BaseEstimator:
        return self.model_config.model_class(**self.model_config.hyperparameters)

    def _encode_labels(self, y) -> Any:
        if isinstance(y, pd.Series):
            y = y.values
        if y.dtype == object:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                return self.label_encoder.fit_transform(y)
            return self.label_encoder.transform(y)
        return y

    def _log_to_mlflow(self, X, y) -> None:
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)

        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.training_config.mlflow_experiment)

        with mlflow.start_run(run_name=self.model_config.name):

            for k, v in self.model_config.hyperparameters.items():
                mlflow.log_param(k, v if v is not None else "None")


            metrics = self.evaluate(X, y)
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Modelo
            sample_size = min(5, len(X))
            X_sample = X.iloc[:sample_size] if hasattr(X, "iloc") else X[:sample_size]
            predictions = self.model.predict(X_sample)
            signature = infer_signature(X_sample, predictions)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*artifact_path.*")
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    signature=signature,
                    input_example=X_sample
                )

            logger.info(f"Métricas registradas en MLflow: {metrics}")
            logger.info("Modelo registrado en MLflow.")


# -------------------------------
# Función de entrenamiento baseline
# -------------------------------
def train_baseline_model(X_train, y_train) -> tuple[BaseEstimator, BaseModelTrainer]:
    
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
