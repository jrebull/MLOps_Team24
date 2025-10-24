"""
Módulo de inferencia de modelos con diseño OOP y manejo robusto de errores.
"""
import pickle
import logging
from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np

from acoustic_ml.config import MODELS_DIR

# Configurar logging
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Maneja la carga de modelos y predicciones con validación y manejo de errores.
    
    Atributos:
        model_path (Path): Path to the trained model file
        model: Loaded scikit-learn model
    """
    
    def __init__(self, model_name: str = "baseline_model.pkl"):
        """
        Inicializar predictor con ruta del modelo.
        
        Argumentos:
            model_name: Name of the model file
        """
        self.model_path = MODELS_DIR / model_name
        self.model = None
        logger.info(f"ModelPredictor initialized with model: {model_name}")
    
    def load_model(self) -> None:
        """
        Cargar modelo entrenado desde disco.
        
        Lanza:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado en {self.model_path}. "
                "Por favor entrene un modelo primero."
            )
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Modelo cargado exitosamente desde {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Falló al cargar modelo: {e}")
            raise
    
    def predict(
        self, 
        X: pd.DataFrame, 
        return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Realizar predicciones sobre features de entrada.
        
        Argumentos:
            X: Feature DataFrame
            return_proba: If True, return class probabilities
        
        Retorna:
            Predictions array or DataFrame with probabilities
        
        Lanza:
            ValueError: If model not loaded or invalid input
        """
        if self.model is None:
            raise ValueError("Modelo no cargado. Llame load_model() primero.")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("La entrada debe ser un pandas DataFrame")
        
        if X.empty:
            raise ValueError("El DataFrame de entrada está vacío")
        
        try:
            if return_proba and hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X)
                logger.info(f"✅ Generadas predicciones de probabilidad para {len(X)} muestras")
            else:
                predictions = self.model.predict(X)
                logger.info(f"✅ Generadas predicciones para {len(X)} muestras")
            
            return predictions
        
        except Exception as e:
            logger.error(f"❌ Predicción falló: {e}")
            raise
    
    def predict_single(
        self, 
        features: Union[dict, pd.Series], 
        return_proba: bool = False
    ) -> Union[str, dict]:
        """
        Realizar predicción para una sola muestra.
        
        Argumentos:
            features: Dictionary or Series with feature values
            return_proba: If True, return class probabilities
        
        Retorna:
            Single prediction or probability dictionary
        """
        if isinstance(features, dict):
            X = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            X = features.to_frame().T
        else:
            raise ValueError("Las features deben ser dict o pd.Series")
        
        predictions = self.predict(X, return_proba=return_proba)
        
        if return_proba:
            classes = self.model.classes_
            return {cls: float(prob) for cls, prob in zip(classes, predictions[0])}
        else:
            return predictions[0]
    
    def predict_batch(
        self, 
        X: pd.DataFrame, 
        batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Realizar predicciones en lotes (útil para datasets grandes).
        
        Argumentos:
            X: Feature DataFrame
            batch_size: Number of muestras per batch (None = all at once)
        
        Retorna:
            DataFrame with predictions
        """
        if batch_size is None:
            predictions = self.predict(X)
            return pd.DataFrame({'predictions': predictions})
        
        all_predictions = []
        n_muestras = len(X)
        
        for i in range(0, n_muestras, batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_preds = self.predict(batch)
            all_predictions.extend(batch_preds)
            logger.debug(f"Lote procesado {i//batch_size + 1}")
        
        return pd.DataFrame({'predictions': all_predictions})


# Funciones de conveniencia para retrocompatibilidad
def load_model(model_name: str = "baseline_model.pkl"):
    """
    Cargar un modelo entrenado (interfaz legacy).
    
    Argumentos:
        model_name: Name of the model file
    
    Retorna:
        Loaded model
    """
    predictor = ModelPredictor(model_name)
    predictor.load_model()
    return predictor.model


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Realizar predicciones con un modelo (interfaz legacy).
    
    Argumentos:
        model: Trained model
        X: Features for prediction
    
    Retorna:
        DataFrame with predictions
    """
    predictions = model.predict(X)
    return pd.DataFrame({'predictions': predictions})
