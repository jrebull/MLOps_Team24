"""
Inference con modelos entrenados
"""
import pickle
import pandas as pd
from pathlib import Path
from acoustic_ml.config import MODELS_DIR


def load_model(model_name: str = "baseline_model.pkl"):
    """
    Carga un modelo entrenado
    
    Args:
        model_name: Nombre del archivo del modelo
        
    Returns:
        Modelo cargado
    """
    model_path = MODELS_DIR / model_name
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza predicciones con un modelo
    
    Args:
        model: Modelo entrenado
        X: Features para predicción
        
    Returns:
        DataFrame con predicciones
    """
    predictions = model.predict(X)
    return pd.DataFrame({'predictions': predictions})
