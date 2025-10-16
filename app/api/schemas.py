from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    """
    Define el esquema de datos esperado para la solicitud de predicción (entrada al endpoint /predict).
    """
    features: List[float]
    """La lista de características (features) de entrada para la predicción. Debe ser una lista de números de punto flotante."""

class PredictResponse(BaseModel):
    """
    Define el esquema de datos devuelto por la respuesta de predicción (salida del endpoint /predict).
    """
    prediction: int
    """El resultado de la predicción, representado como un entero (la clase predicha)."""

    probabilities: Optional[List[float]] = None
    """
    Lista opcional de probabilidades de clase. Será None si el modelo no soporta 
    'predict_proba' o si no se calcula.
    """

class TrainRequest(BaseModel):
    """
    Define el esquema de datos esperado para la solicitud de entrenamiento (entrada al endpoint /train).
    """
    # si quieres entrenar por endpoint: ruta al dataset en storage o parámetros
    data_path: str
    """Ruta local o remota (e.g., S3, DVC) al archivo de datos que se utilizará para el entrenamiento."""

    model_name: Optional[str] = Field("model_v1", description="model name to save")
    """Nombre que se le asignará al modelo entrenado al guardarse. Por defecto es 'model_v1'."""
