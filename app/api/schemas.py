from pydantic import BaseModel, Field
from typing import List, Optional

class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: int
    probabilities: Optional[List[float]] = None

class TrainRequest(BaseModel):
    # si quieres entrenar por endpoint: ruta al dataset en storage o parámetros
    data_path: str
    model_name: Optional[str] = Field("model_v1", description="model name to save")
