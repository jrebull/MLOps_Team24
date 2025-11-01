import logging
import pandas as pd
from typing import Dict
import mlflow
from pathlib import Path

logger = logging.getLogger(__name__)

class MusicEmotionPredictor:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import MLFLOW_RUN_ID
            
            project_root = Path(__file__).parent.parent.parent
            tracking_uri = f"file://{project_root}/mlruns"
            mlflow.set_tracking_uri(tracking_uri)
            
            model_uri = f"runs:/{MLFLOW_RUN_ID}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Modelo cargado: {MLFLOW_RUN_ID}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
            
    def predict(self, features_df: pd.DataFrame) -> Dict:
        if self.model is None:
            self.load_model()
            
        try:
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # El modelo devuelve strings directamente
            emotion = str(prediction).lower()
            
            # Obtener classes del modelo
            classes = [str(c).lower() for c in self.model.classes_]
            
            # Encontrar índice de la predicción
            pred_idx = classes.index(emotion)
            confidence = float(probabilities[pred_idx])
            
            # Crear dict de probabilidades
            proba_dict = {
                cls: float(prob) 
                for cls, prob in zip(classes, probabilities)
            }
            
            logger.info(f"✅ Predicción: {emotion} ({confidence:.1%})")
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": proba_dict
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
