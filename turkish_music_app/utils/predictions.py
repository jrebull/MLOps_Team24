import logging
import pandas as pd
from typing import Dict
from pathlib import Path

from utils.audio_feature_extractor import AudioFeatureExtractor

logger = logging.getLogger(__name__)

# Mapeo de clases numéricas
CLASS_MAPPING = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "relax"
}

class MusicEmotionPredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path or Path(__file__).parent.parent / 'models' / 'production_model.pkl'
        self.feature_extractor = AudioFeatureExtractor(sr=22050)
    
    def load_model(self):
        try:
            import joblib
            self.model = joblib.load(self.model_path)
            logger.info(f'✅ Modelo cargado desde: {self.model_path.name}')
            return str(self.model_path.name)
        except Exception as e:
            logger.error(f'Error cargando modelo: {e}')
            raise
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        if self.model is None:
            self.load_model()
        
        try:
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # Detectar tipo de modelo y obtener clases
            if hasattr(self.model, 'model_trainer'):
                # SklearnMLPipeline
                classes = self.model.model_trainer.label_encoder.classes_
                classes = [str(c).lower() for c in classes]
            elif hasattr(self.model, 'classes_'):
                # Modelo sklearn estándar - SIEMPRE convertir usando mapeo
                raw_classes = self.model.classes_
                classes = []
                for c in raw_classes:
                    try:
                        # Intentar convertir a int y mapear
                        classes.append(CLASS_MAPPING.get(int(c), str(c)).lower())
                    except (ValueError, TypeError):
                        # Si no es número, usar como string
                        classes.append(str(c).lower())
            else:
                # Fallback
                classes = ['happy', 'sad', 'angry', 'relax']
            
            # Convertir predicción a emoción (forzar conversión de números)
            try:
                pred_num = int(prediction)
                emotion = CLASS_MAPPING.get(pred_num, f"class_{pred_num}").lower()
            except (ValueError, TypeError):
                emotion = str(prediction).lower()
            
            pred_idx = classes.index(emotion) if emotion in classes else 0
            confidence = float(probabilities[pred_idx])
            
            proba_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            
            logger.info(f"✅ Predicción: {emotion} ({confidence:.1%}) usando {self.model_path.name}")
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": proba_dict,
                "model_used": str(self.model_path.name)
            }
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise
    
    def predict_from_audio(self, audio_path: Path, return_probabilities: bool = True) -> Dict:
        try:
            logger.info(f"📊 Extracting features from: {audio_path.name}")
            features_dict = self.feature_extractor.extract_features(audio_path)
            features_df = pd.DataFrame([features_dict])
            
            result = self.predict(features_df)
            
            response = {
                "predicted_emotion": result["emotion"].capitalize(),
                "confidence": result["confidence"],
                "model_used": result["model_used"]
            }
            
            if return_probabilities:
                response["probabilities"] = {
                    emotion.capitalize(): prob 
                    for emotion, prob in result["probabilities"].items()
                }
                response["features"] = features_dict
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in predict_from_audio: {e}")
            raise
