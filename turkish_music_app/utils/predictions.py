import logging
import pandas as pd
from typing import Dict, Optional
import mlflow
from pathlib import Path

from utils.audio_feature_extractor import AudioFeatureExtractor

logger = logging.getLogger(__name__)

# Mapeo de clases numéricas a nombres
CLASS_MAPPING = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "relax"
}



class MusicEmotionPredictor:
    def __init__(self):
        self.model = None
        self.feature_extractor = AudioFeatureExtractor(sr=22050)
    
    def load_model(self):
        try:
            model_path = Path(__file__).parent.parent / 'models' / 'production_model.pkl'
            import joblib
            self.model = joblib.load(model_path)
            logger.info(f'✅ Modelo cargado desde: {model_path}')
        except Exception as e:
            logger.error(f'Error cargando modelo: {e}')
            raise
            raise
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        if self.model is None:
            self.load_model()
        
        try:
            prediction = self.model.predict(features_df)[0]
            probabilities = self.model.predict_proba(features_df)[0]
            
            # Mapear número a emoción
            emotion = CLASS_MAPPING.get(int(prediction), str(prediction)).lower()
            
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
    
    def predict_from_audio(
        self, 
        audio_path: Path,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
            return_probabilities: Whether to return probabilities
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract features
            logger.info(f"📊 Extracting features from: {audio_path.name}")
            features_dict = self.feature_extractor.extract_features(audio_path)
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features_dict])
            
            # Predict
            result = self.predict(features_df)
            
            # Format response to match expected API
            response = {
                "predicted_emotion": result["emotion"].capitalize(),
                "confidence": result["confidence"]
            }
            
            if return_probabilities:
                # Capitalize emotion names for consistency
                response["probabilities"] = {
                    emotion.capitalize(): prob 
                    for emotion, prob in result["probabilities"].items()
                }
                # Add features if requested
                response["features"] = features_dict
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in predict_from_audio: {e}")
            raise
