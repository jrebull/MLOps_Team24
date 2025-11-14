"""
ModelService - Integración con MLflow y modelo custom SklearnMLPipeline
"""
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Any, Optional, List, Dict
from datetime import datetime

# MLflow
import mlflow
from mlflow import MlflowClient

# Imports del proyecto
from app.core.config import settings
from app.core.logger import logger

# ========================
# CONFIG
# ========================
MODEL_DIR = Path(settings.MODEL_PATH).parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_MODEL_NAME = "turkish-music-emotion-rf"
MLFLOW_RUN_ID = "3e7ecefffa2343d59a23e6d31e0ab705"
MLFLOW_ARTIFACT_PATH = "model/rf_raw_84pct.pkl"

# Nombres de las 50 características (en orden)
FEATURE_NAMES = [
    '_RMSenergy_Mean',
    '_Lowenergy_Mean',
    '_Fluctuation_Mean',
    '_Tempo_Mean',
    '_MFCC_Mean_1', '_MFCC_Mean_2', '_MFCC_Mean_3', '_MFCC_Mean_4', '_MFCC_Mean_5',
    '_MFCC_Mean_6', '_MFCC_Mean_7', '_MFCC_Mean_8', '_MFCC_Mean_9', '_MFCC_Mean_10',
    '_MFCC_Mean_11', '_MFCC_Mean_12', '_MFCC_Mean_13',
    '_Roughness_Mean',
    '_Roughness_Slope',
    '_Zero-crossingrate_Mean',
    '_AttackTime_Mean',
    '_AttackTime_Slope',
    '_Rolloff_Mean',
    '_Eventdensity_Mean',
    '_Pulseclarity_Mean',
    '_Brightness_Mean',
    '_Spectralcentroid_Mean',
    '_Spectralspread_Mean',
    '_Spectralskewness_Mean',
    '_Spectralkurtosis_Mean',
    '_Spectralflatness_Mean',
    '_EntropyofSpectrum_Mean',
    '_Chromagram_Mean_1', '_Chromagram_Mean_2', '_Chromagram_Mean_3', '_Chromagram_Mean_4',
    '_Chromagram_Mean_5', '_Chromagram_Mean_6', '_Chromagram_Mean_7', '_Chromagram_Mean_8',
    '_Chromagram_Mean_9', '_Chromagram_Mean_10', '_Chromagram_Mean_11', '_Chromagram_Mean_12',
    '_HarmonicChangeDetectionFunction_Mean',
    '_HarmonicChangeDetectionFunction_Std',
    '_HarmonicChangeDetectionFunction_Slope',
    '_HarmonicChangeDetectionFunction_PeriodFreq',
    '_HarmonicChangeDetectionFunction_PeriodAmp',
    '_HarmonicChangeDetectionFunction_PeriodEntropy'
]

# Mapeo de índices a emociones (capitalizadas para schema)
EMOTION_CLASSES = {0: "Happy", 1: "Sad", 2: "Angry", 3: "Relax"}

# Mapeo inverso: nombre raw del modelo -> nombre capitalizado
EMOTION_MAPPING = {
    'happy': 'Happy',
    'sad': 'Sad',
    'angry': 'Angry',
    'relax': 'Relax',
    'Happy': 'Happy',
    'Sad': 'Sad',
    'Angry': 'Angry',
    'Relax': 'Relax'
}


class ModelService:
    """
    Servicio de modelo integrado con MLflow.
    
    - Carga modelos desde MLflow artifacts
    - Manejo de SklearnMLPipeline custom
    - Validación robusta de inputs (50 features)
    - Log automático de predicciones para monitoreo
    """
    
    def __init__(self, model_path: Optional[str] = None, use_mlflow: bool = True) -> None:
        """
        Inicializa el servicio.
        
        Args:
            model_path: Ruta local del modelo (fallback)
            use_mlflow: Si True, intenta cargar desde MLflow
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model: Optional[Any] = None
        self.mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        self.use_mlflow = use_mlflow
        self.prediction_count = 0
        
        logger.info(f"ModelService initialized (use_mlflow={use_mlflow})")

    def load(self, model_path: Optional[str] = None) -> Any:
        """
        Carga el modelo. Intenta MLflow primero, fallback a archivo local.
        
        Args:
            model_path: Ruta local alternativa
            
        Returns:
            El modelo cargado
        """
        if self.model is not None:
            logger.debug("Model already loaded")
            return self.model
        
        # Intentar cargar desde MLflow
        if self.use_mlflow:
            try:
                self.model = self._load_from_mlflow()
                logger.info("✅ Model loaded from MLflow")
                return self.model
            except Exception as e:
                logger.warning(f"Failed to load from MLflow: {e}. Falling back to local file.")
        
        # Fallback: cargar desde archivo local
        path = model_path or self.model_path
        logger.info(f"Loading model from local path: {path}")
        self.model = joblib.load(path)
        logger.info("✅ Model loaded from local file")
        return self.model

    def _load_from_mlflow(self) -> Any:
        """
        Carga el modelo desde MLflow artifacts.
        
        Returns:
            El modelo cargado
        """
        # Descargar artifact
        artifact_dir = mlflow.artifacts.download_artifacts(
            run_id=MLFLOW_RUN_ID,
            artifact_path=MLFLOW_ARTIFACT_PATH.split("/")[0],
            tracking_uri=MLFLOW_TRACKING_URI
        )
        
        model_file = Path(artifact_dir) / MLFLOW_ARTIFACT_PATH.split("/")[-1]
        logger.debug(f"Loading model from: {model_file}")
        
        model = joblib.load(str(model_file))
        return model

    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Realiza predicción con validación de 50 features.
        
        Args:
            features: Lista de 50 características acústicas (en orden específico)
            
        Returns:
            Dict con prediction, emotion, probabilities, etc.
            
        Raises:
            ValueError: Si features tiene dimensión incorrecta
        """
        # Validar input
        if not isinstance(features, (list, tuple)):
            raise ValueError(f"Features debe ser una lista, recibió: {type(features)}")
        
        if len(features) != 50:
            raise ValueError(
                f"Features debe tener 50 elementos, recibió: {len(features)}"
            )
        
        # Cargar modelo si no está cargado
        if self.model is None:
            self.load()
        
        # Convertir a DataFrame con nombres de columnas correctos
        X = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        logger.debug(f"Input shape: {X.shape}, Features: {X.columns.tolist()}")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*feature names.*")
            
            # Predicción (retorna string con nombre de emoción)
            pred_raw = self.model.predict(X)
            emotion_raw = str(pred_raw[0]) if len(pred_raw) > 0 else "Unknown"
            
            # ✅ Normalizar nombre de emoción (mapear a capitalizado)
            emotion_label = EMOTION_MAPPING.get(emotion_raw.lower(), "Unknown")
            
            # Mapear a índice numérico
            pred_class = [k for k, v in EMOTION_CLASSES.items() if v == emotion_label]
            pred_class = pred_class[0] if pred_class else 0
            
            # Probabilidades
            probs = None
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0].tolist()
            
        result = {
            "prediction": pred_class,
            "emotion": emotion_label,
            "probabilities": probs,
            "probabilities_mapped": {
                EMOTION_CLASSES[i]: float(p) for i, p in enumerate(probs)
            } if probs else None,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Log en MLflow
        self._log_prediction(features, result)
        
        self.prediction_count += 1
        logger.info(
            f"Prediction #{self.prediction_count}: {emotion_label} "
            f"(confidence: {max(probs) if probs else 'N/A':.2%})"
        )
        
        return result

    def _log_prediction(self, features: List[float], result: Dict[str, Any]) -> None:
        """
        Registra predicción en MLflow para monitoreo.
        """
        try:
            with mlflow.start_run(run_name="api_prediction", nested=False):
                mlflow.log_param("input_dim", len(features))
                mlflow.log_param("predicted_emotion", result["emotion"])
                
                if result["probabilities"]:
                    max_prob = max(result["probabilities"])
                    mlflow.log_metric("prediction_confidence", max_prob)
                
                logger.debug("Prediction logged to MLflow")
        except Exception as e:
            logger.debug(f"Warning: Could not log to MLflow: {e}")

    def list_models(self) -> List[str]:
        """Lista modelos disponibles (locales)."""
        models = list(MODEL_DIR.glob("*.joblib"))
        return [m.name for m in models]

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información del modelo cargado."""
        return {
            "model_name": MLFLOW_MODEL_NAME,
            "run_id": MLFLOW_RUN_ID,
            "artifact_path": MLFLOW_ARTIFACT_PATH,
            "model_type": type(self.model).__name__ if self.model else None,
            "emotion_classes": EMOTION_CLASSES,
            "expected_input_dim": 50,
            "feature_names": FEATURE_NAMES,
            "predictions_served": self.prediction_count,
        }
