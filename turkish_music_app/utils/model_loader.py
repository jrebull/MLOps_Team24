"""
Model Loading Module - Direct File Loading
Loads trained models from file for production deployment.
Author: MLOps Team 24
Date: November 2025
"""
import logging
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Mapeo de clases del modelo
CLASS_MAPPING = {
    0: "Happy",
    1: "Sad", 
    2: "Angry",
    3: "Relax"
}

class MLflowModelLoader:
    """Handles loading ML models from file for production."""
    
    def __init__(self, run_id: str = None):
        """Initialize model loader (run_id ignored, kept for compatibility)."""
        self.model = None
        self.model_path = Path(__file__).parent.parent / "models" / "production_model.pkl"
        
    def load_model(self):
        """Load model from file."""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info("✅ Model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
            
    def predict(self, features: pd.DataFrame) -> tuple:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        try:
            predictions = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            # Convert to class names
            if hasattr(predictions[0], 'item'):
                pred_labels = [CLASS_MAPPING.get(p.item(), "Unknown") for p in predictions]
            else:
                pred_labels = [CLASS_MAPPING.get(p, p) for p in predictions]
                
            return pred_labels, probabilities
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
