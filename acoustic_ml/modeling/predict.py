"""
Model inference module with OOP design and robust error handling.
"""
import pickle
import logging
from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import numpy as np

from acoustic_ml.config import MODELS_DIR

# Configure logging
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Handles model loading and predictions with validation and error handling.
    
    Attributes:
        model_path (Path): Path to the trained model file
        model: Loaded scikit-learn model
    """
    
    def __init__(self, model_name: str = "baseline_model.pkl"):
        """
        Initialize predictor with model path.
        
        Args:
            model_name: Name of the model file
        """
        self.model_path = MODELS_DIR / model_name
        self.model = None
        logger.info(f"ModelPredictor initialized with model: {model_name}")
    
    def load_model(self) -> None:
        """
        Load trained model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train a model first."
            )
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def predict(
        self, 
        X: pd.DataFrame, 
        return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions on input features.
        
        Args:
            X: Feature DataFrame
            return_proba: If True, return class probabilities
        
        Returns:
            Predictions array or DataFrame with probabilities
        
        Raises:
            ValueError: If model not loaded or invalid input
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        try:
            if return_proba and hasattr(self.model, 'predict_proba'):
                predictions = self.model.predict_proba(X)
                logger.info(f"✅ Generated probability predictions for {len(X)} samples")
            else:
                predictions = self.model.predict(X)
                logger.info(f"✅ Generated predictions for {len(X)} samples")
            
            return predictions
        
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def predict_single(
        self, 
        features: Union[dict, pd.Series], 
        return_proba: bool = False
    ) -> Union[str, dict]:
        """
        Make prediction for a single sample.
        
        Args:
            features: Dictionary or Series with feature values
            return_proba: If True, return class probabilities
        
        Returns:
            Single prediction or probability dictionary
        """
        if isinstance(features, dict):
            X = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            X = features.to_frame().T
        else:
            raise ValueError("Features must be dict or pd.Series")
        
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
        Make predictions in batches (useful for large datasets).
        
        Args:
            X: Feature DataFrame
            batch_size: Number of samples per batch (None = all at once)
        
        Returns:
            DataFrame with predictions
        """
        if batch_size is None:
            predictions = self.predict(X)
            return pd.DataFrame({'predictions': predictions})
        
        all_predictions = []
        n_samples = len(X)
        
        for i in range(0, n_samples, batch_size):
            batch = X.iloc[i:i+batch_size]
            batch_preds = self.predict(batch)
            all_predictions.extend(batch_preds)
            logger.debug(f"Processed batch {i//batch_size + 1}")
        
        return pd.DataFrame({'predictions': all_predictions})


# Convenience functions for backward compatibility
def load_model(model_name: str = "baseline_model.pkl"):
    """
    Load a trained model (legacy interface).
    
    Args:
        model_name: Name of the model file
    
    Returns:
        Loaded model
    """
    predictor = ModelPredictor(model_name)
    predictor.load_model()
    return predictor.model


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions with a model (legacy interface).
    
    Args:
        model: Trained model
        X: Features for prediction
    
    Returns:
        DataFrame with predictions
    """
    predictions = model.predict(X)
    return pd.DataFrame({'predictions': predictions})
