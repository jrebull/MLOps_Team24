"""
Model Loading Module - MLflow Integration

Loads trained models from MLflow following MLOps best practices.
Handles custom SklearnMLPipeline wrapper.

Author: MLOps Team 24
Date: November 2025
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import mlflow
# Configure MLflow tracking URI
mlflow.set_tracking_uri("file:../mlruns")
import mlflow.sklearn

logger = logging.getLogger(__name__)

# Mapeo de clases del modelo (basado en acoustic_ml/dataset.py)
CLASS_MAPPING = {
    0: "Happy",
    1: "Sad",
    2: "Angry",
    3: "Relax"
}


class MLflowModelLoader:
    """
    Handles loading ML models from MLflow tracking server.
    
    Special handling for SklearnMLPipeline custom wrapper that includes
    feature_pipeline and model_trainer attributes.
    
    Single Responsibility: Model loading from MLflow
    Open/Closed: Can be extended with model registry support
    """
    
    def __init__(self, run_id: str):
        """
        Initialize MLflow model loader.
        
        Args:
            run_id: MLflow run ID containing the model
        """
        self.run_id = run_id
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._is_custom_pipeline = False
        self._inner_model = None
        self._feature_pipeline = None
    
    def load_model(self) -> Any:
        """
        Load model from MLflow.
        
        Returns:
            Loaded model (may be wrapped in SklearnMLPipeline)
            
        Raises:
            Exception: If model loading fails
        """
        model_uri = f"runs:/{self.run_id}/model"
        
        try:
            logger.info(f"📥 Loading model from MLflow: {self.run_id[:8]}...")
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Model loaded successfully from MLflow")
            
            # Detect if it's a custom SklearnMLPipeline
            self._detect_pipeline_structure()
            
            # Extract metadata
            self._extract_model_info()
            self._load_run_metadata()
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model from MLflow: {e}")
            raise
    
    def _detect_pipeline_structure(self) -> None:
        """Detect if model is a custom SklearnMLPipeline wrapper."""
        model_type = type(self.model).__name__
        
        if model_type == 'SklearnMLPipeline':
            self._is_custom_pipeline = True
            logger.info("🔧 Detected SklearnMLPipeline wrapper")
            
            # Extract internal components
            if hasattr(self.model, 'feature_pipeline'):
                self._feature_pipeline = self.model.feature_pipeline
                logger.info("   ✅ feature_pipeline found")
            
            if hasattr(self.model, 'model_trainer') and hasattr(self.model.model_trainer, 'model'):
                self._inner_model = self.model.model_trainer.model
                logger.info(f"   ✅ inner model found: {type(self._inner_model).__name__}")
    
    def _extract_model_info(self) -> None:
        """Extract metadata from loaded model."""
        if self.model is None:
            return
        
        try:
            # Get the actual model (inner or outer)
            actual_model = self._inner_model if self._is_custom_pipeline else self.model
            
            self.metadata.update({
                "model_type": type(self.model).__name__,
                "is_custom_pipeline": self._is_custom_pipeline,
                "has_predict_proba": self._check_predict_proba_support(),
                "has_classes": hasattr(actual_model, 'classes_'),
            })
            
            if hasattr(actual_model, 'classes_'):
                # El modelo devuelve índices numéricos, usar el mapeo
                self.metadata["classes"] = [CLASS_MAPPING.get(i, str(i)) for i in actual_model.classes_]
                self.metadata["class_mapping"] = CLASS_MAPPING
            
            if hasattr(actual_model, 'n_features_in_'):
                self.metadata["n_features"] = actual_model.n_features_in_
            
            # Check if it's a standard sklearn pipeline
            if hasattr(self.model, 'named_steps') and not self._is_custom_pipeline:
                self.metadata["is_pipeline"] = True
                self.metadata["pipeline_steps"] = list(self.model.named_steps.keys())
            else:
                self.metadata["is_pipeline"] = False
                
            logger.info(f"📊 Model type: {self.metadata.get('model_type', 'Unknown')}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not extract model info: {e}")
    
    def _check_predict_proba_support(self) -> bool:
        """Check if model supports predict_proba."""
        if self._is_custom_pipeline:
            return hasattr(self._inner_model, 'predict_proba')
        return hasattr(self.model, 'predict_proba')
    
    def _load_run_metadata(self) -> None:
        """Load metadata from MLflow run."""
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(self.run_id)
            
            # Get metrics
            self.metadata["metrics"] = run.data.metrics
            
            # Get key params
            params = run.data.params
            self.metadata["model_params"] = {
                k: v for k, v in params.items() 
                if k.startswith('model_')
            }
            
            accuracy = run.data.metrics.get('test_accuracy', 'N/A')
            logger.info(f"📈 Test Accuracy: {accuracy}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load run metadata: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using loaded model.
        
        Handles both standard models and SklearnMLPipeline wrapper.
        
        Args:
            X: Feature DataFrame (raw features)
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        try:
            if self._is_custom_pipeline:
                # Apply feature pipeline first
                if self._feature_pipeline is not None:
                    X_transformed = self._feature_pipeline.transform(X)
                    X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
                else:
                    X_transformed = X
                
                # Predict with inner model
                predictions = self._inner_model.predict(X_transformed)
            else:
                # Standard model
                predictions = self.model.predict(X)
            
            logger.info(f"✅ Predictions generated for {len(X)} samples")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Handles both standard models and SklearnMLPipeline wrapper.
        
        Args:
            X: Feature DataFrame (raw features)
            
        Returns:
            Probability array [n_samples, n_classes]
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not self._check_predict_proba_support():
            raise ValueError("Model doesn't support probability predictions")
        
        try:
            if self._is_custom_pipeline:
                # Apply feature pipeline first
                if self._feature_pipeline is not None:
                    X_transformed = self._feature_pipeline.transform(X)
                    X_transformed = pd.DataFrame(X_transformed, columns=X.columns)
                else:
                    X_transformed = X
                
                # Predict probabilities with inner model
                probas = self._inner_model.predict_proba(X_transformed)
            else:
                # Standard model
                probas = self.model.predict_proba(X)
            
            logger.info(f"✅ Probabilities generated for {len(X)} samples")
            return probas
            
        except Exception as e:
            logger.error(f"❌ Probability prediction failed: {e}")
            raise
    
    def get_feature_importance(self, feature_names: list) -> Optional[pd.DataFrame]:
        """
        Get feature importance if model supports it.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with features and importance scores, or None
        """
        if self.model is None:
            return None
        
        # Get the actual model
        model_to_check = self._inner_model if self._is_custom_pipeline else self.model
        
        # Handle pipeline case
        if hasattr(model_to_check, 'named_steps'):
            # Try common names for the final estimator
            for step_name in ['classifier', 'estimator', 'model']:
                if step_name in model_to_check.named_steps:
                    model_to_check = model_to_check.named_steps[step_name]
                    break
            else:
                # Get last step
                model_to_check = list(model_to_check.named_steps.values())[-1]
        
        if not hasattr(model_to_check, 'feature_importances_'):
            logger.warning("⚠️  Model doesn't have feature_importances_ attribute")
            return None
        
        try:
            importances = model_to_check.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.warning(f"⚠️  Could not extract feature importance: {e}")
            return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_production_model(run_id: str = "4b2f54ba46ed4e1d8500da915cf05ceb") -> MLflowModelLoader:
    """
    Load production model from MLflow.
    
    Args:
        run_id: MLflow run ID (default: best model 78.51% accuracy)
        
    Returns:
        Initialized MLflowModelLoader
    """
    loader = MLflowModelLoader(run_id=run_id)
    loader.load_model()
    return loader


def quick_predict(X: pd.DataFrame, run_id: str = "4b2f54ba46ed4e1d8500da915cf05ceb") -> np.ndarray:
    """
    Quick prediction convenience function.
    
    Args:
        X: Features to predict (raw features)
        run_id: MLflow run ID
        
    Returns:
        Predictions
    """
    loader = load_production_model(run_id=run_id)
    return loader.predict(X)
