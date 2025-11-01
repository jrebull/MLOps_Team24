"""Production inference module"""
import joblib
import pandas as pd
from pathlib import Path
from .features import FeaturePreprocessor

class ProductionPredictor:
    def __init__(self, model_path: Path):
        self.model = joblib.load(model_path)
        self.preprocessor = FeaturePreprocessor()
    
    def predict(self, features: pd.DataFrame):
        features_processed = self.preprocessor.transform(features)
        return self.model.predict(features_processed)
    
    def predict_proba(self, features: pd.DataFrame):
        features_processed = self.preprocessor.transform(features)
        return self.model.predict_proba(features_processed)
