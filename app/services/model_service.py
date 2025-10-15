import joblib
from pathlib import Path
from typing import Any
from src.ml_pipeline import DataLoader, Preprocessor, ModelTrainer
from app.core.config import settings
from app.core.logger import logger

MODEL_DIR = Path(settings.MODEL_PATH).parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ModelService:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self.model = None

    def train(self, data_path: str, model_name: str = "model_v1"):
        logger.info("Starting training pipeline")
        loader = DataLoader(data_path)
        df = loader.load_csv()
        pre = Preprocessor(df)
        pre.encode_labels()
        pre.drop_duplicates()
        pre.drop_constant_columns()
        pre.detect_outliers()
        pre.iqr_outliers()
        pre.normalize_skewed()
        cleaned = pre.get_clean_df()

        trainer = ModelTrainer(cleaned)
        trainer.split_and_scale()
        trainer.train_logistic()  # or choose based on config
        model = trainer.get_trained_model()  # assume method to get sklearn estimator

        path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, path)
        logger.info("Saved model to %s", path)
        return str(path)

    def load(self, model_path: str | None = None):
        path = model_path or self.model_path
        logger.info("Loading model from %s", path)
        self.model = joblib.load(path)
        return self.model

    def predict(self, features: list[float]):
        if self.model is None:
            self.load()
        import numpy as np
        X = np.array(features).reshape(1, -1)
        pred = int(self.model.predict(X)[0])
        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X).tolist()[0]
        return {"prediction": pred, "probabilities": probs}
