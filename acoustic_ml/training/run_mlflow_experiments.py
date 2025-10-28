import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
from acoustic_ml.config import PROJECT_DIR, MLFLOW_TRACKING_URI

# Asegura que el proyecto esté en sys.path
sys.path.append(str(PROJECT_DIR))


class MLflowRunner:
    def __init__(self, experiment_name: str = "turkish-music-emotion-recognition"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        print(f"Experiment: {experiment_name}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load train/test splits using DatasetManager.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        dm = DatasetManager()
        X_train, X_test, y_train, y_test = dm.load_train_test_split()
        print(f"Loaded data: {len(X_train)} train samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def get_model_configs(self) -> List[Dict]:
        return [
            {
                "name": "RandomForest",
                "model_type": "random_forest",
                "model_params": {"n_estimators": 200, "max_depth": 20, "random_state": 42}
            },
            {
                "name": "GradientBoosting",
                "model_type": "gradient_boosting",
                "model_params": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1, "random_state": 42}
            },
            {
                "name": "LogisticRegression",
                "model_type": "logistic_regression",
                "model_params": {"max_iter": 1000, "random_state": 42}
            }
        ]

    @staticmethod
    def plot_cm(y_true, y_pred, labels, save_path: Path):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return cm

    def run_experiment(self, config: Dict, X_train, X_test, y_train, y_test, run_number: int):
        model_name = config['name']
        print(f"\nRunning experiment {run_number}: {model_name}")

        with mlflow.start_run(run_name=f"{run_number}_{model_name}"):
            pipeline = create_sklearn_pipeline(config['model_type'], config['model_params'])
            pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            metrics = {
                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),
                "test_precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                "test_recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                "test_f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            }

            mlflow.log_params(config['model_params'])
            mlflow.log_metrics(metrics)

            artifacts_dir = PROJECT_DIR / "mlflow_artifacts" / f"{run_number}_{model_name}"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            cm_path = artifacts_dir / "confusion_matrix.png"
            self.plot_cm(y_test, y_test_pred, labels=sorted(y_test.unique()), save_path=cm_path)
            mlflow.log_artifact(str(cm_path))

            signature = infer_signature(X_train.head(5), pipeline.predict(X_train.head(5)))
            mlflow.sklearn.log_model(pipeline, "model", signature=signature)

            print(f"Experiment {model_name} logged successfully!")
            return {"run_number": run_number, "model_name": model_name, **metrics}

    def run_all(self):
        X_train, X_test, y_train, y_test = self.load_data()
        configs = self.get_model_configs()
        results = []

        for i, cfg in enumerate(configs, 1):
            try:
                result = self.run_experiment(cfg, X_train, X_test, y_train, y_test, i)
                results.append(result)
            except Exception as e:
                print(f"Error in experiment {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        summary_path = PROJECT_DIR / "mlflow_artifacts" / "summary.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    runner = MLflowRunner()
    runner.run_all()
    print("\nAll experiments completed. Run `mlflow ui --port 5001` to view results.")
