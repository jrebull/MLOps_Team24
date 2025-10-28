import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline


class MLflowRunner:
    def __init__(self, experiment_name="acoustic_ml_experiments", tracking_uri="./mlruns"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"MLflow tracking URI: {tracking_uri}")
        print(f"Experiment: {experiment_name}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        dm = DatasetManager()
        X_train, X_test, y_train, y_test = dm.get_train_test_split(validate=True)
        print(f"Loaded data: {len(X_train)} train samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def get_model_configs(self) -> List[Dict]:
        return [
            {"name": "RandomForest", "model_type": "random_forest", "model_params": {"n_estimators": 200, "max_depth": 20, "random_state": 42}},
            {"name": "GradientBoosting", "model_type": "gradient_boosting", "model_params": {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.1, "random_state": 42}},
            {"name": "LogisticRegression", "model_type": "logistic_regression", "model_params": {"max_iter": 1000, "random_state": 42}}
        ]

    def plot_cm(self, y_true, y_pred, labels, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return cm

    def run_experiment(self, config, X_train, X_test, y_train, y_test, run_number):
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


            artifacts_dir = Path("mlflow_artifacts") / f"{run_number}_{model_name}"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            cm_path = artifacts_dir / "confusion_matrix.png"
            self.plot_cm(y_test, y_test_pred, labels=sorted(y_test.unique()), save_path=cm_path)
            mlflow.log_artifact(str(cm_path))

            # Log modelo
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

        # Guardar resumen
        summary_path = Path("mlflow_artifacts") / "summary.csv"
        pd.DataFrame(results).to_csv(summary_path, index=False)
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    runner = MLflowRunner()
    runner.run_all()
    print("\nAll experiments completed. Run `mlflow ui --port 5001` to view results.")
