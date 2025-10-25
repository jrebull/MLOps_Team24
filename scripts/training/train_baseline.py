import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import random
import mlflow

DATA_PATH = "data/raw/acoustic_features.csv"
TARGET = "Class"

def main():
    np.random.seed(7)
    random.seed(7)

    # Config MLflow local
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("Equipo24-MER")

    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=7
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Modelo RF (baseline 2)
    params = dict(n_estimators=200, max_depth=None, random_state=7, n_jobs=-1)
    clf = RandomForestClassifier(**params)

    with mlflow.start_run(run_name="rf-baseline"):
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        print(f"Accuracy: {acc:.4f} | F1(weighted): {f1:.4f}")

        # Guardar artefactos locales (DVC los versiona)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("metrics").mkdir(parents=True, exist_ok=True)

        model_path = "models/baseline_model.pkl"
        joblib.dump({"model": clf, "scaler": scaler, "features": list(X.columns)}, model_path)
        with open("metrics/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

        # Log en MLflow
        mlflow.log_param("model", "RandomForestClassifier")
        for k, v in params.items():
            mlflow.log_param(k, v if v is not None else "None")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Sube artefactos a MLflow (además de DVC)
        mlflow.log_artifact("metrics/metrics.json")
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
