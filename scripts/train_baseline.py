import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import json
import random

DATA_PATH = "data/raw/acoustic_features.csv"   # ajusta si cambiaste el nombre
TARGET = "Class"                                # <- tu columna objetivo

def main():
    # Reproducibilidad
    np.random.seed(42)
    random.seed(42)

    # Carga
    df = pd.read_csv(DATA_PATH)

    # Separar X/y
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    # Label encode si target es string
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Limpieza mínima
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Escalado
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Modelo baseline
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    # Métricas
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    print(f"Accuracy: {acc:.4f} | F1(weighted): {f1:.4f}")

    # Guardar artefactos
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": clf, "scaler": scaler, "features": list(X.columns)},
                "models/baseline_model.pkl")

    with open("metrics/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

if __name__ == "__main__":
    main()
