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

DATA_PATH = "data/raw/acoustic_features.csv"
TARGET = "Class"

def main():
    np.random.seed(7)
    random.seed(7)

    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET]).copy()

    if y.dtype == "object":
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=7
    )

    # Cambiamos a RandomForest
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Como RF no requiere escalado, lo mantenemos por consistencia del pipeline
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=7,
        n_jobs=-1
    )
    clf.fit(X_train_sc, y_train)
    y_pred = clf.predict(X_test_sc)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    print(f"Accuracy: {acc:.4f} | F1(weighted): {f1:.4f}")

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("metrics").mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": clf, "scaler": scaler, "features": list(X.columns)},
                "models/baseline_model.pkl")

    with open("metrics/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

if __name__ == "__main__":
    main()
