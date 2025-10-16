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


# Definición de constantes para la ruta de los datos y la columna objetivo.
DATA_PATH = "data/raw/acoustic_features.csv"
TARGET = "Class"

def main():
    """
    Función principal que ejecuta el pipeline de entrenamiento del modelo de
    clasificación de características acústicas (MER).
    """
    # Establecer semillas para reproducibilidad del experimento.
    np.random.seed(7)
    random.seed(7)

    # Configuración de MLflow local para el seguimiento de experimentos.
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("Equipo24-MER")

    # 1. Carga de Datos y Separación de Variables
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET].copy() # Variable objetivo (Clase)
    X = df.drop(columns=[TARGET]).copy() # Variables predictoras

    # 2. Preprocesamiento de la Variable Objetivo
    # Si la variable objetivo es de tipo 'object' (cadenas), se codifica a numérica.
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Reemplazo de valores infinitos por NaN y luego se imputan con 0.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 3. División de Datos
    # Separación del dataset en conjuntos de entrenamiento (80%) y prueba (20%)
    # con estratificación para mantener la proporción de clases en 'y'.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=7
    )

    # 4. Escalamiento de Características
    # Inicialización y ajuste del StandardScaler a los datos de entrenamiento
    # y transformación de ambos conjuntos (entrenamiento y prueba).
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Modelo RF (baseline 2)
    # 5. Definición y Entrenamiento del Modelo
    # Se utiliza un Random Forest Classifier (modelo RF, baseline 2).
    params = dict(n_estimators=200, max_depth=None, random_state=7, n_jobs=-1)
    clf = RandomForestClassifier(**params)

    # Inicia el registro de una nueva ejecución (run) en MLflow.
    with mlflow.start_run(run_name="rf-baseline"):
        # Entrenamiento del modelo
        clf.fit(X_train_sc, y_train)
        # Predicción en el conjunto de prueba
        y_pred = clf.predict(X_test_sc)

        # 6. Evaluación de Métricas
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        print(f"Accuracy: {acc:.4f} | F1(weighted): {f1:.4f}")

        # Guardar artefactos locales (DVC los versiona)
        # 7. Persistencia Local de Artefactos (para DVC u otros)
        # Crear directorios 'models' y 'metrics' si no existen.
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("metrics").mkdir(parents=True, exist_ok=True)

        # Guardar el modelo, el escalador y la lista de características.
        model_path = "models/baseline_model.pkl"
        joblib.dump({"model": clf, "scaler": scaler, "features": list(X.columns)}, model_path)
        with open("metrics/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_weighted": f1}, f, indent=2)

        # 8. Registro en MLflow
        # Log del tipo de modelo usado.
        mlflow.log_param("model", "RandomForestClassifier")
        # Log de los hiperparámetros del modelo.
        for k, v in params.items():
            mlflow.log_param(k, v if v is not None else "None")
        # Log de las métricas clave.
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1)

        # Sube artefactos (el JSON de métricas y el modelo serializado) a MLflow.        mlflow.log_artifact("metrics/metrics.json")
        mlflow.log_artifact(model_path)

if __name__ == "__main__":
    # Punto de entrada de la ejecución del script.
    main()
