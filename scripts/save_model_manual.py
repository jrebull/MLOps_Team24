#!/usr/bin/env python3
"""Guardar modelo manualmente para MLflow 3.4.0"""
import pandas as pd
import mlflow
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline

print("Cargando datos...")
dm = DatasetManager()
df = dm.load_processed(filename="turkish_music_emotion_v2_cleaned_full.csv")
if 'mixed_type_col' in df.columns:
    df = df.drop('mixed_type_col', axis=1)
X = df.drop('Class', axis=1).fillna(0)
y = df['Class'].str.strip().str.lower()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Entrenando modelo...")
pipeline = SklearnMLPipeline(
    model_type='random_forest',
    model_params={'n_estimators': 200, 'max_depth': None, 'random_state': 42, 'n_jobs': -1},
    scale_method='robust'
)
pipeline.fit(X_train, y_train)
y_pred_test = pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Guardar con joblib primero
model_path = Path("models/optimized/rf_raw_84pct.pkl")
joblib.dump(pipeline, model_path)
print(f"✅ Modelo guardado en: {model_path}")

# Ahora registrar en MLflow
mlflow.set_tracking_uri("file:///Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24/mlruns")
mlflow.set_experiment("turkish-music-emotion-recognition")

with mlflow.start_run(run_name="RandomForest_RAW_84pct_FIXED"):
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("features_type", "RAW")
    mlflow.log_metric("test_accuracy", test_acc)
    
    # Guardar modelo como artefacto directamente
    mlflow.log_artifact(str(model_path), artifact_path="model")
    
    run_id = mlflow.active_run().info.run_id
    print(f"✅ Run ID: {run_id}")
