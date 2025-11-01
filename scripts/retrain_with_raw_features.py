#!/usr/bin/env python3
"""
Reentrenamiento con Features RAW
=================================

Este script reentrena el modelo usando features RAW (sin transformar),
corrigiendo el problema de doble transformación.

Autor: MLOps Team 24
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Imports del proyecto
from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline

print("="*70)
print("🚀 REENTRENAMIENTO CON FEATURES RAW")
print("="*70)

# 1. Cargar datos RAW
print("\n[1/5] Cargando datos RAW...")
dm = DatasetManager()
df = dm.load_processed(filename="turkish_music_emotion_v2_cleaned_full.csv")
print(f"   ✅ Dataset cargado: {df.shape}")

# 2. Limpieza de datos
print("\n[2/5] Limpiando datos...")

if 'mixed_type_col' in df.columns:
    df = df.drop('mixed_type_col', axis=1)
    print(f"   ✅ Eliminada columna 'mixed_type_col'")

X = df.drop('Class', axis=1)
y = df['Class']

print(f"   Labels antes: {y.unique()}")
y = y.str.strip().str.lower()
print(f"   Labels después: {y.unique()}")

valid_mask = ~y.isna()
X = X.loc[valid_mask]
y = y.loc[valid_mask]
print(f"   ✅ Eliminadas {(~valid_mask).sum()} filas con NaN")

if X.isna().any().any():
    print(f"   ⚠️  Imputando NaN en features con 0")
    X = X.fillna(0)

print(f"\n   📊 Datos finales:")
print(f"      X: {X.shape}")
print(f"      y: {y.shape}")
print(f"      Clases: {sorted(y.unique())}")

sample_mean = X.iloc[:, 0].mean()
sample_std = X.iloc[:, 0].std()
if abs(sample_mean) < 1 and sample_std < 5:
    print(f"   ⚠️  WARNING: Datos parecen transformados!")
else:
    print(f"   ✅ Datos confirmados como RAW")

# 3. Split train/test
print("\n[3/5] Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# 4. Entrenar modelo
print("\n[4/5] Entrenando modelo con features RAW...")
pipeline = SklearnMLPipeline(
    model_type='random_forest',
    model_params={
        'n_estimators': 200,
        'max_depth': None,
        'random_state': 42,
        'n_jobs': -1
    },
    scale_method='robust'
)

pipeline.fit(X_train, y_train)

# Evaluar - ✅ predict() ahora devuelve strings directamente
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n   🎯 RESULTADOS:")
print(f"      Train Accuracy: {train_acc*100:.2f}%")
print(f"      Test Accuracy: {test_acc*100:.2f}%")

print(f"\n   📋 Classification Report (Test):")
print(classification_report(y_test, y_pred_test))

# 5. Guardar en MLflow
print("\n[5/5] Guardando en MLflow...")

mlflow.set_tracking_uri("file:///Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24/mlruns")
mlflow.set_experiment("turkish-music-emotion-recognition")

with mlflow.start_run(run_name="RandomForest_RAW_Features_v2_FINAL"):
    # Log params
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", "None")
    mlflow.log_param("scale_method", "robust")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("features_type", "RAW")
    mlflow.log_param("source_file", "turkish_music_emotion_v2_cleaned_full.csv")
    mlflow.log_param("data_cleaning", "removed_nan_normalized_labels")
    mlflow.log_param("fix_applied", "double_transformation_bug_v2")
    mlflow.log_param("predict_returns", "strings")
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("n_samples_train", len(y_train))
    mlflow.log_metric("n_samples_test", len(y_test))
    mlflow.log_metric("n_features", X_train.shape[1])
    
    # Log model
    from mlflow.models import infer_signature
    signature = infer_signature(X_train, y_pred_train)
    
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        signature=signature,
        input_example=X_train.head(5)
    )
    
    run_id = mlflow.active_run().info.run_id
    
    print(f"\n   ✅ Modelo guardado exitosamente!")
    print(f"   📝 Run ID: {run_id}")

print("\n" + "="*70)
print("✅ REENTRENAMIENTO COMPLETADO")
print("="*70)
print(f"\n📝 Siguiente paso:")
print(f"   1. Copiar el run_id de arriba")
print(f"   2. Actualizar turkish_music_app/config.py:")
print(f'      MLFLOW_RUN_ID = "{run_id}"')
print(f"   3. Probar la app con audios reales")
print("="*70)
