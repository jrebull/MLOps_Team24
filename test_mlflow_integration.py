#!/usr/bin/env python3
"""
Test MLflow Integration - Quick Verification

Entrena el pipeline y verifica que los logs aparecen en MLflow UI.
"""

import pandas as pd
from pathlib import Path
import time

print("=" * 70)
print("🧪 TEST: MLflow Integration con SklearnMLPipeline")
print("=" * 70)

# 1. Cargar datos
print("\n[1/4] Cargando datos...")
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
X_test = pd.read_csv(data_dir / "X_test.csv")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# 2. Importar pipeline
print("\n[2/4] Importando pipeline...")
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
print("   ✓ Pipeline importado")

# 3. Crear y entrenar pipeline
print("\n[3/4] Entrenando pipeline con MLflow tracking...")
print("   → Esto puede tomar 15-30 segundos...")

pipeline = create_sklearn_pipeline(
    model_type="random_forest",
    model_params={
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
)

# Entrenar con datos completos
start_time = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - start_time

# Evaluar
test_accuracy = pipeline.score(X_test, y_test)

print(f"   ✓ Pipeline entrenado en {train_time:.2f}s")
print(f"   ✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# 4. Verificar MLflow
print("\n[4/4] Verificando logs en MLflow...")
print("   → Abre MLflow UI: http://127.0.0.1:5000")
print("   → Busca experimento: Equipo24-MER")
print("   → Deberías ver un nuevo run con:")
print("      • Params: n_estimators, max_depth, random_state")
print("      • Metrics: accuracy_mean, accuracy_std")
print("      • Model: RandomForest guardado")

print("\n" + "=" * 70)
print("✅ TEST COMPLETADO")
print("=" * 70)
print("\n🎯 ACCIONES:")
print("   1. Abre MLflow UI: http://127.0.0.1:5000")
print("   2. Click en experimento 'Equipo24-MER'")
print("   3. Verifica el último run (debe ser reciente)")
print("   4. Explora params, metrics, y modelo guardado")
print("\n💡 Si ves el run nuevo → ¡MLflow funcionando perfectamente!")
