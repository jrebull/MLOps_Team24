#!/usr/bin/env python3
"""
Test Clean MLflow - Sin warnings
"""

import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧪 TEST: MLflow SIN Warnings")
print("=" * 70)

# Cargar datos
print("\n[1/2] Cargando datos...")
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv").head(100)
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze().head(100)
X_test = pd.read_csv(data_dir / "X_test.csv").head(50)
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze().head(50)
print(f"   ✓ Subset cargado")

# Entrenar
print("\n[2/2] Entrenando pipeline...")
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline

pipeline = create_sklearn_pipeline(
    model_type="random_forest",
    model_params={'n_estimators': 50, 'max_depth': 8, 'random_state': 42}
)

print("   → Entrenando... (esto NO debería mostrar warnings)")
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"\n✅ Entrenamiento completado")
print(f"   Accuracy: {accuracy:.4f}")
print("\n🎯 Si NO viste warnings de MLflow → ¡Perfecto!")
print("   ℹ️  Los únicos logs deberían ser INFO de acoustic_ml")
