#!/usr/bin/env python3
"""
Entrenar nuevo modelo para generar actividad DVC
"""
from datetime import datetime
from pathlib import Path
import pickle
import json

from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline

print("=" * 70)
print(f"🎯 NUEVO EXPERIMENTO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# 1. Cargar datos
print("\n[1/4] Cargando datos...")
dm = DatasetManager()
X_train, X_test, y_train, y_test = dm.load_train_test_split()
print(f"✅ Datos: Train({X_train.shape}), Test({X_test.shape})")

# 2. Entrenar modelo
print("\n[2/4] Entrenando nuevo modelo...")
pipeline = SklearnMLPipeline(model_type='random_forest')
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f"✅ Accuracy: {accuracy:.4f}")

# 3. Guardar modelo
print("\n[3/4] Guardando modelo...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("models/optimized")
output_dir.mkdir(parents=True, exist_ok=True)

model_path = output_dir / f"experiment_{timestamp}.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"✅ Modelo guardado: {model_path}")

# 4. Guardar metadata
metadata = {
    "timestamp": timestamp,
    "model_type": "random_forest",
    "accuracy": float(accuracy),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "features": X_train.shape[1]
}

metadata_path = output_dir / f"experiment_{timestamp}_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata guardado: {metadata_path}")

print("\n[4/4] Resumen:")
print(f"  → Modelo: {model_path.name}")
print(f"  → Accuracy: {accuracy:.4f}")
print(f"  → Listo para DVC")
print("=" * 70)
