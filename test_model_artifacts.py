#!/usr/bin/env python3
"""
Test Model Artifacts - Verificar que el modelo se guarda en MLflow
"""

import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧪 TEST: Model Artifacts en MLflow")
print("=" * 70)

# Cargar datos
print("\n[1/3] Cargando datos...")
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv").head(100)  # Subset para rapidez
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze().head(100)
X_test = pd.read_csv(data_dir / "X_test.csv").head(50)
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze().head(50)
print(f"   ✓ Subset: Train={X_train.shape}, Test={X_test.shape}")

# Crear pipeline
print("\n[2/3] Entrenando pipeline CON model artifacts...")
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline

pipeline = create_sklearn_pipeline(
    model_type="random_forest",
    model_params={
        'n_estimators': 50,  # Menos árboles para rapidez
        'max_depth': 8,
        'random_state': 42
    }
)

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"   ✓ Accuracy: {accuracy:.4f}")

# Instrucciones
print("\n[3/3] Verificación en MLflow UI...")
print("   → Refresca MLflow UI: http://127.0.0.1:5000")
print("   → Busca el run más reciente en 'Equipo24-MER'")
print("   → Click en el run")
print("   → Ve a tab 'Artifacts'")
print("\n   ✅ Deberías ver:")
print("      📁 model/")
print("         ├── MLmodel")
print("         ├── model.pkl")
print("         ├── conda.yaml")
print("         ├── python_env.yaml")
print("         └── requirements.txt")

print("\n" + "=" * 70)
print("✅ TEST COMPLETADO - Verifica artifacts en MLflow UI")
print("=" * 70)
print("\n💡 Si ves la carpeta 'model/' en artifacts:")
print("   🎉 ¡Model logging funcionando perfectamente!")
print("   🚀 Listo para siguiente fase: Comparación de experimentos")
