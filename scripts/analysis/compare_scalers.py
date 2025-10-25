#!/usr/bin/env python3
"""
Comparación: StandardScaler vs RobustScaler

Compara el rendimiento del pipeline con ambos escaladores
para tomar la mejor decisión basada en datos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score
import time

print("=" * 70)
print("⚖️  COMPARACIÓN: StandardScaler vs RobustScaler")
print("=" * 70)

# Cargar datos
print("\n[1/3] Cargando datos...")
data_dir = Path("data/processed")
X_train = pd.read_csv(data_dir / "X_train.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
X_test = pd.read_csv(data_dir / "X_test.csv")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()

print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

# Importar módulos
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline

# ============================================================================
# TEST 1: StandardScaler
# ============================================================================
print("\n[2/3] Probando StandardScaler...")
print("   " + "-" * 66)

# Crear pipeline con StandardScaler
from acoustic_ml.features import create_full_pipeline
from acoustic_ml.modeling.train import BaseModelTrainer, ModelConfig, TrainingConfig
from sklearn.ensemble import RandomForestClassifier

# Crear pipeline manualmente para probar StandardScaler
print("   → Creando pipeline con StandardScaler...")
feature_pipeline_standard = create_full_pipeline(
    exclude_cols=None,
    remove_outliers=False,
    scale_method='standard'
)

# Transform features
X_train_standard = feature_pipeline_standard.fit_transform(X_train)
X_test_standard = feature_pipeline_standard.transform(X_test)

# Train model
model_config = ModelConfig(
    name="RandomForest_Standard",
    model_class=RandomForestClassifier,
    hyperparameters={'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'n_jobs': -1}
)
training_config = TrainingConfig(cv_folds=5, scoring="accuracy", mlflow_tracking_uri="")

trainer_standard = BaseModelTrainer(model_config, training_config)

start_time = time.time()
trainer_standard.train(X_train_standard, y_train)
train_time_standard = time.time() - start_time

# Evaluate
from sklearn.metrics import accuracy_score, f1_score
y_pred_standard = trainer_standard.model.predict(X_test_standard)
acc_standard = accuracy_score(y_test, y_pred_standard)
f1_standard = f1_score(y_test, y_pred_standard, average='weighted')

# Cross-validation
cv_scores_standard = cross_val_score(
    trainer_standard.model, X_train_standard, y_train, cv=5, scoring='accuracy'
)

print(f"   ✅ StandardScaler completado")
print(f"      Training time: {train_time_standard:.2f}s")
print(f"      Test Accuracy: {acc_standard:.4f}")
print(f"      Test F1-Score: {f1_standard:.4f}")
print(f"      CV Accuracy: {cv_scores_standard.mean():.4f} (±{cv_scores_standard.std():.4f})")

# ============================================================================
# TEST 2: RobustScaler
# ============================================================================
print("\n[3/3] Probando RobustScaler...")
print("   " + "-" * 66)

# Crear pipeline con RobustScaler
print("   → Creando pipeline con RobustScaler...")
feature_pipeline_robust = create_full_pipeline(
    exclude_cols=None,
    remove_outliers=False,
    scale_method='robust'
)

# Transform features
X_train_robust = feature_pipeline_robust.fit_transform(X_train)
X_test_robust = feature_pipeline_robust.transform(X_test)

# Train model
model_config_robust = ModelConfig(
    name="RandomForest_Robust",
    model_class=RandomForestClassifier,
    hyperparameters={'n_estimators': 100, 'max_depth': None, 'random_state': 42, 'n_jobs': -1}
)

trainer_robust = BaseModelTrainer(model_config_robust, training_config)

start_time = time.time()
trainer_robust.train(X_train_robust, y_train)
train_time_robust = time.time() - start_time

# Evaluate
y_pred_robust = trainer_robust.model.predict(X_test_robust)
acc_robust = accuracy_score(y_test, y_pred_robust)
f1_robust = f1_score(y_test, y_pred_robust, average='weighted')

# Cross-validation
cv_scores_robust = cross_val_score(
    trainer_robust.model, X_train_robust, y_train, cv=5, scoring='accuracy'
)

print(f"   ✅ RobustScaler completado")
print(f"      Training time: {train_time_robust:.2f}s")
print(f"      Test Accuracy: {acc_robust:.4f}")
print(f"      Test F1-Score: {f1_robust:.4f}")
print(f"      CV Accuracy: {cv_scores_robust.mean():.4f} (±{cv_scores_robust.std():.4f})")

# ============================================================================
# COMPARACIÓN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESULTADOS COMPARATIVOS")
print("=" * 70)

comparison_data = {
    'Métrica': ['Test Accuracy', 'Test F1-Score', 'CV Accuracy (mean)', 'CV Std Dev', 'Training Time (s)'],
    'StandardScaler': [
        f"{acc_standard:.4f}",
        f"{f1_standard:.4f}",
        f"{cv_scores_standard.mean():.4f}",
        f"{cv_scores_standard.std():.4f}",
        f"{train_time_standard:.2f}"
    ],
    'RobustScaler': [
        f"{acc_robust:.4f}",
        f"{f1_robust:.4f}",
        f"{cv_scores_robust.mean():.4f}",
        f"{cv_scores_robust.std():.4f}",
        f"{train_time_robust:.2f}"
    ]
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Determinar ganador
print("\n" + "=" * 70)
print("🏆 DECISIÓN FINAL")
print("=" * 70)

if acc_robust > acc_standard:
    winner = "RobustScaler"
    diff = acc_robust - acc_standard
    print(f"\n✅ GANADOR: RobustScaler")
    print(f"   Mejora en accuracy: +{diff:.4f} ({diff*100:.2f}%)")
elif acc_standard > acc_robust:
    winner = "StandardScaler"
    diff = acc_standard - acc_robust
    print(f"\n✅ GANADOR: StandardScaler")
    print(f"   Mejora en accuracy: +{diff:.4f} ({diff*100:.2f}%)")
else:
    winner = "Empate"
    print(f"\n⚖️  EMPATE: Ambos tienen igual performance")

print("\n🎯 RECOMENDACIÓN:")
if winner == "RobustScaler" or winner == "Empate":
    print("   → Usar RobustScaler")
    print("   → Razones:")
    print("     1. Igual o mejor accuracy que StandardScaler")
    print("     2. Más robusto a outliers (mejor generalización)")
    print("     3. Best practice MLOps para datos con outliers")
    print("     4. Mejor para producción (más estable con datos nuevos)")
else:
    print("   → StandardScaler tiene mejor performance")
    print("   → Pero RobustScaler es más conservador y estable")
    print("   → Decisión: depende de prioridad (accuracy vs robustez)")

# Guardar resultados
reports_dir = Path("reports/figures")
reports_dir.mkdir(parents=True, exist_ok=True)

results_file = reports_dir / "scaler_comparison_results.txt"
with open(results_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("COMPARACIÓN: StandardScaler vs RobustScaler\n")
    f.write("=" * 70 + "\n\n")
    f.write(df_comparison.to_string(index=False))
    f.write(f"\n\nGanador: {winner}\n")

print(f"\n💾 Resultados guardados: {results_file}")
print("\n✅ Comparación completada")
