#!/usr/bin/env python3
"""
Habilitar MLflow en sklearn_pipeline.py
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/sklearn_pipeline.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Cambio 1: Habilitar MLflow y usar puerto correcto + experimento correcto
old_config = """            cv_folds=5,
            scoring="accuracy",
            mlflow_tracking_uri="",  # Deshabilitado por ahora
            mlflow_experiment="sklearn_pipeline"
        )"""

new_config = """            cv_folds=5,
            scoring="accuracy",
            mlflow_tracking_uri="http://127.0.0.1:5000",  # ✅ Habilitado
            mlflow_experiment="Equipo24-MER"  # Experimento del equipo
        )"""

if old_config in content:
    content = content.replace(old_config, new_config)
    file_path.write_text(content)
    print("✅ MLflow habilitado en sklearn_pipeline.py")
    print("   → Tracking URI: http://127.0.0.1:5000")
    print("   → Experimento: Equipo24-MER")
    print("   → Logs automáticos de params y metrics")
else:
    print("⚠️  No se encontró el patrón exacto")
    print("   Buscando alternativa...")
    
    # Intentar cambio más específico
    if 'mlflow_tracking_uri=""' in content:
        content = content.replace('mlflow_tracking_uri=""', 'mlflow_tracking_uri="http://127.0.0.1:5000"')
        print("✅ Tracking URI actualizado")
    
    if 'mlflow_experiment="sklearn_pipeline"' in content:
        content = content.replace('mlflow_experiment="sklearn_pipeline"', 'mlflow_experiment="Equipo24-MER"')
        print("✅ Experimento actualizado")
    
    file_path.write_text(content)

print("\n📊 Configuración MLflow:")
print("   • Params automáticos: n_estimators, max_depth, etc.")
print("   • Metrics automáticos: accuracy_mean, accuracy_std")
print("   • Modelo guardado: RandomForest serializado")
print("\n🎯 Próximo paso: Ejecutar pipeline y ver logs en MLflow UI")
