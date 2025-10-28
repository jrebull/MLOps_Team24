#!/usr/bin/env python3
"""
Validación rápida de run_mlflow_experiments.py
Verifica que el script corregido funciona ANTES de ejecutar suite completa
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔍 VALIDACIÓN: run_mlflow_experiments.py")
print("=" * 70)

# 1. Validar que el archivo existe
print("\n[1/5] Verificando archivo...")
script_path = Path("acoustic_ml/training/run_mlflow_experiments.py")
if not script_path.exists():
    print(f"❌ ERROR: No se encuentra {script_path}")
    sys.exit(1)
print(f"✅ Archivo encontrado: {script_path}")

# 2. Validar imports
print("\n[2/5] Verificando imports...")
try:
    from acoustic_ml.dataset import DatasetManager
    from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
    from acoustic_ml.config import PROJECT_DIR, MLFLOW_TRACKING_URI
    import mlflow
    import mlflow.sklearn
    print("✅ Todos los imports exitosos")
except Exception as e:
    print(f"❌ ERROR en imports: {e}")
    sys.exit(1)

# 3. Validar que DatasetManager tiene método correcto
print("\n[3/5] Verificando API de DatasetManager...")
try:
    dm = DatasetManager()
    if not hasattr(dm, 'load_train_test_split'):
        print("❌ ERROR: DatasetManager no tiene método load_train_test_split()")
        sys.exit(1)
    print("✅ Método load_train_test_split() disponible")
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

# 4. Validar que puede cargar datos
print("\n[4/5] Verificando carga de datos...")
try:
    X_train, X_test, y_train, y_test = dm.load_train_test_split()
    print(f"✅ Datos cargados: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"   Features: {X_train.shape[1]}")
except Exception as e:
    print(f"❌ ERROR cargando datos: {e}")
    sys.exit(1)

# 5. Validar contenido del script corregido
print("\n[5/5] Verificando corrección en script...")
with open(script_path, 'r') as f:
    content = f.read()
    
    # Verificar que usa método correcto
    if 'dm.load_train_test_split()' in content:
        print("✅ Script usa load_train_test_split() (correcto)")
    else:
        print("⚠️  ADVERTENCIA: Script puede no estar usando método correcto")
    
    # Verificar que NO usa método incorrecto
    if 'get_train_test_split' in content:
        print("❌ ERROR: Script aún contiene get_train_test_split (incorrecto)")
        print("   Reemplaza el archivo con la versión corregida")
        sys.exit(1)
    
    # Verificar que loggea modelo
    if 'mlflow.sklearn.log_model' in content:
        print("✅ Script loggea modelos con mlflow.sklearn.log_model()")
    else:
        print("⚠️  ADVERTENCIA: Script puede no estar loggeando modelos")

# Resumen
print("\n" + "=" * 70)
print("✅ VALIDACIÓN COMPLETA - Script listo para ejecutar")
print("=" * 70)
print("\n💡 Siguiente paso:")
print("   python acoustic_ml/training/run_mlflow_experiments.py")
print("\n📊 Después verifica en MLflow UI:")
print("   mlflow ui --port 5000")
print("   → Columna 'Models' debe mostrar 🔗 model")
print()
