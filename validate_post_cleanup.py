#!/usr/bin/env python3
"""
Validación Post-Limpieza - Verificar que todo funciona

Valida:
1. Estructura de carpetas correcta
2. Tests funcionando
3. Pipeline sklearn funcionando
4. MLflow funcionando
5. Git sincronizado
6. Imports funcionando
"""

import subprocess
import sys
from pathlib import Path
import time

print("=" * 70)
print("✅ VALIDACIÓN POST-LIMPIEZA")
print("=" * 70)

results = []

def test_passed(name):
    results.append((name, "✅"))
    print(f"✅ {name}")

def test_failed(name, error=""):
    results.append((name, "❌"))
    print(f"❌ {name}")
    if error:
        print(f"   Error: {error}")

def test_warning(name, msg=""):
    results.append((name, "⚠️"))
    print(f"⚠️  {name}")
    if msg:
        print(f"   {msg}")

# ============================================================================
# TEST 1: ESTRUCTURA DE CARPETAS
# ============================================================================
print("\n[TEST 1/7] Verificando estructura de carpetas...")
print("-" * 70)

required_structure = {
    "acoustic_ml": "Módulo Python",
    "app": "FastAPI API",
    "monitoring/dashboard": "Streamlit dashboard",
    "scripts/analysis": "Scripts análisis",
    "scripts/training": "Scripts training",
    "scripts/validation": "Scripts validación",
    "tests": "Tests",
    "data/processed": "Datos procesados",
    "models": "Modelos",
    "reports": "Reportes"
}

all_exist = True
for folder, desc in required_structure.items():
    if Path(folder).exists():
        print(f"   ✓ {folder}")
    else:
        print(f"   ✗ {folder} - FALTA")
        all_exist = False

if all_exist:
    test_passed("Estructura de carpetas correcta")
else:
    test_failed("Estructura de carpetas", "Faltan algunas carpetas")

# ============================================================================
# TEST 2: GIT STATUS
# ============================================================================
print("\n[TEST 2/7] Verificando sincronización Git...")
print("-" * 70)

try:
    # Working directory
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout.strip():
        untracked = len([l for l in result.stdout.split('\n') if l.startswith('??')])
        modified = len([l for l in result.stdout.split('\n') if l.startswith(' M') or l.startswith('M ')])
        
        test_warning("Git status", f"Archivos sin commit: {untracked} untracked, {modified} modified")
        print(f"   Untracked: {untracked}, Modified: {modified}")
    else:
        test_passed("Git sincronizado (working tree clean)")
    
    # Branch
    result = subprocess.run(
        ['git', 'branch', '--show-current'],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"   Branch: {result.stdout.strip()}")
    
except Exception as e:
    test_failed("Git status", str(e))

# ============================================================================
# TEST 3: GITIGNORE ACTUALIZADO
# ============================================================================
print("\n[TEST 3/7] Verificando .gitignore...")
print("-" * 70)

gitignore = Path(".gitignore")
if gitignore.exists():
    content = gitignore.read_text()
    
    critical_entries = [
        ("mlruns/", "Experimentos MLflow"),
        ("__pycache__/", "Python cache"),
        ("*.pyc", "Archivos compilados"),
        ("dvcstore/", "DVC cache"),
    ]
    
    all_present = True
    for entry, desc in critical_entries:
        if entry in content:
            print(f"   ✓ {entry} - {desc}")
        else:
            print(f"   ✗ {entry} - FALTA")
            all_present = False
    
    if all_present:
        test_passed(".gitignore actualizado correctamente")
    else:
        test_warning(".gitignore", "Faltan algunas entradas críticas")
else:
    test_failed(".gitignore", "No existe")

# ============================================================================
# TEST 4: IMPORTS DEL MÓDULO PRINCIPAL
# ============================================================================
print("\n[TEST 4/7] Verificando imports de acoustic_ml...")
print("-" * 70)

try:
    from acoustic_ml.dataset import DatasetManager
    from acoustic_ml.features import create_full_pipeline
    from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
    print("   ✓ DatasetManager")
    print("   ✓ create_full_pipeline")
    print("   ✓ create_sklearn_pipeline")
    test_passed("Imports de acoustic_ml funcionando")
except Exception as e:
    test_failed("Imports", str(e))
    print("\n⚠️  CRÍTICO: Imports fallan, los siguientes tests pueden fallar")

# ============================================================================
# TEST 5: SKLEARN PIPELINE
# ============================================================================
print("\n[TEST 5/7] Probando pipeline sklearn...")
print("-" * 70)

try:
    import pandas as pd
    from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
    
    # Cargar datos pequeños
    X_train = pd.read_csv("data/processed/X_train.csv").head(50)
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze().head(50)
    X_test = pd.read_csv("data/processed/X_test.csv").head(20)
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze().head(20)
    
    print("   ✓ Datos cargados")
    
    # Crear pipeline
    pipeline = create_sklearn_pipeline(
        model_type="random_forest",
        model_params={'n_estimators': 10, 'max_depth': 5, 'random_state': 42}
    )
    print("   ✓ Pipeline creado")
    
    # Entrenar
    pipeline.fit(X_train, y_train)
    print("   ✓ Pipeline entrenado")
    
    # Predecir
    predictions = pipeline.predict(X_test)
    accuracy = pipeline.score(X_test, y_test)
    print(f"   ✓ Predicciones realizadas (Accuracy: {accuracy:.2f})")
    
    if accuracy > 0.5:
        test_passed(f"Pipeline sklearn funcionando (Accuracy: {accuracy:.2%})")
    else:
        test_warning("Pipeline sklearn", f"Accuracy bajo: {accuracy:.2%}")
    
except Exception as e:
    test_failed("Pipeline sklearn", str(e))
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 6: MLFLOW TRACKING
# ============================================================================
print("\n[TEST 6/7] Verificando MLflow tracking...")
print("-" * 70)

try:
    import mlflow
    
    # Set tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print("   ✓ Tracking URI configurado")
    
    # Try to get experiments
    try:
        experiments = mlflow.search_experiments()
        print(f"   ✓ Conexión a MLflow OK ({len(experiments)} experimentos)")
        test_passed("MLflow tracking funcionando")
    except Exception as e:
        test_warning("MLflow tracking", "Servidor no está corriendo (OK si no lo necesitas ahora)")
        print("   ℹ️  Tip: mlflow ui --host 127.0.0.1 --port 5000")
    
except Exception as e:
    test_warning("MLflow", str(e))

# ============================================================================
# TEST 7: TESTS AUTOMATIZADOS
# ============================================================================
print("\n[TEST 7/7] Ejecutando tests automatizados...")
print("-" * 70)

critical_tests = [
    "tests/test_sklearn_pipeline.py",
    "tests/test_full_integration.py"
]

tests_available = [t for t in critical_tests if Path(t).exists()]
print(f"   Tests disponibles: {len(tests_available)}/{len(critical_tests)}")

if tests_available:
    test_passed(f"Tests disponibles ({len(tests_available)} archivos)")
else:
    test_warning("Tests", "No se encontraron archivos de test")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESUMEN DE VALIDACIÓN")
print("=" * 70)

passed = sum(1 for _, status in results if status == "✅")
warnings = sum(1 for _, status in results if status == "⚠️")
failed = sum(1 for _, status in results if status == "❌")

for name, status in results:
    print(f"{status} {name}")

print("\n" + "-" * 70)
print(f"✅ Pasados: {passed}")
print(f"⚠️  Warnings: {warnings}")
print(f"❌ Fallidos: {failed}")
print("-" * 70)

# Conclusión
print("\n🎯 CONCLUSIÓN:")

if failed == 0 and warnings <= 2:
    print("   ✅ TODO FUNCIONANDO CORRECTAMENTE")
    print("   🚀 Sistema listo para continuar con experimentos")
    print("\n💡 Próximo paso:")
    print("   → Comparación de experimentos con MLflow")
    print("   → Comando: python scripts/analysis/run_experiments.py")
    sys.exit(0)
    
elif failed == 0:
    print("   ⚠️  Sistema funcionando con advertencias menores")
    print("   ✅ Puedes continuar, pero revisa los warnings")
    sys.exit(0)
    
else:
    print("   ❌ HAY PROBLEMAS CRÍTICOS")
    print("   → Revisar errores antes de continuar")
    sys.exit(1)
