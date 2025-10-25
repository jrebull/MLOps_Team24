#!/usr/bin/env python3
"""
Pre-MLflow/DVC Sync Verification

Verifica que todo está sincronizado y listo antes de empezar
con experiment tracking y data versioning.

Autor: MLOps Team 24
"""

import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("🔍 VERIFICACIÓN DE SINCRONIZACIÓN PRE-MLFLOW/DVC")
print("=" * 70)

checks = []

def check_passed(name, details=""):
    checks.append((name, "✅", details))
    print(f"✅ {name}")
    if details:
        print(f"   {details}")

def check_warning(name, details=""):
    checks.append((name, "⚠️", details))
    print(f"⚠️  {name}")
    if details:
        print(f"   {details}")

def check_failed(name, details=""):
    checks.append((name, "❌", details))
    print(f"❌ {name}")
    if details:
        print(f"   {details}")

# ============================================================================
# CHECK 1: GIT STATUS
# ============================================================================
print("\n[CHECK 1/6] Estado de Git...")
print("-" * 70)

try:
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True,
        text=True,
        check=True
    )
    
    if result.stdout.strip():
        untracked = []
        modified = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('??'):
                untracked.append(line[3:])
            elif line.startswith(' M') or line.startswith('M '):
                modified.append(line[3:])
        
        if untracked:
            check_warning("Git: Archivos sin trackear", f"{len(untracked)} archivos")
            for f in untracked[:5]:
                print(f"      - {f}")
            if len(untracked) > 5:
                print(f"      ... y {len(untracked)-5} más")
        
        if modified:
            check_warning("Git: Archivos modificados sin commit", f"{len(modified)} archivos")
            for f in modified[:5]:
                print(f"      - {f}")
    else:
        check_passed("Git: Working directory limpio")
    
    # Ver último commit
    result = subprocess.run(
        ['git', 'log', '-1', '--oneline'],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"   Último commit: {result.stdout.strip()}")
    
    # Ver branch actual
    result = subprocess.run(
        ['git', 'branch', '--show-current'],
        capture_output=True,
        text=True,
        check=True
    )
    print(f"   Branch actual: {result.stdout.strip()}")
    
except Exception as e:
    check_failed("Git status", str(e))

# ============================================================================
# CHECK 2: ESTRUCTURA DE ARCHIVOS CRÍTICOS
# ============================================================================
print("\n[CHECK 2/6] Archivos críticos presentes...")
print("-" * 70)

critical_files = {
    "acoustic_ml/__init__.py": "Módulo principal",
    "acoustic_ml/modeling/sklearn_pipeline.py": "Pipeline sklearn",
    "acoustic_ml/modeling/train.py": "Training module",
    "acoustic_ml/features.py": "Feature engineering",
    "acoustic_ml/dataset.py": "Dataset management",
    "data/processed/X_train.csv": "Training data",
    "data/processed/X_test.csv": "Test data",
    "requirements.txt": "Dependencies",
    "README.md": "Documentation"
}

missing_files = []
for filepath, description in critical_files.items():
    if Path(filepath).exists():
        print(f"   ✓ {description}: {filepath}")
    else:
        print(f"   ✗ {description}: {filepath}")
        missing_files.append(filepath)

if missing_files:
    check_failed("Archivos críticos", f"{len(missing_files)} archivos faltantes")
else:
    check_passed("Archivos críticos presentes")

# ============================================================================
# CHECK 3: TESTS FUNCIONANDO
# ============================================================================
print("\n[CHECK 3/6] Tests funcionando...")
print("-" * 70)

test_files = [
    "test_sklearn_pipeline.py",
    "test_full_integration.py"
]

tests_ok = True
for test_file in test_files:
    if Path(test_file).exists():
        print(f"   ✓ Test existe: {test_file}")
    else:
        print(f"   ✗ Test falta: {test_file}")
        tests_ok = False

if tests_ok:
    check_passed("Tests presentes y listos")
else:
    check_warning("Tests", "Algunos tests faltan")

# ============================================================================
# CHECK 4: MODELOS Y DATOS
# ============================================================================
print("\n[CHECK 4/6] Modelos y datos...")
print("-" * 70)

data_dir = Path("data")
models_dir = Path("models")

# Verificar data/
if data_dir.exists():
    raw_files = list((data_dir / "raw").glob("*.csv")) if (data_dir / "raw").exists() else []
    processed_files = list((data_dir / "processed").glob("*.csv")) if (data_dir / "processed").exists() else []
    
    print(f"   ✓ Data directory existe")
    print(f"   - Raw files: {len(raw_files)}")
    print(f"   - Processed files: {len(processed_files)}")
    
    if processed_files:
        check_passed("Datos procesados disponibles")
    else:
        check_warning("Datos procesados", "No se encontraron archivos procesados")
else:
    check_failed("Data directory", "No existe")

# Verificar models/
if models_dir.exists():
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
    print(f"   ✓ Models directory existe")
    print(f"   - Modelos guardados: {len(model_files)}")
    
    if model_files:
        check_passed("Modelos guardados encontrados")
    else:
        check_warning("Modelos", "No hay modelos guardados aún (normal)")
else:
    check_warning("Models directory", "No existe (se creará con MLflow)")

# ============================================================================
# CHECK 5: REPORTES Y ANÁLISIS
# ============================================================================
print("\n[CHECK 5/6] Reportes y análisis generados...")
print("-" * 70)

reports_dir = Path("reports/figures")
if reports_dir.exists():
    reports = list(reports_dir.glob("*"))
    print(f"   ✓ Reports directory existe")
    print(f"   - Archivos generados: {len(reports)}")
    
    expected_reports = [
        "outlier_analysis.png",
        "outlier_boxplots.png",
        "outlier_analysis_report.txt",
        "scaler_comparison_results.txt"
    ]
    
    found = 0
    for report in expected_reports:
        if (reports_dir / report).exists():
            print(f"   ✓ {report}")
            found += 1
        else:
            print(f"   ✗ {report}")
    
    if found == len(expected_reports):
        check_passed("Todos los reportes generados")
    else:
        check_warning("Reportes", f"{found}/{len(expected_reports)} reportes encontrados")
else:
    check_warning("Reports directory", "No existe")

# ============================================================================
# CHECK 6: REQUIREMENTS Y ENVIRONMENT
# ============================================================================
print("\n[CHECK 6/6] Dependencies y environment...")
print("-" * 70)

if Path("requirements.txt").exists():
    with open("requirements.txt") as f:
        reqs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"   ✓ requirements.txt existe ({len(reqs)} dependencias)")
    
    # Verificar packages críticos
    critical_packages = ['scikit-learn', 'pandas', 'numpy', 'matplotlib']
    found_packages = []
    
    for pkg in critical_packages:
        if any(pkg.lower() in req.lower() for req in reqs):
            found_packages.append(pkg)
    
    print(f"   ✓ Packages críticos: {', '.join(found_packages)}")
    
    # Verificar si MLflow/DVC ya están
    has_mlflow = any('mlflow' in req.lower() for req in reqs)
    has_dvc = any('dvc' in req.lower() for req in reqs)
    
    if has_mlflow:
        print(f"   ✓ MLflow ya en requirements")
    else:
        print(f"   ℹ️  MLflow no en requirements (se agregará)")
    
    if has_dvc:
        print(f"   ✓ DVC ya en requirements")
    else:
        print(f"   ℹ️  DVC no en requirements (se agregará)")
    
    check_passed("Dependencies configuradas")
else:
    check_failed("requirements.txt", "No existe")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESUMEN DE VERIFICACIÓN")
print("=" * 70)

passed = sum(1 for _, status, _ in checks if status == "✅")
warnings = sum(1 for _, status, _ in checks if status == "⚠️")
failed = sum(1 for _, status, _ in checks if status == "❌")

for name, status, details in checks:
    detail_str = f" ({details})" if details else ""
    print(f"{status} {name}{detail_str}")

print("\n" + "-" * 70)
print(f"✅ Pasados: {passed}")
print(f"⚠️  Warnings: {warnings}")
print(f"❌ Fallidos: {failed}")
print("-" * 70)

# Recomendaciones
print("\n🎯 RECOMENDACIONES:")

if failed > 0:
    print("   ❌ HAY CHECKS FALLIDOS - Resolver antes de continuar")
    print("   → Revisar archivos faltantes")
    print("   → Verificar estructura del proyecto")
elif warnings > 0:
    print("   ⚠️  Hay algunos warnings")
    print("   → Revisar archivos sin commitear (si los hay)")
    print("   → Pero puedes continuar con MLflow")
else:
    print("   ✅ TODO ESTÁ SINCRONIZADO Y LISTO")
    print("   🚀 Puedes comenzar con MLflow/DVC")

print("\n💡 PRÓXIMOS PASOS:")
print("   1. Revisar y resolver cualquier warning/error")
print("   2. Hacer commit si hay cambios pendientes")
print("   3. Iniciar setup de MLflow (tracking server)")
print("   4. Configurar DVC con AWS S3")
print("   5. Integrar tracking en sklearn_pipeline")

if failed == 0:
    sys.exit(0)
else:
    sys.exit(1)
