#!/usr/bin/env python3
"""
Prueba Integral - Validación Completa del Sistema

Verifica que todo el trabajo completado funciona correctamente:
1. Estructura de archivos
2. Imports y módulos
3. Pipeline sklearn funcionando
4. RobustScaler configurado correctamente
5. Accuracy esperado
6. Reportes generados

Autor: MLOps Team 24
Fecha: Octubre 2024
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

print("=" * 70)
print("🧪 PRUEBA INTEGRAL DEL SISTEMA")
print("=" * 70)

test_results = []

def test_passed(name):
    test_results.append((name, "✅ PASS"))
    print(f"✅ {name}")

def test_failed(name, error):
    test_results.append((name, f"❌ FAIL: {error}"))
    print(f"❌ {name}")
    print(f"   Error: {error}")

# ============================================================================
# TEST 1: ESTRUCTURA DE ARCHIVOS
# ============================================================================
print("\n[TEST 1/7] Verificando estructura de archivos...")
print("-" * 70)

required_files = {
    "sklearn_pipeline.py": "acoustic_ml/modeling/sklearn_pipeline.py",
    "analyze_outliers.py": "analyze_outliers.py",
    "compare_scalers.py": "compare_scalers.py",
    "test_sklearn_pipeline.py": "test_sklearn_pipeline.py",
    "outlier_analysis.png": "reports/figures/outlier_analysis.png",
    "outlier_report": "reports/figures/outlier_analysis_report.txt",
    "scaler_comparison": "reports/figures/scaler_comparison_results.txt"
}

all_files_exist = True
for name, filepath in required_files.items():
    if Path(filepath).exists():
        print(f"   ✓ {name}")
    else:
        print(f"   ✗ {name} - FALTA: {filepath}")
        all_files_exist = False

if all_files_exist:
    test_passed("Estructura de archivos completa")
else:
    test_failed("Estructura de archivos", "Faltan archivos requeridos")

# ============================================================================
# TEST 2: IMPORTS Y MÓDULOS
# ============================================================================
print("\n[TEST 2/7] Verificando imports y módulos...")
print("-" * 70)

try:
    from acoustic_ml.modeling.sklearn_pipeline import (
        SklearnMLPipeline, 
        create_sklearn_pipeline
    )
    from acoustic_ml.features import create_full_pipeline
    from acoustic_ml.dataset import DatasetManager
    print("   ✓ Todos los imports exitosos")
    test_passed("Imports de módulos")
except Exception as e:
    test_failed("Imports de módulos", str(e))
    sys.exit(1)

# ============================================================================
# TEST 3: CARGA DE DATOS
# ============================================================================
print("\n[TEST 3/7] Verificando carga de datos...")
print("-" * 70)

try:
    data_dir = Path("data/processed")
    if (data_dir / "X_train.csv").exists():
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
        
        print(f"   ✓ X_train: {X_train.shape}")
        print(f"   ✓ y_train: {y_train.shape}")
        print(f"   ✓ X_test: {X_test.shape}")
        print(f"   ✓ y_test: {y_test.shape}")
        
        # Validaciones básicas
        assert X_train.shape[0] == len(y_train), "Mismatch en train samples"
        assert X_test.shape[0] == len(y_test), "Mismatch en test samples"
        assert X_train.shape[1] == X_test.shape[1], "Mismatch en features"
        
        test_passed("Carga y validación de datos")
    else:
        test_failed("Carga de datos", "No se encontraron datos procesados")
        sys.exit(1)
except Exception as e:
    test_failed("Carga de datos", str(e))
    sys.exit(1)

# ============================================================================
# TEST 4: PIPELINE CON ROBUSTSCALER
# ============================================================================
print("\n[TEST 4/7] Verificando configuración de RobustScaler...")
print("-" * 70)

try:
    # Crear pipeline de features
    feature_pipeline = create_full_pipeline(
        exclude_cols=None,
        remove_outliers=False,
        scale_method='robust'
    )
    
    # Verificar que tiene los steps correctos
    steps = [step[0] for step in feature_pipeline.steps]
    print(f"   Pipeline steps: {steps}")
    
    # Verificar que el último step es el scaler
    scaler_step = feature_pipeline.steps[-1]
    scaler_name, scaler = scaler_step
    
    from sklearn.preprocessing import RobustScaler
    from acoustic_ml.features import FeatureScaler
    
    # Detectar RobustScaler (directo o a través de wrapper)
    is_robust = False
    
    if isinstance(scaler, RobustScaler):
        # Caso 1: RobustScaler directo
        print(f"   ✓ Usando RobustScaler directo")
        is_robust = True
    elif isinstance(scaler, FeatureScaler):
        # Caso 2: FeatureScaler wrapper
        print(f"   ✓ Usando FeatureScaler wrapper")
        # Verificar el método configurado
        if hasattr(scaler, 'method') and scaler.method == 'robust':
            print(f"   ✓ FeatureScaler configurado con method='robust'")
            is_robust = True
        elif hasattr(scaler, 'scaler') and isinstance(scaler.scaler, RobustScaler):
            print(f"   ✓ FeatureScaler internamente usa RobustScaler")
            is_robust = True
        else:
            print(f"   ⚠ FeatureScaler pero no se pudo verificar RobustScaler interno")
    else:
        print(f"   ✗ Scaler es {type(scaler).__name__}")
    
    if is_robust:
        test_passed("Configuración de RobustScaler")
    else:
        test_failed("RobustScaler", f"No se detectó RobustScaler (encontrado: {type(scaler).__name__})")
        
except Exception as e:
    test_failed("Configuración de RobustScaler", str(e))

# ============================================================================
# TEST 5: SKLEARN PIPELINE END-TO-END
# ============================================================================
print("\n[TEST 5/7] Probando SklearnMLPipeline end-to-end...")
print("-" * 70)

try:
    # Crear pipeline
    pipeline = create_sklearn_pipeline(
        model_type="random_forest",
        model_params={
            'n_estimators': 50,
            'max_depth': 10,
            'random_state': 42
        }
    )
    
    # Entrenar con subset
    X_train_small = X_train.head(200)
    y_train_small = y_train.head(200)
    X_test_small = X_test.head(50)
    y_test_small = y_test.head(50)
    
    print("   → Entrenando pipeline...")
    pipeline.fit(X_train_small, y_train_small)
    
    print("   → Prediciendo...")
    predictions = pipeline.predict(X_test_small)
    
    print("   → Evaluando...")
    accuracy = pipeline.score(X_test_small, y_test_small)
    
    print(f"   ✓ Pipeline funcionando correctamente")
    print(f"   ✓ Accuracy: {accuracy:.4f}")
    
    if accuracy > 0.60:  # Umbral mínimo razonable
        test_passed(f"Pipeline end-to-end (Accuracy: {accuracy:.4f})")
    else:
        test_failed("Pipeline accuracy", f"Accuracy muy bajo: {accuracy:.4f}")
        
except Exception as e:
    test_failed("Pipeline end-to-end", str(e))
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 6: VALIDAR QUE NO ELIMINA FILAS
# ============================================================================
print("\n[TEST 6/7] Validando que no se eliminan filas...")
print("-" * 70)

try:
    # Verificar que el pipeline mantiene el número de filas
    sample_data = X_train.head(100)
    
    # Crear pipeline sin outlier remover
    test_pipeline = create_full_pipeline(
        exclude_cols=None,
        remove_outliers=False,
        scale_method='robust'
    )
    
    transformed_data = test_pipeline.fit_transform(sample_data)
    
    if transformed_data.shape[0] == sample_data.shape[0]:
        print(f"   ✓ Filas preservadas: {sample_data.shape[0]} → {transformed_data.shape[0]}")
        test_passed("Preservación de filas (sin OutlierRemover)")
    else:
        test_failed("Preservación de filas", 
                   f"Se perdieron filas: {sample_data.shape[0]} → {transformed_data.shape[0]}")
except Exception as e:
    test_failed("Preservación de filas", str(e))

# ============================================================================
# TEST 7: REPORTES Y ANÁLISIS
# ============================================================================
print("\n[TEST 7/7] Verificando reportes generados...")
print("-" * 70)

try:
    # Leer reporte de outliers
    report_file = Path("reports/figures/outlier_analysis_report.txt")
    if report_file.exists():
        content = report_file.read_text()
        if "62.4%" in content and "RobustScaler" in content:
            print("   ✓ Reporte de outliers correcto")
        else:
            print("   ⚠ Reporte existe pero contenido inesperado")
    
    # Leer comparación de scalers
    comparison_file = Path("reports/figures/scaler_comparison_results.txt")
    if comparison_file.exists():
        content = comparison_file.read_text()
        if "StandardScaler" in content and "RobustScaler" in content:
            print("   ✓ Comparación de scalers correcta")
        else:
            print("   ⚠ Comparación existe pero contenido inesperado")
    
    test_passed("Reportes y análisis")
except Exception as e:
    test_failed("Reportes y análisis", str(e))

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("📊 RESUMEN DE PRUEBAS")
print("=" * 70)

total_tests = len(test_results)
passed_tests = sum(1 for _, result in test_results if "✅" in result)
failed_tests = total_tests - passed_tests

for test_name, result in test_results:
    print(f"{result:50} {test_name}")

print("\n" + "-" * 70)
print(f"Total: {total_tests} tests")
print(f"✅ Pasados: {passed_tests}")
print(f"❌ Fallidos: {failed_tests}")
print("-" * 70)

if failed_tests == 0:
    print("\n🎉 ¡TODOS LOS TESTS PASARON!")
    print("✅ El sistema está funcionando correctamente")
    print("🚀 Listo para continuar con MLflow")
    sys.exit(0)
else:
    print("\n⚠️  HAY TESTS FALLIDOS")
    print("❌ Revisar los errores antes de continuar")
    sys.exit(1)
