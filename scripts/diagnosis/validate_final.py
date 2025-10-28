#!/usr/bin/env python3
"""
🔍 VALIDACIÓN FINAL - MLOps Team 24
Verifica que todo esté correcto antes de entrega
"""

import sys
import subprocess
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import warnings

# Suprimir warnings cosméticos
warnings.filterwarnings('ignore')

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_mark(condition, message):
    """Print check mark based on condition"""
    symbol = "✅" if condition else "❌"
    status = "OK" if condition else "FAIL"
    print(f"{symbol} [{status}] {message}")
    return condition

def main():
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║  🔍 VALIDACIÓN FINAL - MLOps Team 24                                ║")
    print("║  Turkish Music Emotion Recognition - Fase 2                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    all_checks_passed = True
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. ESTRUCTURA DEL PROYECTO
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("1️⃣  ESTRUCTURA DEL PROYECTO")
    
    required_files = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "acoustic_ml/__init__.py",
        "acoustic_ml/config.py",
        "acoustic_ml/dataset.py",
        "acoustic_ml/features.py",
        "acoustic_ml/modeling/sklearn_pipeline.py",
        "acoustic_ml/modeling/train.py",
        "scripts/training/run_mlflow_experiments.py",
        "data.dvc",
    ]
    
    for file in required_files:
        exists = Path(file).exists()
        all_checks_passed &= check_mark(exists, f"{file}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. MÓDULO ACOUSTIC_ML
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("2️⃣  MÓDULO ACOUSTIC_ML")
    
    try:
        from acoustic_ml.dataset import DatasetManager
        all_checks_passed &= check_mark(True, "Import DatasetManager")
    except Exception as e:
        all_checks_passed &= check_mark(False, f"Import DatasetManager: {e}")
    
    try:
        from acoustic_ml.features import create_full_pipeline
        all_checks_passed &= check_mark(True, "Import create_full_pipeline")
    except Exception as e:
        all_checks_passed &= check_mark(False, f"Import create_full_pipeline: {e}")
    
    try:
        from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
        all_checks_passed &= check_mark(True, "Import create_sklearn_pipeline")
    except Exception as e:
        all_checks_passed &= check_mark(False, f"Import create_sklearn_pipeline: {e}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. DATOS Y DVC
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("3️⃣  DATOS Y DVC")
    
    # Verificar data files
    data_files = [
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "data/processed/y_train.csv",
        "data/processed/y_test.csv",
    ]
    
    for file in data_files:
        exists = Path(file).exists()
        all_checks_passed &= check_mark(exists, f"{file}")
    
    # Verificar DVC status
    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            timeout=10
        )
        # DVC retorna "Data and pipelines are up to date." cuando está OK
        dvc_clean = (len(result.stdout.strip()) == 0 or 
                     "up to date" in result.stdout.lower())
        all_checks_passed &= check_mark(dvc_clean, "DVC status limpio")
        if not dvc_clean:
            print(f"   ⚠️  DVC output: {result.stdout[:100]}")
    except Exception as e:
        all_checks_passed &= check_mark(False, f"DVC status check: {e}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. PIPELINE SKLEARN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("4️⃣  PIPELINE SKLEARN")
    
    try:
        from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
        from acoustic_ml.dataset import DatasetManager
        
        # Cargar datos usando DatasetManager (mejor práctica MLOps)
        print("   📂 Cargando datos con DatasetManager...")
        dm = DatasetManager()
        X_train, X_test, y_train, y_test = dm.get_train_test_split()
        all_checks_passed &= check_mark(True, f"Datos cargados: {len(X_train)} train, {len(X_test)} test")
        
        # Crear pipeline
        print("   🔧 Creando pipeline...")
        pipeline = create_sklearn_pipeline(
            model_type="random_forest",
            model_params={"n_estimators": 10, "random_state": 42},
            scale_method="robust"
        )
        all_checks_passed &= check_mark(True, "Pipeline creado con scale_method='robust'")
        
        # Entrenar
        print("   🎓 Entrenando pipeline...")
        pipeline.fit(X_train, y_train)
        all_checks_passed &= check_mark(True, "Pipeline entrenado")
        
        # Predecir
        print("   🔮 Haciendo predicciones...")
        predictions = pipeline.predict(X_test)
        all_checks_passed &= check_mark(len(predictions) == len(X_test), "Predicciones generadas")
        
        # Score
        print("   📊 Calculando accuracy...")
        score = pipeline.score(X_test, y_test)
        all_checks_passed &= check_mark(score > 0.5, f"Accuracy: {score:.4f}")
        
        print(f"   ✅ Pipeline funcional - Accuracy: {score:.4f}")
        
    except Exception as e:
        all_checks_passed &= check_mark(False, f"Pipeline test: {str(e)[:100]}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. MLFLOW EXPERIMENTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("5️⃣  MLFLOW EXPERIMENTS")
    
    try:
        # Configurar MLflow
        mlflow.set_tracking_uri("./mlruns")
        client = MlflowClient()
        
        # Buscar experimento
        experiment = mlflow.get_experiment_by_name("turkish-music-emotion-recognition")
        
        if experiment:
            all_checks_passed &= check_mark(True, f"Experimento encontrado: {experiment.name}")
            
            # Obtener runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=20
            )
            
            num_runs = len(runs)
            all_checks_passed &= check_mark(num_runs >= 7, f"Número de runs: {num_runs} (esperado: ≥7)")
            
            if num_runs > 0:
                print(f"\n   📊 Últimos {min(10, num_runs)} experimentos:")
                print("   " + "-"*66)
                
                for idx, run in runs.head(10).iterrows():
                    run_name = run.get('tags.mlflow.runName', 'N/A')
                    test_acc = run.get('metrics.test_accuracy', 0)
                    
                    # Verificar que tiene métricas
                    has_metrics = test_acc > 0
                    symbol = "✅" if has_metrics else "⚠️"
                    
                    print(f"   {symbol} {run_name[:35]:35s} | Acc: {test_acc:.4f}")
                
                # Verificar mejor modelo
                best_run = runs.loc[runs['metrics.test_accuracy'].idxmax()]
                best_name = best_run.get('tags.mlflow.runName', 'N/A')
                best_acc = best_run.get('metrics.test_accuracy', 0)
                
                print("\n   🏆 Mejor Modelo:")
                print(f"      Nombre: {best_name}")
                print(f"      Test Accuracy: {best_acc:.4f}")
                
                all_checks_passed &= check_mark(
                    best_acc >= 0.75, 
                    f"Mejor accuracy ≥ 75%: {best_acc:.2%}"
                )
                
                # Verificar artefactos del mejor modelo
                best_run_id = best_run['run_id']
                artifacts = client.list_artifacts(best_run_id)
                artifact_names = [a.path for a in artifacts]
                
                has_model = any('model' in name.lower() for name in artifact_names)
                has_artifacts = len(artifact_names) > 0
                
                # Si tiene artefactos pero no modelo específicamente, es warning no error
                if has_model:
                    check_mark(True, "Modelo guardado en artefactos")
                elif has_artifacts:
                    print(f"   ⚠️  [INFO] Artefactos encontrados: {', '.join(artifact_names[:3])}")
                    print(f"   ℹ️  No se encontró carpeta 'model' pero hay {len(artifact_names)} artefactos")
                else:
                    all_checks_passed &= check_mark(False, "No hay artefactos guardados")
                
                # Verificar que NO hay warnings (chequeando los logs)
                print("\n   🔍 Verificando warnings en MLflow...")
                # Esto es informativo, no afecta el check
                print("   ℹ️  Ejecuta experimentos y verifica que no haya warnings en consola")
                
        else:
            all_checks_passed &= check_mark(False, "Experimento 'turkish-music-emotion-recognition' no encontrado")
            print("   ⚠️  Ejecuta: python scripts/training/run_mlflow_experiments.py")
    
    except Exception as e:
        all_checks_passed &= check_mark(False, f"MLflow check: {str(e)[:100]}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. GIT Y GITHUB
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("6️⃣  GIT Y GITHUB")
    
    try:
        # Git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5
        )
        git_clean = len(result.stdout.strip()) == 0
        all_checks_passed &= check_mark(git_clean, "Git working tree limpio")
        if not git_clean:
            print("   ⚠️  Archivos sin commit:")
            for line in result.stdout.strip().split('\n')[:5]:
                print(f"      {line}")
        
        # Verificar remote
        result = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True,
            timeout=5
        )
        has_remote = "origin" in result.stdout
        all_checks_passed &= check_mark(has_remote, "Git remote configurado")
        
        # Últimos commits
        result = subprocess.run(
            ["git", "log", "--oneline", "-3"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("\n   📝 Últimos 3 commits:")
        for line in result.stdout.strip().split('\n'):
            print(f"      {line}")
        
    except Exception as e:
        all_checks_passed &= check_mark(False, f"Git check: {e}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. TESTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print_section("7️⃣  TESTS")
    
    test_scripts = [
        "scripts/validate_dataset.py",
        "scripts/validate_features.py",
        "scripts/validate_plots.py",
    ]
    
    for script in test_scripts:
        if Path(script).exists():
            try:
                result = subprocess.run(
                    ["python", script],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                passed = result.returncode == 0
                script_name = Path(script).name
                all_checks_passed &= check_mark(passed, f"{script_name}")
                if not passed and result.stderr:
                    print(f"   ⚠️  Error: {result.stderr[:100]}")
            except Exception as e:
                all_checks_passed &= check_mark(False, f"{script}: {e}")
        else:
            print(f"   ⏭️  {script} no encontrado")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RESUMEN FINAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print("\n" + "="*70)
    if all_checks_passed:
        print("  ✅ VALIDACIÓN COMPLETADA - TODO OK")
        print("  🎉 Proyecto listo para entrega")
    else:
        print("  ⚠️  VALIDACIÓN COMPLETADA - ALGUNOS CHECKS FALLARON")
        print("  🔧 Revisa los items marcados con ❌")
    print("="*70)
    
    print("\n🎯 Siguiente paso:")
    print("   → Abrir MLflow UI: mlflow ui --port 5001")
    print("   → URL: http://localhost:5001")
    print("   → Verificar visualmente que no hay warnings en los logs")
    print("\n")
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
