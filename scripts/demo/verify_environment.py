#!/usr/bin/env python3
"""
✅ VERIFICACIÓN DEL AMBIENTE - PIPELINE SKLEARN
================================================

Verifica que todo esté configurado correctamente antes de ejecutar las demos.

Uso:
    python verify_environment.py
"""

import sys
from pathlib import Path

def print_status(check, message):
    """Imprime el estado de una verificación"""
    if check:
        print(f"✅ {message}")
        return True
    else:
        print(f"❌ {message}")
        return False


def main():
    print("\n" + "="*80)
    print("✅ VERIFICACIÓN DEL AMBIENTE - SKLEARN PIPELINE")
    print("="*80 + "\n")
    
    all_ok = True
    
    # 1. Verificar Python
    print("1️⃣  PYTHON")
    print("-" * 80)
    py_version = sys.version_info
    py_ok = py_version >= (3, 8)
    all_ok &= print_status(
        py_ok,
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro} (requiere ≥3.8)"
    )
    print()
    
    # 2. Verificar módulos
    print("2️⃣  MÓDULOS REQUERIDOS")
    print("-" * 80)
    
    modules_to_check = [
        ('acoustic_ml', 'Módulo del proyecto'),
        ('sklearn', 'Scikit-Learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib')
    ]
    
    for module_name, description in modules_to_check:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'N/A')
            all_ok &= print_status(True, f"{description:20s} - v{version}")
        except ImportError:
            all_ok &= print_status(False, f"{description:20s} - NO INSTALADO")
    print()
    
    # 3. Verificar estructura del proyecto
    print("3️⃣  ESTRUCTURA DEL PROYECTO")
    print("-" * 80)
    
    required_dirs = [
        ('data/processed', 'Directorio de datos procesados'),
        ('acoustic_ml', 'Módulo acoustic_ml'),
        ('acoustic_ml/modeling', 'Módulo de modelado'),
    ]
    
    for dir_path, description in required_dirs:
        exists = Path(dir_path).exists()
        all_ok &= print_status(exists, f"{description:35s} - {dir_path}")
    print()
    
    # 4. Verificar archivos de datos
    print("4️⃣  ARCHIVOS DE DATOS")
    print("-" * 80)
    
    data_dir = Path('data/processed')
    required_files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    
    for filename in required_files:
        file_path = data_dir / filename
        exists = file_path.exists()
        if exists:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            all_ok &= print_status(True, f"{filename:15s} - {size_mb:.2f} MB")
        else:
            all_ok &= print_status(False, f"{filename:15s} - NO ENCONTRADO")
    print()
    
    # 5. Verificar archivos de pipeline
    print("5️⃣  ARCHIVOS DEL PIPELINE")
    print("-" * 80)
    
    pipeline_files = [
        'acoustic_ml/modeling/sklearn_pipeline.py',
        'acoustic_ml/dataset.py',
        'acoustic_ml/features.py',
    ]
    
    for file_path in pipeline_files:
        exists = Path(file_path).exists()
        all_ok &= print_status(exists, file_path)
    print()
    
    # 6. Prueba de importación
    print("6️⃣  PRUEBA DE IMPORTACIÓN")
    print("-" * 80)
    
    try:
        from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
        all_ok &= print_status(True, "create_sklearn_pipeline importado correctamente")
    except ImportError as e:
        all_ok &= print_status(False, f"Error importando create_sklearn_pipeline: {e}")
    
    try:
        from acoustic_ml.dataset import DatasetManager
        all_ok &= print_status(True, "DatasetManager importado correctamente")
    except ImportError as e:
        all_ok &= print_status(False, f"Error importando DatasetManager: {e}")
    print()
    
    # 7. Prueba básica del pipeline
    print("7️⃣  PRUEBA BÁSICA DEL PIPELINE")
    print("-" * 80)
    
    try:
        from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
        pipeline = create_sklearn_pipeline(model_type="random_forest")
        all_ok &= print_status(True, "Pipeline creado exitosamente")
        
        # Verificar atributos
        has_attrs = all([
            hasattr(pipeline, 'fit'),
            hasattr(pipeline, 'predict'),
            hasattr(pipeline, 'score'),
        ])
        all_ok &= print_status(has_attrs, "Pipeline tiene métodos fit/predict/score")
        
    except Exception as e:
        all_ok &= print_status(False, f"Error creando pipeline: {e}")
    print()
    
    # Resumen final
    print("="*80)
    if all_ok:
        print("✅ AMBIENTE VERIFICADO - TODO OK")
        print("\n🚀 Puedes ejecutar:")
        print("   • python test_pipeline_quick.py")
        print("   • python validate_sklearn_pipeline_demo.py")
    else:
        print("❌ PROBLEMAS DETECTADOS")
        print("\n🔧 Soluciones:")
        print("   1. Asegúrate de estar en el directorio raíz del proyecto")
        print("   2. Activa el entorno virtual: source .venv/bin/activate")
        print("   3. Instala el módulo: pip install -e .")
        print("   4. Verifica que los datos existen en data/processed/")
    print("="*80 + "\n")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
