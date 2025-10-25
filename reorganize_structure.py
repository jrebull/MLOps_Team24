#!/usr/bin/env python3
"""
Reorganización Profesional - MLOps Best Practices

Reorganiza la estructura del proyecto siguiendo estándares MLOps:
- Raíz limpia (solo configs y docs)
- Scripts organizados por propósito
- Tests en carpeta dedicada
"""

import shutil
from pathlib import Path
import sys

print("=" * 70)
print("🏗️  REORGANIZACIÓN PROFESIONAL DE ESTRUCTURA")
print("=" * 70)

# 1. Crear estructura de carpetas
print("\n[1/4] Creando estructura de carpetas...")

folders = {
    "scripts/analysis": "Scripts de análisis y experimentación",
    "scripts/validation": "Scripts de validación y verificación",
    "tests": "Tests automatizados y validaciones"
}

for folder, description in folders.items():
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ {folder}/ - {description}")

# 2. Mapeo de archivos
print("\n[2/4] Mapeando archivos a nuevas ubicaciones...")

file_mapping = {
    # Análisis
    "analyze_outliers.py": "scripts/analysis/",
    "compare_scalers.py": "scripts/analysis/",
    "run_full_analysis.py": "scripts/analysis/",
    
    # Validación
    "verify_sync.py": "scripts/validation/",
    
    # Tests
    "test_sklearn_pipeline.py": "tests/",
    "test_full_integration.py": "tests/",
    "validate_cookiecutter.py": "tests/",
    "validate_dataset.py": "tests/",
    "validate_features.py": "tests/",
    "validate_plots.py": "tests/",
}

# 3. Mover archivos
print("\n[3/4] Moviendo archivos...")

moved = []
not_found = []

for source, destination in file_mapping.items():
    src = Path(source)
    if src.exists():
        dest = Path(destination) / source
        shutil.move(str(src), str(dest))
        moved.append(f"{source} → {destination}")
        print(f"   ✓ {source} → {destination}")
    else:
        not_found.append(source)
        print(f"   ⚠️  {source} (ya no existe en raíz)")

# 4. Crear __init__.py en carpetas para imports
print("\n[4/4] Configurando estructura Python...")

for folder in ["scripts/analysis", "scripts/validation"]:
    init_file = Path(folder) / "__init__.py"
    init_file.write_text('"""Scripts de utilidad para el proyecto."""\n')
    print(f"   ✓ {folder}/__init__.py")

# Crear README en cada carpeta
readmes = {
    "scripts/analysis/README.md": """# Scripts de Análisis

Scripts para análisis estadístico y comparación de métodos:

- `analyze_outliers.py`: Análisis de outliers con IQR
- `compare_scalers.py`: Comparación StandardScaler vs RobustScaler
- `run_full_analysis.py`: Script maestro que ejecuta análisis completo
""",
    "scripts/validation/README.md": """# Scripts de Validación

Scripts para validar el estado del sistema:

- `verify_sync.py`: Verificación pre-MLflow (Git, archivos, dependencies)
""",
    "tests/README.md": """# Tests

Tests automatizados y scripts de validación:

## Tests de Integración
- `test_sklearn_pipeline.py`: Test del pipeline sklearn
- `test_full_integration.py`: Validación completa del sistema (7 tests)

## Validaciones de Estructura
- `validate_cookiecutter.py`: Validación estructura Cookiecutter
- `validate_dataset.py`: Validación módulo dataset
- `validate_features.py`: Validación módulo features
- `validate_plots.py`: Validación módulo plots

## Ejecutar Tests
```bash
# Test individual
python tests/test_sklearn_pipeline.py

# Test completo
python tests/test_full_integration.py

# Validaciones
python tests/validate_cookiecutter.py
```
"""
}

for readme_path, content in readmes.items():
    Path(readme_path).write_text(content)
    print(f"   ✓ {readme_path}")

# Resumen
print("\n" + "=" * 70)
print("✅ REORGANIZACIÓN COMPLETADA")
print("=" * 70)

print("\n📁 Nueva estructura:")
print("""
MLOps_Team24/
├── acoustic_ml/          # Código productivo
├── scripts/
│   ├── analysis/         # Análisis y experimentación
│   │   ├── analyze_outliers.py
│   │   ├── compare_scalers.py
│   │   └── run_full_analysis.py
│   └── validation/       # Validación del sistema
│       └── verify_sync.py
├── tests/                # Tests automatizados
│   ├── test_sklearn_pipeline.py
│   ├── test_full_integration.py
│   └── validate_*.py (4 archivos)
├── data/
├── models/
├── notebooks/
├── reports/
└── README.md            # Raíz limpia
""")

print(f"\n📊 Archivos movidos: {len(moved)}")
print(f"⚠️  No encontrados: {len(not_found)}")

print("\n🎯 Próximo paso:")
print("   git add -A")
print('   git commit -m "refactor: Reorganización según MLOps best practices"')
print("   git push origin main")

print("\n💡 Cómo ejecutar scripts ahora:")
print("   # Análisis")
print("   python scripts/analysis/run_full_analysis.py")
print("   ")
print("   # Tests")
print("   python tests/test_full_integration.py")
print("   ")
print("   # Validación")
print("   python scripts/validation/verify_sync.py")
