#!/usr/bin/env python3
"""
Limpieza Inteligente - Basada en Contexto del Equipo

Decisiones:
✅ app/ - MANTENER (FastAPI serving)
✅ dashboard/ - MOVER a monitoring/
🗑️ anaconda_projects/ - BORRAR (auto-generado)
🗑️ artifacts/ - BORRAR (no usado)
"""

import shutil
from pathlib import Path

print("=" * 70)
print("🧹 LIMPIEZA INTELIGENTE - MLOPS TEAM 24")
print("=" * 70)

changes = []

# 1. ACTUALIZAR .GITIGNORE
print("\n[1/6] Actualizando .gitignore...")

gitignore_additions = """
# MLflow (experimentos no se versionan)
mlruns/
mlartifacts/
.mlflow/

# Python compiled
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
.Python
build/
dist/

# DVC cache (no versionar cache local)
dvcstore/

# Temporales
scripts/temp/
*.tmp
*.log
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*checkpoint.ipynb

# Environment
.env
config.env
*.env

# IDE
.vscode/
.idea/

# Anaconda auto-generated
anaconda_projects/
"""

gitignore_path = Path(".gitignore")
current = gitignore_path.read_text() if gitignore_path.exists() else ""

lines_to_add = []
for line in gitignore_additions.strip().split('\n'):
    if line and not line.startswith('#') and line not in current:
        lines_to_add.append(line)

if lines_to_add:
    with open(gitignore_path, 'a') as f:
        f.write('\n\n# === Limpieza MLOps Team 24 ===\n')
        f.write(gitignore_additions)
    print(f"   ✓ Agregadas {len(lines_to_add)} entradas")
    changes.append(f".gitignore actualizado ({len(lines_to_add)} nuevas reglas)")

# 2. BORRAR CARPETAS NO NECESARIAS
print("\n[2/6] Eliminando carpetas innecesarias...")

to_delete_folders = [
    ("anaconda_projects", "Auto-generado por Anaconda"),
    ("artifacts", "Solo sample_preview.csv no usado"),
]

for folder, reason in to_delete_folders:
    path = Path(folder)
    if path.exists():
        shutil.rmtree(path)
        print(f"   ✓ Eliminado {folder}/ - {reason}")
        changes.append(f"Eliminado {folder}/")

# 3. REORGANIZAR DASHBOARD
print("\n[3/6] Reorganizando dashboard...")

if Path("dashboard").exists():
    # Crear estructura monitoring
    monitoring_path = Path("monitoring/dashboard")
    monitoring_path.mkdir(parents=True, exist_ok=True)
    
    # Mover contenido
    for item in Path("dashboard").iterdir():
        dst = monitoring_path / item.name
        shutil.move(str(item), str(dst))
        print(f"   ✓ dashboard/{item.name} → monitoring/dashboard/{item.name}")
    
    # Eliminar dashboard/ vacío
    Path("dashboard").rmdir()
    
    # Crear README en monitoring/
    readme = Path("monitoring/README.md")
    readme.write_text("""# Monitoring y Dashboards

Herramientas de monitoreo y visualización del proyecto.

## Dashboard Cookiecutter
`dashboard/` - Streamlit app para validar estructura Cookiecutter

### Ejecutar Dashboard
```bash
cd monitoring/dashboard
streamlit run streamlit_dashboard.py
```
""")
    
    changes.append("Dashboard → monitoring/dashboard/")

# 4. ELIMINAR ARCHIVOS TEMPORALES
print("\n[4/6] Eliminando archivos temporales...")

temp_files = [
    "reorganize_structure.py",
    "audit_structure.py", 
    "reorganize_project.sh",
    "ejecuta"
]

for f in temp_files:
    path = Path(f)
    if path.exists():
        path.unlink()
        print(f"   ✓ {f}")
        changes.append(f"Eliminado {f}")

# 5. REORGANIZAR SCRIPTS MAL UBICADOS
print("\n[5/6] Reubicando scripts...")

moves = {
    "scripts/test_dataset_equivalence.py": "tests/test_dataset_equivalence.py",
    "scripts/train_baseline.py": "scripts/training/train_baseline.py",
    "notebooks/ml_pipeline.py": "scripts/pipelines/ml_pipeline.py",
    "notebooks/test_ml_pipeline.py": "tests/test_ml_pipeline.py",
}

for src, dst in moves.items():
    src_path = Path(src)
    if src_path.exists():
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        print(f"   ✓ {src} → {dst}")
        changes.append(f"Movido {src}")

# 6. CONSOLIDAR METRICS
print("\n[6/6] Consolidando metrics/ en reports/...")

if Path("metrics").exists():
    for f in Path("metrics").glob("*"):
        if f.is_file():
            dst = Path("reports") / f.name
            shutil.move(str(f), str(dst))
            print(f"   ✓ {f.name} → reports/")
    
    try:
        Path("metrics").rmdir()
        changes.append("Consolidado metrics/ en reports/")
    except:
        pass

# RESUMEN
print("\n" + "=" * 70)
print("✅ LIMPIEZA COMPLETADA")
print("=" * 70)

print(f"\n📊 Total de cambios: {len(changes)}")
for i, change in enumerate(changes, 1):
    print(f"  {i}. {change}")

print("\n📁 Nueva estructura (destacados):")
print("""
MLOps_Team24/
├── acoustic_ml/          # Módulo Python principal
├── app/                  # ✅ FastAPI API (serving/deployment)
├── monitoring/           # 🆕 Dashboards y monitoreo
│   └── dashboard/        # Streamlit Cookiecutter tracker
├── scripts/
│   ├── analysis/
│   ├── training/         # 🆕 Scripts de entrenamiento
│   ├── pipelines/        # 🆕 ML pipelines
│   └── validation/
├── tests/                # Tests consolidados
├── data/                 # Versionado con DVC
├── models/               # Modelos versionados
└── reports/              # Reportes y metrics
""")

print("\n🎯 PASOS SIGUIENTES:")
print("\n1. Limpiar archivos trackeados en Git:")
print("   git rm -r --cached mlruns mlartifacts dvcstore __pycache__ 2>/dev/null")
print("   find . -name '*.pyc' -delete")
print("   find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null")

print("\n2. Verificar cambios:")
print("   git status")

print("\n3. Commit:")
print("   git add -A")
print('   git commit -m "chore: Limpieza profesional estructura MLOps')
print('')
print('   - Eliminado: anaconda_projects/, artifacts/')
print('   - Reorganizado: dashboard/ → monitoring/dashboard/')
print('   - Consolidado: metrics/ → reports/')
print('   - Actualizado .gitignore (mlruns, __pycache__, etc.)')
print('   - Reubicados scripts a carpetas correctas')
print('   - Estructura lista para producción"')

print("\n4. Push:")
print("   git push origin main")

print("\n💡 NOTAS:")
print("   • app/ - FastAPI para serving (mantener)")
print("   • monitoring/dashboard/ - Streamlit app (ahora organizado)")
print("   • mlruns/ - Ahora en .gitignore (experimentos locales)")
print("   • dvcstore/ - Ahora en .gitignore (cache DVC local)")
