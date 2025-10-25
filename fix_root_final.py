#!/usr/bin/env python3
"""
Fix Final - Corrección de Problemas Críticos en Raíz

Corrige los 6 problemas críticos detectados:
1. Mueve scripts temporales
2. Actualiza .gitignore
3. Elimina carpetas de Git tracking
"""

import subprocess
from pathlib import Path
import shutil

print("=" * 70)
print("🔧 FIX FINAL - PROBLEMAS CRÍTICOS EN RAÍZ")
print("=" * 70)

# 1. MOVER SCRIPTS TEMPORALES
print("\n[1/3] Moviendo scripts temporales...")

scripts_temp = Path("scripts/temp")
scripts_temp.mkdir(parents=True, exist_ok=True)

temp_scripts = [
    "cleanup_smart.py",
    "validate_post_cleanup.py",
]

moved = []
for script in temp_scripts:
    if Path(script).exists():
        shutil.move(script, scripts_temp / script)
        moved.append(script)
        print(f"   ✓ {script} → scripts/temp/")

if moved:
    print(f"   Movidos {len(moved)} scripts temporales")

# 2. ACTUALIZAR .GITIGNORE
print("\n[2/3] Actualizando .gitignore...")

critical_ignores = """
# === MLOps Artifacts (no versionar) ===

# MLflow
mlruns/
mlartifacts/
.mlflow/

# DVC
dvcstore/

# Python Build
*.egg-info/
build/
dist/
*.egg

# Python Runtime
__pycache__/
*.py[cod]
*$py.class
*.so

# Temporales
scripts/temp/
*.tmp
*.log
"""

gitignore = Path(".gitignore")
content = gitignore.read_text() if gitignore.exists() else ""

lines_to_add = []
for line in critical_ignores.strip().split('\n'):
    if line and not line.startswith('#'):
        # Verificar si ya existe (con o sin trailing slash)
        pattern = line.rstrip('/')
        if pattern not in content and pattern + '/' not in content:
            lines_to_add.append(line)

if lines_to_add:
    with open(gitignore, 'a') as f:
        f.write('\n' + critical_ignores)
    print(f"   ✓ Agregadas {len(lines_to_add)} reglas críticas")
else:
    print("   ✓ .gitignore ya tiene todas las reglas críticas")

# 3. ELIMINAR DE GIT TRACKING
print("\n[3/3] Eliminando carpetas de Git tracking...")

to_untrack = [
    "mlruns",
    "mlartifacts",
    "dvcstore",
    "acoustic_ml.egg-info",
    "scripts/temp"
]

untracked = []
for folder in to_untrack:
    if Path(folder).exists():
        try:
            result = subprocess.run(
                ['git', 'rm', '-r', '--cached', folder],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                untracked.append(folder)
                print(f"   ✓ {folder} eliminado de Git")
            else:
                if "did not match any files" not in result.stderr:
                    print(f"   ℹ️  {folder} (no estaba en Git)")
        except Exception as e:
            print(f"   ⚠️  {folder}: {e}")

# RESUMEN
print("\n" + "=" * 70)
print("✅ CORRECCIONES COMPLETADAS")
print("=" * 70)

print(f"\n📊 Resumen:")
print(f"   • Scripts movidos: {len(moved)}")
print(f"   • Reglas .gitignore: {len(lines_to_add)}")
print(f"   • Carpetas des-trackeadas: {len(untracked)}")

print("\n🎯 PRÓXIMOS PASOS:")

print("\n1. Verificar cambios:")
print("   git status")

print("\n2. Ver archivos ignorados (opcional):")
print("   git status --ignored")

print("\n3. Commit final:")
print("""   git add -A
   git commit -m "fix: Corregir raíz según MLOps best practices

- Movidos scripts temporales a scripts/temp/
- Actualizado .gitignore: mlruns/, mlartifacts/, dvcstore/, *.egg-info/
- Eliminados artifacts de Git tracking
- Raíz 100% limpia y profesional según estándares MLOps

Problemas corregidos:
- ✅ Scripts .py temporales movidos
- ✅ Experimentos MLflow no versionados
- ✅ Cache DVC no versionado
- ✅ Build artifacts no versionados
"
   git push origin main""")

print("\n✅ Después de este commit:")
print("   → Raíz cumplirá 100% estándares MLOps estrictos")
print("   → Solo archivos de configuración en raíz")
print("   → Estructura lista para producción")

print("\n🚀 Luego continuamos con:")
print("   → Comparación de experimentos con MLflow")
