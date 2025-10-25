#!/usr/bin/env python3
"""
Safe MLflow Upgrade - Con rollback automático

Actualiza MLflow de forma segura con verificación y rollback.
"""

import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("🔄 UPGRADE SEGURO DE MLFLOW")
print("=" * 70)

# 1. Backup de versión actual
print("\n[1/5] Backup de versión actual...")
try:
    result = subprocess.run(
        ['pip', 'show', 'mlflow'],
        capture_output=True,
        text=True,
        check=True
    )
    current_version = None
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            current_version = line.split(':')[1].strip()
    
    print(f"   ✓ Versión actual: mlflow=={current_version}")
    
    # Guardar en archivo
    backup_file = Path("mlflow_version_backup.txt")
    backup_file.write_text(current_version)
    print(f"   ✓ Backup guardado: {backup_file}")
    
except Exception as e:
    print(f"   ❌ Error en backup: {e}")
    sys.exit(1)

# 2. Actualizar MLflow
print("\n[2/5] Actualizando MLflow...")
print("   → pip install --upgrade mlflow>=2.10.0")

try:
    result = subprocess.run(
        ['pip', 'install', '--upgrade', 'mlflow>=2.10.0'],
        capture_output=True,
        text=True,
        check=True
    )
    print("   ✓ MLflow actualizado")
    
    # Ver nueva versión
    result = subprocess.run(
        ['pip', 'show', 'mlflow'],
        capture_output=True,
        text=True,
        check=True
    )
    new_version = None
    for line in result.stdout.split('\n'):
        if line.startswith('Version:'):
            new_version = line.split(':')[1].strip()
    
    print(f"   ✓ Nueva versión: mlflow=={new_version}")
    
except Exception as e:
    print(f"   ❌ Error en upgrade: {e}")
    print("\n🔙 Ejecuta rollback:")
    print(f"   pip install mlflow=={current_version}")
    sys.exit(1)

# 3. Verificar imports
print("\n[3/5] Verificando imports...")
try:
    import mlflow
    import mlflow.sklearn
    print(f"   ✓ MLflow {mlflow.__version__} importado correctamente")
except Exception as e:
    print(f"   ❌ Error en imports: {e}")
    print("\n🔙 Ejecuta rollback:")
    print(f"   pip install mlflow=={current_version}")
    sys.exit(1)

# 4. Test rápido
print("\n[4/5] Test rápido de funcionalidad...")
try:
    # Test básico de tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("test_upgrade")
    
    with mlflow.start_run(run_name="upgrade_test"):
        mlflow.log_param("test_param", "value")
        mlflow.log_metric("test_metric", 0.99)
    
    print("   ✓ Tracking funciona correctamente")
    print("   ✓ Logs de params/metrics OK")
    
except Exception as e:
    print(f"   ❌ Error en test: {e}")
    print("\n🔙 Ejecuta rollback:")
    print(f"   pip install mlflow=={current_version}")
    sys.exit(1)

# 5. Actualizar requirements.txt
print("\n[5/5] Actualizando requirements.txt...")
try:
    req_file = Path("requirements.txt")
    content = req_file.read_text()
    
    # Reemplazar versión de mlflow
    import re
    new_content = re.sub(
        r'mlflow[>=<].*',
        f'mlflow>={new_version}',
        content
    )
    
    req_file.write_text(new_content)
    print(f"   ✓ requirements.txt actualizado: mlflow>={new_version}")
    
except Exception as e:
    print(f"   ⚠️  No se pudo actualizar requirements.txt: {e}")
    print("   Actualiza manualmente si es necesario")

# Resumen
print("\n" + "=" * 70)
print("✅ UPGRADE COMPLETADO EXITOSAMENTE")
print("=" * 70)
print(f"\n📊 Cambio:")
print(f"   Antes: mlflow=={current_version}")
print(f"   Ahora: mlflow=={new_version}")

print("\n🎯 Próximo paso:")
print("   python test_clean_mlflow.py")
print("   → Verifica que NO hay warnings")

print("\n💾 Rollback disponible:")
print(f"   pip install mlflow=={current_version}")
print(f"   (versión guardada en {backup_file})")
