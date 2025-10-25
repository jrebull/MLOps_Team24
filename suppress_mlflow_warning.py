#!/usr/bin/env python3
"""
Suprimir Warning Cosmético de MLflow

El warning es solo informativo y no afecta funcionalidad.
El modelo se guarda correctamente con signature y example.
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/train.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Agregar supresión de warning al inicio del método
old_method_start = """    def _log_to_mlflow(self, X, y):
        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)"""

new_method_start = """    def _log_to_mlflow(self, X, y):
        import warnings
        # Suprimir warning deprecation cosmético de artifact_path
        warnings.filterwarnings('ignore', category=FutureWarning, module='mlflow')
        
        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)"""

if old_method_start in content:
    content = content.replace(old_method_start, new_method_start)
    file_path.write_text(content)
    print("✅ Warning suprimido")
    print("   → FutureWarning de MLflow filtrado")
    print("   → Funcionalidad intacta")
    print("   → Output 100% limpio")
else:
    print("❌ No se encontró el patrón")
    sys.exit(1)

print("\n📝 Nota técnica:")
print("   • El warning es solo informativo")
print("   • MLflow 2.8+ prefiere 'name' sobre 'artifact_path'")
print("   • Ambos funcionan igual (backward compatible)")
print("   • La supresión es práctica estándar en producción")
