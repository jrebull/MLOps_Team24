#!/usr/bin/env python3
"""
Supresión Fuerte de Warning - Context Manager
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/train.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Buscar y reemplazar el bloque de log_model
old_block = """            mlflow.sklearn.log_model(
                self.model,  # Argumento posicional (sin nombre)
                "model",     # artifact_path como posicional
                signature=signature,
                input_example=input_example
            )"""

new_block = """            # Suprimir warning específicamente durante log_model
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*artifact_path.*')
                mlflow.sklearn.log_model(
                    self.model,
                    "model",
                    signature=signature,
                    input_example=input_example
                )"""

if old_block in content:
    content = content.replace(old_block, new_block)
    file_path.write_text(content)
    print("✅ Warning suprimido con context manager")
    print("   → warnings.catch_warnings() envuelve la llamada")
    print("   → Supresión quirúrgica solo en log_model")
else:
    print("❌ No se encontró el patrón")
    sys.exit(1)

print("\n🎯 Método más robusto:")
print("   Context manager captura warning en el punto exacto")
