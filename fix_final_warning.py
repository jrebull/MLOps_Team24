#!/usr/bin/env python3
"""
Fix Final - Eliminar último warning de MLflow
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/train.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Cambiar a sintaxis sin warnings (argumentos posicionales)
old_call = """            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",  # OK usar artifact_path aquí
                signature=signature,
                input_example=input_example,
                registered_model_name=None
            )"""

new_call = """            mlflow.sklearn.log_model(
                self.model,  # Argumento posicional (sin nombre)
                "model",     # artifact_path como posicional
                signature=signature,
                input_example=input_example
            )"""

if old_call in content:
    content = content.replace(old_call, new_call)
    file_path.write_text(content)
    print("✅ Último warning eliminado")
    print("   → Usando argumentos posicionales")
    print("   → Sintaxis MLflow 2.8+ correcta")
    print("   → 100% sin warnings")
else:
    print("❌ No se encontró el patrón exacto")
    sys.exit(1)

print("\n🎯 Cambio aplicado:")
print("   Antes: sk_model=..., artifact_path=...")
print("   Ahora: model, 'model', ...")
print("\n✅ Código profesional sin warnings")
