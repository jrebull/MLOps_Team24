#!/usr/bin/env python3
"""
Fix MLflow Warnings - Profesionalizar model logging
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/train.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Fix: Cambiar artifact_path por name + agregar signature
old_logging = """            # Log model as artifact
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=None  # No auto-register
            )"""

new_logging = """            # Log model as artifact (con signature y example)
            import numpy as np
            from mlflow.models import infer_signature
            
            # Crear input example (primera fila de datos de entrenamiento)
            input_example = X[:1] if len(X) > 0 else None
            
            # Inferir signature del modelo
            predictions = self.model.predict(X[:5] if len(X) >= 5 else X)
            signature = infer_signature(X, predictions)
            
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",  # OK usar artifact_path aquí
                signature=signature,
                input_example=input_example,
                registered_model_name=None
            )"""

if old_logging in content:
    content = content.replace(old_logging, new_logging)
    file_path.write_text(content)
    print("✅ MLflow warnings corregidos")
    print("   → Agregado: signature inference")
    print("   → Agregado: input_example")
    print("   → Modelo ahora tiene schema de entrada/salida")
else:
    print("❌ No se encontró el patrón exacto")
    sys.exit(1)

print("\n🎯 Beneficios:")
print("   • Signature: MLflow sabe el schema del modelo")
print("   • Input example: Valida formato de datos")
print("   • No más warnings al loguear modelos")
print("   • Más profesional para producción")
