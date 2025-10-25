#!/usr/bin/env python3
"""
Script para actualizar sklearn_pipeline.py con RobustScaler
basado en análisis de outliers
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/sklearn_pipeline.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Cambiar de 'standard' a 'robust'
old_line = """        # Pipeline completo SIN OutlierRemover (no elimina filas)
        self.feature_pipeline = create_full_pipeline(
            exclude_cols=None,
            remove_outliers=False,  # ⚠️ Crítico: False para no eliminar filas
            scale_method='standard'
        )"""

new_line = """        # Pipeline completo SIN OutlierRemover (no elimina filas)
        # Usando RobustScaler basado en análisis de outliers
        # Decisión: RobustScaler es más robusto a outliers (usa mediana e IQR)
        self.feature_pipeline = create_full_pipeline(
            exclude_cols=None,
            remove_outliers=False,  # ⚠️ Crítico: False para no eliminar filas
            scale_method='robust'  # ✅ RobustScaler para manejo de outliers
        )"""

if old_line in content:
    content = content.replace(old_line, new_line)
    file_path.write_text(content)
    print("✅ Pipeline actualizado: scale_method='robust'")
    print("   → RobustScaler usa mediana (Q2) e IQR")
    print("   → Menos sensible a valores extremos")
    print("   → Best practice MLOps para datos con outliers")
else:
    print("⚠️  No se encontró el patrón exacto")
    print("   Buscando alternativas...")
    
    # Intentar solo reemplazar el parámetro
    if "scale_method='standard'" in content:
        content = content.replace("scale_method='standard'", "scale_method='robust'")
        file_path.write_text(content)
        print("✅ Pipeline actualizado: scale_method='robust'")
    else:
        print("❌ No se pudo actualizar automáticamente")
        print("   Por favor, cambia manualmente: scale_method='robust'")
        sys.exit(1)

print("\n📊 Diferencias entre scalers:")
print("   StandardScaler:")
print("     - Usa media y desviación estándar")
print("     - Sensible a outliers extremos")
print("     - X_scaled = (X - mean) / std")
print("\n   RobustScaler:")
print("     - Usa mediana (Q2) e IQR (Q3-Q1)")
print("     - Robusto a outliers")
print("     - X_scaled = (X - median) / IQR")
