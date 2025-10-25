#!/usr/bin/env python3
"""
Master Script: Análisis Completo de Outliers y Optimización

Ejecuta todo el proceso de análisis y optimización en orden:
1. Análisis de outliers
2. Comparación de scalers
3. Actualización del pipeline
4. Validación final

Autor: MLOps Team 24
"""

import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("🚀 PROCESO COMPLETO: OPTIMIZACIÓN DE MANEJO DE OUTLIERS")
print("=" * 70)

scripts_to_run = [
    ("analyze_outliers.py", "Análisis de Outliers"),
    ("compare_scalers.py", "Comparación StandardScaler vs RobustScaler"),
    ("update_to_robust_scaler.py", "Actualización de Pipeline"),
    ("test_sklearn_pipeline.py", "Validación Final")
]

print("\n📋 Plan de ejecución:")
for i, (script, description) in enumerate(scripts_to_run, 1):
    print(f"   {i}. {description} ({script})")

print("\n¿Deseas continuar? (Enter para sí, Ctrl+C para cancelar)")
input()

for i, (script, description) in enumerate(scripts_to_run, 1):
    print("\n" + "=" * 70)
    print(f"[{i}/{len(scripts_to_run)}] {description}")
    print("=" * 70)
    
    if not Path(script).exists():
        print(f"⚠️  Script {script} no encontrado - saltando")
        continue
    
    try:
        result = subprocess.run(['python', script], check=True)
        print(f"✅ {description} completado")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"   ¿Continuar con siguiente paso? (Enter para sí, Ctrl+C para cancelar)")
        input()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(0)
    
    if i < len(scripts_to_run):
        print("\nPresiona Enter para continuar al siguiente paso...")
        input()

print("\n" + "=" * 70)
print("🎉 PROCESO COMPLETADO")
print("=" * 70)
print("\n📊 Archivos generados:")
print("   - reports/figures/outlier_analysis.png")
print("   - reports/figures/outlier_boxplots.png")
print("   - reports/figures/outlier_analysis_report.txt")
print("   - reports/figures/scaler_comparison_results.txt")
print("\n✅ Pipeline optimizado y listo para producción")
