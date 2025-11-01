#!/usr/bin/env python3
"""
Master Script: Complete Angry Classification Analysis
======================================================
Ejecuta todos los análisis en secuencia para diagnosticar
el problema de clasificación de la clase "Angry".

Usage:
    python3 run_complete_analysis.py
    
    O ejecutar scripts individuales:
    python3 analyze_1_dataset_distribution.py
    python3 analyze_2_confusion_matrix.py
    python3 analyze_3_sample_audio_features.py
    python3 analyze_4_feature_importance.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def print_header(title):
    """Imprime header decorado."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_script(script_name, description):
    """Ejecuta un script y captura resultados."""
    print_header(f"EJECUTANDO: {description}")
    print(f"📄 Script: {script_name}")
    print(f"⏰ Inicio: {datetime.now().strftime('%H:%M:%S')}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,  # Mostrar output en tiempo real
            text=True,
            check=True
        )
        
        print(f"\n✅ {script_name} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR en {script_name}")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró {script_name}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR inesperado: {e}")
        return False


def main():
    """Ejecuta análisis completo."""
    
    print_header("🚀 ANÁLISIS COMPLETO: PROBLEMA DE CLASIFICACIÓN 'ANGRY'")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {Path.cwd()}")
    
    # Verificar que estamos en el directorio correcto
    if not Path("data/processed/turkish_music_emotion_v2_cleaned_full.csv").exists():
        print("\n❌ ERROR: No se encontró el dataset")
        print("Asegúrate de estar en el directorio raíz del proyecto:")
        print("  cd /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24")
        sys.exit(1)
    
    # Lista de análisis a ejecutar
    analyses = [
        ("analyze_1_dataset_distribution.py", "Análisis de Distribución de Dataset"),
        ("analyze_2_confusion_matrix.py", "Análisis de Confusion Matrix"),
        ("analyze_3_sample_audio_features.py", "Análisis de Sample Audio Features"),
        ("analyze_4_feature_importance.py", "Análisis de Feature Importance"),
    ]
    
    results = {}
    start_time = datetime.now()
    
    # Ejecutar cada análisis
    for script, description in analyses:
        success = run_script(script, description)
        results[script] = "✅ SUCCESS" if success else "❌ FAILED"
        
        if not success:
            print(f"\n⚠️  El script {script} falló. ¿Continuar con el resto? (y/n): ", end="")
            response = input().strip().lower()
            if response != 'y':
                print("Análisis interrumpido por el usuario.")
                break
    
    # Resumen final
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_header("📊 RESUMEN DE ANÁLISIS")
    print(f"Tiempo total: {duration}")
    print(f"\nResultados:")
    for script, result in results.items():
        print(f"  {result} {script}")
    
    # Archivos generados
    print("\n📁 Archivos generados:")
    output_files = [
        "confusion_matrix_angry_analysis.png",
        "sample_audio_features_analysis.csv",
        "feature_importance_analysis.csv"
    ]
    
    for file in output_files:
        if Path(file).exists():
            size = Path(file).stat().st_size / 1024  # KB
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ⚠️  {file} (no generado)")
    
    # Recomendaciones finales
    print_header("🎯 PRÓXIMOS PASOS")
    print("""
1. Revisar los resultados de cada análisis en orden:
   - Dataset distribution: ¿Hay desbalance en angry?
   - Confusion matrix: ¿Con qué se confunde angry?
   - Sample audio: ¿Los audios de prueba son representativos?
   - Feature importance: ¿Qué features discriminan angry?

2. Basado en los hallazgos, considerar:
   - Re-etiquetar samples si hay inconsistencias
   - Balancear dataset si hay desproporción severa
   - Ajustar hyperparámetros del modelo
   - Agregar más features discriminativas
   - Mejorar preprocessing de audio

3. Documentar hallazgos y crear plan de acción para Phase 3

4. Compartir resultados con el equipo (David y Javier)
    """)
    
    success_count = sum(1 for r in results.values() if "SUCCESS" in r)
    total_count = len(results)
    
    if success_count == total_count:
        print("🎉 ¡Todos los análisis completados exitosamente!")
        return 0
    else:
        print(f"⚠️  {total_count - success_count} análisis fallaron. Revisa los errores arriba.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
