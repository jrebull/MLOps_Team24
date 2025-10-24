#!/usr/bin/env python3
"""
Script para hispanizar docstrings y comentarios en módulos de modeling/
Mantiene el código (nombres de clases, métodos, variables) en inglés.
"""
import re
from pathlib import Path

# Diccionario de traducciones comunes
TRADUCCIONES = {
    # Docstrings principales
    "Model training con SOLID principles y Design Patterns.": 
        "Entrenamiento de modelos con principios SOLID y patrones de diseño.",
    
    "Model inference module with OOP design and robust error handling.":
        "Módulo de inferencia de modelos con diseño OOP y manejo robusto de errores.",
    
    "Model evaluation module with comprehensive metrics and MLflow integration.":
        "Módulo de evaluación de modelos con métricas completas e integración con MLflow.",
    
    # Clases y métodos
    "Handles model loading and predictions with validation and error handling.":
        "Maneja la carga de modelos y predicciones con validación y manejo de errores.",
    
    "Evaluates trained models with comprehensive metrics and visualizations.":
        "Evalúa modelos entrenados con métricas completas y visualizaciones.",
    
    "Initialize predictor with model path.":
        "Inicializar predictor con ruta del modelo.",
    
    "Initialize evaluator with model and report directory.":
        "Inicializar evaluador con modelo y directorio de reportes.",
    
    "Load trained model from disk.":
        "Cargar modelo entrenado desde disco.",
    
    "Make predictions on input features.":
        "Realizar predicciones sobre features de entrada.",
    
    "Make prediction for a single sample.":
        "Realizar predicción para una sola muestra.",
    
    "Make predictions in batches (useful for large datasets).":
        "Realizar predicciones en lotes (útil para datasets grandes).",
    
    "Evaluate model on test data.":
        "Evaluar modelo con datos de prueba.",
    
    "Generate detailed classification report.":
        "Generar reporte de clasificación detallado.",
    
    "Plot and optionally save confusion matrix.":
        "Graficar y opcionalmente guardar matriz de confusión.",
    
    "Save computed metrics to JSON file.":
        "Guardar métricas computadas en archivo JSON.",
    
    "Log metrics to MLflow.":
        "Registrar métricas en MLflow.",
    
    "Generate complete evaluation report with all metrics and visualizations.":
        "Generar reporte completo de evaluación con todas las métricas y visualizaciones.",
    
    "Quick model evaluation with all metrics and visualizations.":
        "Evaluación rápida de modelo con todas las métricas y visualizaciones.",
    
    # Args y Returns
    "Args:": "Argumentos:",
    "Returns:": "Retorna:",
    "Raises:": "Lanza:",
    "Attributes:": "Atributos:",
    
    # Comentarios comunes
    "Configure logging": "Configurar logging",
    "Convenience functions for backward compatibility": "Funciones de conveniencia para retrocompatibilidad",
    "Load a trained model (legacy interface).": "Cargar un modelo entrenado (interfaz legacy).",
    "Make predictions with a model (legacy interface).": "Realizar predicciones con un modelo (interfaz legacy).",
    "Convenience function for quick evaluation": "Función de conveniencia para evaluación rápida",
    
    # Strings específicos
    "Model not found at": "Modelo no encontrado en",
    "Please train a model first.": "Por favor entrene un modelo primero.",
    "Model not loaded. Call load_model() first.": "Modelo no cargado. Llame load_model() primero.",
    "Input must be a pandas DataFrame": "La entrada debe ser un pandas DataFrame",
    "Input DataFrame is empty": "El DataFrame de entrada está vacío",
    "Features must be dict or pd.Series": "Las features deben ser dict o pd.Series",
    "Model loaded successfully from": "Modelo cargado exitosamente desde",
    "Failed to load model:": "Falló al cargar modelo:",
    "Generated probability predictions for": "Generadas predicciones de probabilidad para",
    "Generated predictions for": "Generadas predicciones para",
    "samples": "muestras",
    "Prediction failed:": "Predicción falló:",
    "Processed batch": "Lote procesado",
    
    # Evaluate.py específico
    "Test data cannot be empty": "Los datos de prueba no pueden estar vacíos",
    "must have same length": "deben tener la misma longitud",
    "Starting evaluation on": "Iniciando evaluación en",
    "test samples": "muestras de prueba",
    "Evaluation complete - Accuracy:": "Evaluación completa - Accuracy:",
    "Classification report saved to": "Reporte de clasificación guardado en",
    "Confusion matrix saved to": "Matriz de confusión guardada en",
    "Metrics saved to": "Métricas guardadas en",
    "No metrics to save. Run evaluate() first.": "No hay métricas para guardar. Ejecute evaluate() primero.",
    "No metrics to log. Run evaluate() first.": "No hay métricas para registrar. Ejecute evaluate() primero.",
    "Metrics logged to MLflow": "Métricas registradas en MLflow",
    "MLflow not available. Skipping MLflow logging.": "MLflow no disponible. Omitiendo registro en MLflow.",
    "Failed to log to MLflow:": "Falló al registrar en MLflow:",
    "Generating full evaluation report...": "Generando reporte completo de evaluación...",
    "Full evaluation report generated": "Reporte completo de evaluación generado",
    
    # Train.py específico
    "Configuracion de un modelo.": "Configuración de un modelo.",
    "Configuracion de entrenamiento.": "Configuración de entrenamiento.",
    "Interface para entrenadores.": "Interfaz para entrenadores.",
    "Trainer base con Template Method Pattern.": "Entrenador base con patrón Template Method.",
    "Modelo no entrenado": "Modelo no entrenado",
    "Factory function para entrenar modelo baseline.": "Función factory para entrenar modelo baseline.",
}

def hispanizar_archivo(filepath: Path, traducciones: dict):
    """Hispaniza docstrings y comentarios de un archivo Python"""
    print(f"\n🔄 Procesando: {filepath.name}")
    
    # Leer contenido
    with open(filepath, 'r', encoding='utf-8') as f:
        contenido = f.read()
    
    # Backup
    backup_path = filepath.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(contenido)
    print(f"   ✅ Backup creado: {backup_path.name}")
    
    # Aplicar traducciones
    contenido_nuevo = contenido
    traducciones_aplicadas = 0
    
    for ingles, espanol in traducciones.items():
        if ingles in contenido_nuevo:
            contenido_nuevo = contenido_nuevo.replace(ingles, espanol)
            traducciones_aplicadas += 1
    
    # Guardar archivo actualizado
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(contenido_nuevo)
    
    print(f"   ✅ {traducciones_aplicadas} traducciones aplicadas")
    return traducciones_aplicadas

def main():
    """Ejecutar hispanización en los archivos de modeling/"""
    print("=" * 60)
    print("🇲🇽 HISPANIZACIÓN DE MÓDULOS MODELING")
    print("=" * 60)
    
    archivos = [
        Path("acoustic_ml/modeling/train.py"),
        Path("acoustic_ml/modeling/predict.py"),
        Path("acoustic_ml/modeling/evaluate.py"),
    ]
    
    total_traducciones = 0
    
    for archivo in archivos:
        if not archivo.exists():
            print(f"⚠️  Archivo no encontrado: {archivo}")
            continue
        
        traducciones = hispanizar_archivo(archivo, TRADUCCIONES)
        total_traducciones += traducciones
    
    print("\n" + "=" * 60)
    print(f"✅ HISPANIZACIÓN COMPLETADA")
    print(f"   Total de traducciones: {total_traducciones}")
    print("=" * 60)
    print("\n💡 Backups creados con extensión .backup")
    print("💡 Si algo sale mal, puedes restaurar desde los backups")

if __name__ == "__main__":
    main()
