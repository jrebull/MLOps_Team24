#!/usr/bin/env python3
"""
🧪 PRUEBA RÁPIDA - PIPELINE SCIKIT-LEARN
==========================================

Script minimalista para verificar que el pipeline funciona correctamente.

Uso:
    python test_pipeline_quick.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np

print("\n" + "="*80)
print("🧪 PRUEBA RÁPIDA - SKLEARN PIPELINE")
print("="*80 + "\n")

try:
    # 1. Imports
    print("📦 Importando módulos...")
    from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
    from acoustic_ml.dataset import DatasetManager
    print("   ✓ Imports exitosos\n")
    
    # 2. Cargar datos
    print("📂 Cargando datos...")
    manager = DatasetManager()
    X_train, X_test, y_train, y_test = manager.load_train_test_split(validate=True)
    print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    # 3. Crear pipeline
    print("🔧 Creando pipeline...")
    pipeline = create_sklearn_pipeline(model_type="random_forest")
    print("   ✓ Pipeline creado\n")
    
    # 4. Entrenar
    print("📊 Entrenando modelo...")
    pipeline.fit(X_train, y_train)
    print("   ✓ Entrenamiento completado\n")
    
    # 5. Mostrar componentes
    print("🔍 Componentes del Feature Pipeline:")
    if pipeline.feature_pipeline is not None:
        for i, (step_name, step) in enumerate(pipeline.feature_pipeline.steps, 1):
            print(f"   {i}. {step_name}: {step.__class__.__name__}")
    
    print(f"\n   {len(pipeline.feature_pipeline.steps)+1}. model: {pipeline.model_trainer.model.__class__.__name__}\n")
    
    # 6. Evaluar
    print("📈 Evaluando modelo...")
    accuracy = pipeline.score(X_test, y_test)
    print(f"   ✓ Accuracy: {accuracy*100:.2f}%\n")
    
    # 7. Predicciones de muestra
    print("🎯 Predicciones de muestra (primeras 5):")
    predictions = pipeline.predict(X_test.head(5))
    
    # Mapeo de números a emociones (por si acaso retorna números)
    emotion_map = {0: 'happy', 1: 'sad', 2: 'angry', 3: 'relax'}
    
    for i, pred in enumerate(predictions, 1):
        actual = y_test.iloc[i-1]
        
        # Convertir a string si es necesario
        pred_str = emotion_map.get(pred, str(pred)) if isinstance(pred, (int, np.integer)) else str(pred)
        actual_str = emotion_map.get(actual, str(actual)) if isinstance(actual, (int, np.integer)) else str(actual)
        
        match = "✓" if pred == actual else "✗"
        print(f"   {i}. Predicho: {pred_str:8s} | Real: {actual_str:8s} {match}")
    
    print("\n" + "="*80)
    print("✅ PRUEBA EXITOSA - Pipeline funcionando correctamente")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}\n")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("❌ PRUEBA FALLIDA")
    print("="*80 + "\n")
