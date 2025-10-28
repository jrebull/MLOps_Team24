#!/usr/bin/env python3
"""
Test Script: Validación del SklearnMLPipeline

Valida que el pipeline end-to-end funciona correctamente:
1. Load datos
2. Fit pipeline
3. Predict
4. Score
"""

import sys
import pandas as pd
from pathlib import Path

print("=" * 70)
print("🧪 TEST: SklearnMLPipeline End-to-End")
print("=" * 70)

# 1. Imports
print("\n[1/5] Importando módulos...")
try:
    from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline, create_sklearn_pipeline
    from acoustic_ml.dataset import DatasetManager
    print("✅ Imports exitosos")
except Exception as e:
    print(f"❌ Error en imports: {e}")
    sys.exit(1)

# 2. Cargar datos
print("\n[2/5] Cargando datos de prueba...")
try:
    dataset_manager = DatasetManager()
    
    # Intentar cargar datos procesados
    data_dir = Path("data/processed")
    if (data_dir / "X_train.csv").exists():
        X_train = pd.read_csv(data_dir / "X_train.csv")
        y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
        X_test = pd.read_csv(data_dir / "X_test.csv")
        y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
        print(f"✅ Datos cargados desde {data_dir}")
    else:
        # Cargar y dividir datos raw
        df = dataset_manager.load_data("data/raw/turkish_music_emotion.csv")
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=['Class'])
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("✅ Datos cargados y divididos desde raw")
    
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    sys.exit(1)

# 3. Crear pipeline
print("\n[3/5] Creando pipeline sklearn...")
try:
    pipeline = create_sklearn_pipeline(
        model_type="random_forest",
        model_params={
            'n_estimators': 50,  # Pocos para test rápido
            'max_depth': 10,
            'random_state': 42
        }
    )
    print("✅ Pipeline creado")
    print(f"   Model type: {pipeline.model_type}")
    
except Exception as e:
    print(f"❌ Error creando pipeline: {e}")
    sys.exit(1)

# 4. Entrenar pipeline
print("\n[4/5] Entrenando pipeline...")
try:
    # Usar subset pequeño para test rápido
    X_train_small = X_train.head(200)
    y_train_small = y_train.head(200)
    
    pipeline.fit(X_train_small, y_train_small)
    print("✅ Pipeline entrenado")
    
except Exception as e:
    print(f"❌ Error entrenando: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Predecir y evaluar
print("\n[5/5] Prediciendo y evaluando...")
try:
    # Usar subset pequeño para test
    X_test_small = X_test.head(50)
    y_test_small = y_test.head(50)
    
    # Predict
    predictions = pipeline.predict(X_test_small)
    print(f"✅ Predicciones realizadas: {len(predictions)} muestras")
    
    # Score
    accuracy = pipeline.score(X_test_small, y_test_small)
    print(f"✅ Accuracy calculado: {accuracy:.4f}")
    
except Exception as e:
    print(f"❌ Error en predicción/evaluación: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Resumen
print("\n" + "=" * 70)
print("✅ TODOS LOS TESTS PASADOS")
print("=" * 70)
print(f"📊 Accuracy en test subset: {accuracy:.2%}")
print(f"🎯 Pipeline sklearn funcionando correctamente")
print("\n💡 El pipeline está listo para:")
print("   - Usar con GridSearchCV")
print("   - Usar con cross_val_score")
print("   - Integrar con MLflow")
print("   - Deploy en producción")
