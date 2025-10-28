#!/usr/bin/env python3
"""
Script de validación rápida post-correcciones
Verifica que las correcciones funcionan antes de ejecutar pytest completo
"""

import sys
from pathlib import Path

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline

def main():
    print("=" * 70)
    print("🔍 VALIDACIÓN RÁPIDA POST-CORRECCIONES")
    print("=" * 70)
    
    # Test 1: DatasetManager
    print("\n[1/4] Verificando DatasetManager...")
    try:
        dm = DatasetManager()
        X_train, X_test, y_train, y_test = dm.load_train_test_split(validate=True)
        
        # Validaciones dinámicas (como en el test corregido)
        assert X_train.shape[1] == X_test.shape[1], "Features no coinciden"
        assert 40 <= X_train.shape[1] <= 60, f"Features fuera de rango: {X_train.shape[1]}"
        
        print(f"   ✅ Dataset OK: {X_train.shape[1]} features")
        print(f"   ✅ Train: {X_train.shape}, Test: {X_test.shape}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 2: Pipeline fit
    print("\n[2/4] Verificando Pipeline fit...")
    try:
        pipeline = SklearnMLPipeline(model_type='random_forest')
        pipeline.fit(X_train, y_train)
        print("   ✅ Pipeline fit OK")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 3: Contrato público predict (corrección aplicada)
    print("\n[3/4] Verificando contrato público (predict)...")
    try:
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(y_test), "Predicciones incorrectas"
        print(f"   ✅ Predict OK: {len(predictions)} predicciones")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 4: Score
    print("\n[4/4] Verificando accuracy...")
    try:
        accuracy = pipeline.score(X_test, y_test)
        assert accuracy > 0.70, f"Accuracy muy baja: {accuracy}"
        print(f"   ✅ Score OK: {accuracy:.4f}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Resumen
    print("\n" + "=" * 70)
    print("✅ TODAS LAS VALIDACIONES PASARON")
    print("=" * 70)
    print("\n💡 Siguiente paso:")
    print("   pytest tests/test_integration.py -v")
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
