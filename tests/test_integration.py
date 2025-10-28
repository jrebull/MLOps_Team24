"""
Test de integración completa usando pytest
Usa el DatasetManager para cargar datos correctamente
"""
import pytest
from pathlib import Path
import sys

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline
from acoustic_ml.dataset import DatasetManager

# Inicializar el manager
dataset_manager = DatasetManager()

def test_pipeline_end_to_end():
    """Test del pipeline completo end-to-end"""
    X_train, X_test, y_train, y_test = dataset_manager.load_train_test_split(validate=True)
    
    pipeline = SklearnMLPipeline(model_type='random_forest')
    pipeline.fit(X_train, y_train)
    
    accuracy = pipeline.score(X_test, y_test)
    
    assert accuracy > 0.70, f"Accuracy muy baja: {accuracy}"
    # Validar contrato público: el pipeline debe poder hacer predicciones
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(y_test), "Pipeline no genera predicciones correctas"
    print(f"✓ Pipeline funciona con accuracy: {accuracy:.4f}")

def test_data_integrity():
    """Test de integridad de datos usando DatasetManager"""
    X_train, X_test, y_train, y_test = dataset_manager.load_train_test_split(validate=True)
    
    assert X_train.shape[0] == y_train.shape[0], "X_train y y_train no coinciden"
    assert X_test.shape[0] == y_test.shape[0], "X_test y y_test no coinciden"
    # Validación dinámica: train y test deben tener mismo número de features
    assert X_train.shape[1] == X_test.shape[1], f"Features no coinciden: train={X_train.shape[1]}, test={X_test.shape[1]}"
    assert 40 <= X_train.shape[1] <= 60, f"Número de features fuera de rango esperado: {X_train.shape[1]}"
    print(f"✓ Datos íntegros: train={X_train.shape}, test={X_test.shape}")

def test_required_files_exist():
    """Test de archivos requeridos del proyecto"""
    required_files = [
        "acoustic_ml/modeling/sklearn_pipeline.py",
        "scripts/analysis/analyze_outliers.py",
        "scripts/analysis/compare_scalers.py",
        "data/processed/X_train.csv",
        "data/processed/X_test.csv",
        "reports/figures/outlier_analysis.png",
    ]
    
    for file_path in required_files:
        assert Path(file_path).exists(), f"Falta archivo: {file_path}"
    
    print(f"✓ Todos los archivos requeridos presentes ({len(required_files)} archivos)")

def test_predictions_shape():
    """Test que las predicciones tienen el formato correcto"""
    X_train, X_test, y_train, y_test = dataset_manager.load_train_test_split()
    
    pipeline = SklearnMLPipeline(model_type='random_forest')
    pipeline.fit(X_train, y_train)
    
    predictions = pipeline.predict(X_test)
    
    assert predictions.shape[0] == X_test.shape[0], "Número de predicciones incorrecto"
    assert all(pred in [0, 1, 2, 3] for pred in predictions), "Predicciones fuera de rango"
    print(f"✓ Predicciones correctas: {predictions.shape}")

def test_dataset_manager_validation():
    """Test específico del DatasetManager con validación"""
    # Este test valida que el DatasetManager.load_train_test_split() funciona con validate=True
    X_train, X_test, y_train, y_test = dataset_manager.load_train_test_split(validate=True)
    
    # Verificar que no hay datos vacíos
    assert not X_train.empty, "X_train está vacío"
    assert not X_test.empty, "X_test está vacío"
    assert len(y_train) > 0, "y_train está vacío"
    assert len(y_test) > 0, "y_test está vacío"
    
    # Verificar dimensiones
    assert X_train.shape[0] > 0, "X_train no tiene filas"
    assert X_test.shape[0] > 0, "X_test no tiene filas"
    
    print(f"✓ DatasetManager validation passed")

def test_model_types():
    """Test que varios tipos de modelos funcionan"""
    X_train, X_test, y_train, y_test = dataset_manager.load_train_test_split()
    
    # Solo probar algunos modelos para no hacer el test muy lento
    model_types = ['random_forest', 'gradient_boosting']
    
    for model_type in model_types:
        pipeline = SklearnMLPipeline(model_type=model_type)
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        
        assert accuracy > 0.50, f"{model_type} accuracy muy baja: {accuracy}"
        print(f"✓ {model_type}: {accuracy:.4f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
