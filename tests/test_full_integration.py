"""
Test de integración completo - End-to-End

Verifica el flujo completo del proyecto:
- Carga de datos
- Feature engineering
- Entrenamiento del pipeline
- Evaluación del modelo
- Guardado y carga de modelos

Autor: MLOps Team 24
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import tempfile
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_turkish_music_dataset():
    """
    Genera un dataset sintético que simula características de música turca
    con las 4 emociones: Happy, Sad, Angry, Relax
    """
    np.random.seed(42)
    n_samples = 400  # 100 por cada emoción
    
    emotions = []
    features_list = []
    
    # Generar patrones diferentes para cada emoción
    for emotion_idx, emotion in enumerate(['Happy', 'Sad', 'Angry', 'Relax']):
        for _ in range(100):
            # Cada emoción tiene patrones característicos
            if emotion == 'Happy':
                features = {
                    'mfcc_mean': np.random.normal(50, 10),
                    'spectral_centroid_mean': np.random.uniform(2000, 4000),
                    'tempo': np.random.uniform(120, 160),
                    'energy': np.random.uniform(0.6, 1.0),
                    'zero_crossing_rate': np.random.uniform(0.1, 0.2)
                }
            elif emotion == 'Sad':
                features = {
                    'mfcc_mean': np.random.normal(30, 8),
                    'spectral_centroid_mean': np.random.uniform(500, 1500),
                    'tempo': np.random.uniform(60, 90),
                    'energy': np.random.uniform(0.2, 0.5),
                    'zero_crossing_rate': np.random.uniform(0.05, 0.15)
                }
            elif emotion == 'Angry':
                features = {
                    'mfcc_mean': np.random.normal(70, 15),
                    'spectral_centroid_mean': np.random.uniform(3000, 5000),
                    'tempo': np.random.uniform(140, 180),
                    'energy': np.random.uniform(0.7, 1.0),
                    'zero_crossing_rate': np.random.uniform(0.2, 0.3)
                }
            else:  # Relax
                features = {
                    'mfcc_mean': np.random.normal(40, 8),
                    'spectral_centroid_mean': np.random.uniform(1000, 2000),
                    'tempo': np.random.uniform(70, 100),
                    'energy': np.random.uniform(0.3, 0.6),
                    'zero_crossing_rate': np.random.uniform(0.08, 0.18)
                }
            
            features_list.append(features)
            emotions.append(emotion)
    
    X = pd.DataFrame(features_list)
    y = np.array(emotions)
    
    return X, y


@pytest.fixture
def train_test_data(synthetic_turkish_music_dataset):
    """
    Split del dataset sintético en train/test
    """
    X, y = synthetic_turkish_music_dataset
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


# ============================================================================
# TEST 1: PIPELINE COMPLETO - RANDOMFOREST
# ============================================================================

def test_full_pipeline_randomforest(train_test_data):
    """
    Test integración completa: RandomForest con StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    # Crear pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        ))
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)
    
    # Predecir
    y_pred = pipeline.predict(X_test)
    
    # Evaluar
    accuracy = accuracy_score(y_test, y_pred)
    
    # Verificar que el modelo aprende algo razonable
    assert accuracy > 0.5, f"Accuracy demasiado baja: {accuracy:.3f}"
    
    # Verificar que predice las 4 clases
    unique_predictions = set(y_pred)
    assert len(unique_predictions) >= 3, "El modelo debe predecir al menos 3 clases"
    
    print(f"\n✅ RandomForest Accuracy: {accuracy:.3f}")
    print(f"   Clases predichas: {sorted(unique_predictions)}")


# ============================================================================
# TEST 2: PIPELINE COMPLETO - LOGISTIC REGRESSION
# ============================================================================

def test_full_pipeline_logistic_regression(train_test_data):
    """
    Test integración completa: Logistic Regression con StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.4, f"Accuracy demasiado baja: {accuracy:.3f}"
    
    print(f"\n✅ Logistic Regression Accuracy: {accuracy:.3f}")


# ============================================================================
# TEST 3: PIPELINE COMPLETO - SVM
# ============================================================================

def test_full_pipeline_svm(train_test_data):
    """
    Test integración completa: SVM con StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            kernel='rbf',
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.4, f"Accuracy demasiado baja: {accuracy:.3f}"
    
    print(f"\n✅ SVM Accuracy: {accuracy:.3f}")


# ============================================================================
# TEST 4: COMPARACIÓN DE SCALERS
# ============================================================================

def test_scaler_comparison(train_test_data):
    """
    Test: Comparar StandardScaler vs MinMaxScaler
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    results = {}
    
    for scaler_name, scaler in scalers.items():
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', RandomForestClassifier(
                n_estimators=30,
                random_state=42
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[scaler_name] = accuracy
    
    # Verificar que ambos scalers funcionan
    for scaler_name, acc in results.items():
        assert acc > 0.4, f"{scaler_name} accuracy demasiado baja: {acc:.3f}"
    
    print(f"\n✅ Scaler Comparison:")
    for scaler_name, acc in results.items():
        print(f"   {scaler_name}: {acc:.3f}")


# ============================================================================
# TEST 5: PERSISTENCIA DEL MODELO
# ============================================================================

def test_model_save_load(train_test_data):
    """
    Test: Guardar y cargar modelo debe mantener predicciones
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    # Entrenar modelo
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=30, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    predictions_before = pipeline.predict(X_test)
    
    # Guardar modelo
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        joblib.dump(pipeline, tmp_path)
    
    try:
        # Cargar modelo
        loaded_pipeline = joblib.load(tmp_path)
        predictions_after = loaded_pipeline.predict(X_test)
        
        # Verificar que las predicciones son idénticas
        np.testing.assert_array_equal(
            predictions_before,
            predictions_after,
            err_msg="Las predicciones cambiaron después de guardar/cargar"
        )
        
        print(f"\n✅ Model persistence: OK")
        print(f"   Predictions match: 100%")
        
    finally:
        # Limpiar archivo temporal
        Path(tmp_path).unlink(missing_ok=True)


# ============================================================================
# TEST 6: REPORT DE CLASIFICACIÓN
# ============================================================================

def test_classification_report_generation(train_test_data):
    """
    Test: Generar reporte de clasificación completo
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Generar reporte
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )
    
    # Verificar que el reporte contiene todas las emociones
    expected_emotions = ['Happy', 'Sad', 'Angry', 'Relax']
    for emotion in expected_emotions:
        assert emotion in report, f"Falta {emotion} en el reporte"
    
    # Verificar métricas básicas
    assert 'accuracy' in report
    assert 'macro avg' in report
    assert 'weighted avg' in report
    
    print(f"\n✅ Classification Report Generated:")
    print(f"   Accuracy: {report['accuracy']:.3f}")
    print(f"   Macro F1: {report['macro avg']['f1-score']:.3f}")


# ============================================================================
# TEST 7: VALIDACIÓN DE FEATURES
# ============================================================================

def test_feature_consistency(synthetic_turkish_music_dataset):
    """
    Test: Verificar que las features son consistentes
    """
    X, y = synthetic_turkish_music_dataset
    
    # Verificar que no hay NaN
    assert not X.isnull().any().any(), "Dataset contiene valores NaN"
    
    # Verificar que no hay infinitos
    assert not np.isinf(X.values).any(), "Dataset contiene valores infinitos"
    
    # Verificar dimensiones
    assert X.shape[0] == 400, f"Número de samples incorrecto: {X.shape[0]}"
    assert X.shape[1] == 5, f"Número de features incorrecto: {X.shape[1]}"
    
    # Verificar labels
    unique_emotions = set(y)
    expected_emotions = {'Happy', 'Sad', 'Angry', 'Relax'}
    assert unique_emotions == expected_emotions, "Emociones incorrectas"
    
    # Verificar distribución balanceada
    emotion_counts = pd.Series(y).value_counts()
    assert all(count == 100 for count in emotion_counts.values), "Dataset desbalanceado"
    
    print(f"\n✅ Feature Consistency: OK")
    print(f"   Shape: {X.shape}")
    print(f"   Emotions: {sorted(unique_emotions)}")


# ============================================================================
# TEST 8: ROBUSTEZ CON DATOS OUTLIERS
# ============================================================================

def test_pipeline_handles_outliers(train_test_data):
    """
    Test: Pipeline debe manejar outliers sin fallar
    """
    X_train, X_test, y_train, y_test = train_test_data
    
    # Añadir outliers extremos al conjunto de test
    X_test_with_outliers = X_test.copy()
    X_test_with_outliers.iloc[0] = X_test_with_outliers.iloc[0] * 100  # Outlier extremo
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=30, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Debe poder predecir sin errores
    try:
        predictions = pipeline.predict(X_test_with_outliers)
        assert len(predictions) == len(X_test_with_outliers)
        print(f"\n✅ Pipeline handles outliers: OK")
    except Exception as e:
        pytest.fail(f"Pipeline falló con outliers: {str(e)}")


# ============================================================================
# TEST 9: CROSS-VALIDATION SMOKE TEST
# ============================================================================

def test_cross_validation_smoke(train_test_data):
    """
    Test: Verificar que el pipeline funciona con cross-validation
    """
    from sklearn.model_selection import cross_val_score
    
    X_train, _, y_train, _ = train_test_data
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=30, random_state=42))
    ])
    
    # Cross-validation rápido (3 folds)
    scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
    
    # Verificar que devuelve scores razonables
    assert len(scores) == 3
    assert all(score > 0.3 for score in scores), "CV scores demasiado bajos"
    
    print(f"\n✅ Cross-Validation: OK")
    print(f"   Mean CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")


# ============================================================================
# TEST 10: TIEMPO DE ENTRENAMIENTO TOTAL
# ============================================================================

def test_full_training_time(train_test_data):
    """
    Test: Verificar que el entrenamiento completo es razonable
    """
    import time
    
    X_train, X_test, y_train, y_test = train_test_data
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    
    start = time.time()
    pipeline.fit(X_train, y_train)
    _ = pipeline.predict(X_test)
    duration = time.time() - start
    
    # El entrenamiento completo debe tomar menos de 10 segundos
    assert duration < 10.0, f"Training demasiado lento: {duration:.2f}s"
    
    print(f"\n✅ Full training time: {duration:.2f}s")


# ============================================================================
# EJECUCIÓN STANDALONE
# ============================================================================

if __name__ == "__main__":
    # Permite ejecutar tests directamente con verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
