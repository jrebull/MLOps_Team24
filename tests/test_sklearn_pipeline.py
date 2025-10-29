"""
Tests unitarios para el pipeline de sklearn

Verifica que el pipeline funcione correctamente con:
- Transformaciones de features
- Scaling
- Encoding
- Feature engineering

Autor: MLOps Team 24
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_audio_features():
    """
    Genera features de audio sintéticas para testing
    Simula las features acústicas del dataset turco
    """
    np.random.seed(42)
    n_samples = 100
    
    # Características acústicas típicas
    data = {
        'mfcc_mean': np.random.randn(n_samples),
        'mfcc_std': np.random.randn(n_samples),
        'spectral_centroid_mean': np.random.uniform(500, 4000, n_samples),
        'spectral_rolloff_mean': np.random.uniform(1000, 8000, n_samples),
        'zero_crossing_rate_mean': np.random.uniform(0, 0.3, n_samples),
        'tempo': np.random.uniform(60, 180, n_samples),
        'rms_mean': np.random.uniform(0, 0.5, n_samples),
        'spectral_bandwidth_mean': np.random.uniform(500, 3000, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_labels():
    """
    Genera labels sintéticas para las 4 emociones
    """
    np.random.seed(42)
    emotions = ['Happy', 'Sad', 'Angry', 'Relax']
    return np.random.choice(emotions, 100)


@pytest.fixture
def basic_pipeline():
    """
    Pipeline básico para testing
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])


# ============================================================================
# TESTS DE PIPELINE BÁSICO
# ============================================================================

def test_pipeline_initialization(basic_pipeline):
    """
    Test 1: Verificar que el pipeline se inicializa correctamente
    """
    assert basic_pipeline is not None
    assert len(basic_pipeline.steps) == 2
    assert basic_pipeline.steps[0][0] == 'scaler'
    assert basic_pipeline.steps[1][0] == 'classifier'


def test_pipeline_fit(basic_pipeline, sample_audio_features, sample_labels):
    """
    Test 2: Verificar que el pipeline puede entrenar
    """
    basic_pipeline.fit(sample_audio_features, sample_labels)
    
    # Verificar que el scaler fue fitted
    assert hasattr(basic_pipeline.named_steps['scaler'], 'mean_')
    assert hasattr(basic_pipeline.named_steps['scaler'], 'scale_')
    
    # Verificar que el classifier fue fitted
    assert hasattr(basic_pipeline.named_steps['classifier'], 'classes_')
    assert len(basic_pipeline.named_steps['classifier'].classes_) == 4


def test_pipeline_predict(basic_pipeline, sample_audio_features, sample_labels):
    """
    Test 3: Verificar que el pipeline puede predecir
    """
    basic_pipeline.fit(sample_audio_features, sample_labels)
    predictions = basic_pipeline.predict(sample_audio_features)
    
    assert len(predictions) == len(sample_audio_features)
    assert all(pred in ['Happy', 'Sad', 'Angry', 'Relax'] for pred in predictions)


def test_pipeline_predict_proba(basic_pipeline, sample_audio_features, sample_labels):
    """
    Test 4: Verificar que el pipeline puede dar probabilidades
    """
    basic_pipeline.fit(sample_audio_features, sample_labels)
    probabilities = basic_pipeline.predict_proba(sample_audio_features)
    
    # Verificar dimensiones
    assert probabilities.shape == (len(sample_audio_features), 4)
    
    # Verificar que las probabilidades suman 1
    np.testing.assert_array_almost_equal(
        probabilities.sum(axis=1),
        np.ones(len(sample_audio_features)),
        decimal=6
    )
    
    # Verificar que las probabilidades están en [0, 1]
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


# ============================================================================
# TESTS DE TRANSFORMACIONES
# ============================================================================

def test_standard_scaler_transformation(sample_audio_features):
    """
    Test 5: Verificar StandardScaler transforma correctamente
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sample_audio_features)
    
    # Verificar que los datos están escalados (media ~0, std ~1)
    np.testing.assert_array_almost_equal(
        scaled_data.mean(axis=0),
        np.zeros(sample_audio_features.shape[1]),
        decimal=10
    )
    np.testing.assert_array_almost_equal(
        scaled_data.std(axis=0),
        np.ones(sample_audio_features.shape[1]),
        decimal=10
    )


def test_minmax_scaler_transformation(sample_audio_features):
    """
    Test 6: Verificar MinMaxScaler transforma correctamente
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(sample_audio_features)
    
    # Verificar que los datos están en [0, 1] con tolerancia numérica
    assert np.all(scaled_data >= -1e-10), "Algunos valores son menores que 0"
    assert np.all(scaled_data <= 1 + 1e-10), "Algunos valores son mayores que 1"
    
    # Verificar que el mínimo y máximo son correctos (con tolerancia realista)
    np.testing.assert_array_almost_equal(
        scaled_data.min(axis=0),
        np.zeros(sample_audio_features.shape[1]),
        decimal=7  # Tolerancia realista para floats
    )
    np.testing.assert_array_almost_equal(
        scaled_data.max(axis=0),
        np.ones(sample_audio_features.shape[1]),
        decimal=7  # Tolerancia realista para floats
    )


# ============================================================================
# TESTS DE VALIDACIÓN DE DATOS
# ============================================================================

def test_pipeline_handles_missing_values():
    """
    Test 7: Verificar que datos con NaN requieren limpieza previa
    
    sklearn moderno propaga NaN sin error, pero el resultado es inválido.
    Este test verifica que limpieza de datos es necesaria.
    """
    # Crear datos con NaN
    data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [5, np.nan, 3, 2, 1]
    })
    
    labels = ['Happy', 'Sad', 'Angry', 'Relax', 'Happy']
    
    # StandardScaler propaga NaN (no falla, pero resultado inválido)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    assert np.isnan(scaled_data).any(), "StandardScaler debería propagar NaN"
    
    # Verificar que limpieza de datos funciona correctamente
    clean_data = data.dropna()
    clean_labels = [labels[i] for i in clean_data.index]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    pipeline.fit(clean_data, clean_labels)
    predictions = pipeline.predict(clean_data)
    
    # Verificar resultados válidos con datos limpios
    assert len(predictions) == len(clean_data)
    assert not any(pd.isna(predictions)), "Predicciones no deben contener NaN"


def test_pipeline_input_shape_consistency(sample_audio_features, sample_labels):
    """
    Test 8: Verificar que el pipeline falla con inconsistencias de shape
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Entrenar con 8 features
    pipeline.fit(sample_audio_features, sample_labels)
    
    # Intentar predecir con diferente número de features
    wrong_data = sample_audio_features.iloc[:, :5]  # Solo 5 features
    
    with pytest.raises(ValueError):
        pipeline.predict(wrong_data)


# ============================================================================
# TESTS DE REPRODUCIBILIDAD
# ============================================================================

def test_pipeline_reproducibility(sample_audio_features, sample_labels):
    """
    Test 9: Verificar que el pipeline es reproducible con random_state
    """
    pipeline1 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    pipeline2 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Entrenar ambos pipelines
    pipeline1.fit(sample_audio_features, sample_labels)
    pipeline2.fit(sample_audio_features, sample_labels)
    
    # Predecir con ambos
    pred1 = pipeline1.predict(sample_audio_features)
    pred2 = pipeline2.predict(sample_audio_features)
    
    # Verificar que las predicciones son idénticas
    np.testing.assert_array_equal(pred1, pred2)


def test_pipeline_different_seeds_different_results(sample_audio_features, sample_labels):
    """
    Test 10: Verificar que diferentes random_state dan diferentes resultados
    """
    pipeline1 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    pipeline2 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=99))
    ])
    
    pipeline1.fit(sample_audio_features, sample_labels)
    pipeline2.fit(sample_audio_features, sample_labels)
    
    pred1 = pipeline1.predict(sample_audio_features)
    pred2 = pipeline2.predict(sample_audio_features)
    
    # Verificar que al menos algunas predicciones son diferentes
    assert not np.array_equal(pred1, pred2)


# ============================================================================
# TESTS DE PERFORMANCE
# ============================================================================

def test_pipeline_training_speed(sample_audio_features, sample_labels, benchmark=None):
    """
    Test 11: Verificar que el pipeline entrena en tiempo razonable
    (benchmark opcional para pytest-benchmark)
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    # Medir tiempo de entrenamiento
    import time
    start = time.time()
    pipeline.fit(sample_audio_features, sample_labels)
    duration = time.time() - start
    
    # Verificar que toma menos de 5 segundos (generoso para CI/CD)
    assert duration < 5.0, f"Training took {duration:.2f}s, expected < 5s"


def test_pipeline_prediction_speed(basic_pipeline, sample_audio_features, sample_labels):
    """
    Test 12: Verificar que el pipeline predice rápido
    """
    basic_pipeline.fit(sample_audio_features, sample_labels)
    
    import time
    start = time.time()
    _ = basic_pipeline.predict(sample_audio_features)
    duration = time.time() - start
    
    # Predicción debe ser muy rápida
    assert duration < 1.0, f"Prediction took {duration:.2f}s, expected < 1s"


# ============================================================================
# EJECUCIÓN STANDALONE
# ============================================================================

if __name__ == "__main__":
    # Permite ejecutar tests directamente
    pytest.main([__file__, "-v", "--tb=short"])
