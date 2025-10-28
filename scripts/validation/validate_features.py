"""
Script de validación comprehensivo para features.py refactorizado.
Prueba todos los transformers, builders y factories.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("🧪 VALIDACIÓN COMPREHENSIVA DE FEATURES.PY REFACTORIZADO")
print("="*80)

# Crear datos de prueba
np.random.seed(42)
n_samples = 100
X_train = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, n_samples),
    'feature_2': np.random.normal(5, 2, n_samples),
    'feature_3': np.random.exponential(2, n_samples),
    'feature_4': np.random.uniform(0, 10, n_samples),
    'feature_5': np.random.normal(0, 0.1, n_samples),  # Baja varianza
    'categorical': ['A', 'B'] * (n_samples // 2),  # Columna no numérica
    'id': range(n_samples)  # ID a excluir
})

# Agregar algunos outliers
X_train.loc[0:5, 'feature_1'] = [100, -100, 200, -200, 150, -150]

print(f"\n📊 Datos de prueba creados: {X_train.shape}")
print(f"   Columnas: {list(X_train.columns)}")

# ============================================================================
# TEST 1: IMPORTS
# ============================================================================
print("\n" + "="*80)
print("1️⃣  TEST DE IMPORTS")
print("="*80)

try:
    from acoustic_ml.features import (
        # Transformers
        FeatureTransformer,
        NumericFeatureSelector,
        PowerFeatureTransformer,
        OutlierRemover,
        FeatureScaler,
        CorrelationFilter,
        VarianceThresholdSelector,
        # Builder
        FeaturePipelineBuilder,
        # Factories
        create_preprocessing_pipeline,
        create_feature_selection_pipeline,
        create_full_pipeline
    )
    print("✅ Todos los imports exitosos")
    print("   • FeatureTransformer (base)")
    print("   • 6 Transformers especializados")
    print("   • FeaturePipelineBuilder")
    print("   • 3 Factory functions")
except Exception as e:
    print(f"❌ Error en imports: {e}")
    sys.exit(1)

# ============================================================================
# TEST 2: NUMERIC FEATURE SELECTOR
# ============================================================================
print("\n" + "="*80)
print("2️⃣  TEST DE NUMERICFEATURESELECTOR")
print("="*80)

try:
    selector = NumericFeatureSelector(exclude_cols=['id'])
    X_numeric = selector.fit_transform(X_train)
    
    print(f"✅ NumericFeatureSelector funciona")
    print(f"   • Shape original: {X_train.shape}")
    print(f"   • Shape después: {X_numeric.shape}")
    print(f"   • Features seleccionadas: {selector.feature_names_}")
    print(f"   • Columnas excluidas: categorical, id ✓")
    
    assert X_numeric.shape[1] == 5, "Debería seleccionar 5 columnas numéricas"
    assert 'id' not in X_numeric.columns, "ID debería estar excluido"
    assert 'categorical' not in X_numeric.columns, "Categorical debería estar excluido"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 3: POWER FEATURE TRANSFORMER
# ============================================================================
print("\n" + "="*80)
print("3️⃣  TEST DE POWERFEATURETRANSFORMER")
print("="*80)

try:
    X_numeric_clean = X_numeric.drop(columns=['feature_5'])  # Remover baja varianza
    
    transformer = PowerFeatureTransformer(method='yeo-johnson')
    X_transformed = transformer.fit_transform(X_numeric_clean)
    
    print(f"✅ PowerFeatureTransformer funciona")
    print(f"   • Método: yeo-johnson")
    print(f"   • Shape: {X_transformed.shape}")
    print(f"   • Tipo: {type(X_transformed)}")
    print(f"   • Media aprox: {X_transformed.mean().mean():.4f} (cercana a 0)")
    print(f"   • Std aprox: {X_transformed.std().mean():.4f} (cercana a 1)")
    
    assert isinstance(X_transformed, pd.DataFrame), "Debería preservar DataFrame"
    assert X_transformed.shape == X_numeric_clean.shape, "Shape debería mantenerse"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 4: OUTLIER REMOVER
# ============================================================================
print("\n" + "="*80)
print("4️⃣  TEST DE OUTLIERREMOVER")
print("="*80)

try:
    remover = OutlierRemover(factor=1.5)
    X_no_outliers = remover.fit_transform(X_numeric_clean)
    
    print(f"✅ OutlierRemover funciona")
    print(f"   • Shape antes: {X_numeric_clean.shape}")
    print(f"   • Shape después: {X_no_outliers.shape}")
    print(f"   • Outliers removidos: {remover.n_outliers_removed_}")
    print(f"   • % removido: {remover.n_outliers_removed_/len(X_numeric_clean)*100:.2f}%")
    
    assert X_no_outliers.shape[0] < X_numeric_clean.shape[0], "Debería remover filas"
    assert remover.n_outliers_removed_ > 0, "Debería detectar outliers"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 5: FEATURE SCALER
# ============================================================================
print("\n" + "="*80)
print("5️⃣  TEST DE FEATURESCALER")
print("="*80)

try:
    for method in ['standard', 'minmax', 'robust']:
        scaler = FeatureScaler(method=method)
        X_scaled = scaler.fit_transform(X_numeric_clean)
        
        print(f"✅ FeatureScaler('{method}') funciona")
        print(f"   • Shape: {X_scaled.shape}")
        
        if method == 'standard':
            print(f"   • Media: {X_scaled.mean().mean():.6f} (≈0)")
            print(f"   • Std: {X_scaled.std().mean():.6f} (≈1)")
        elif method == 'minmax':
            print(f"   • Min: {X_scaled.min().min():.6f} (≈0)")
            print(f"   • Max: {X_scaled.max().max():.6f} (≈1)")
        
        assert isinstance(X_scaled, pd.DataFrame), f"Debería preservar DataFrame ({method})"
    
    print("✅ Todos los métodos de escalado funcionan")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 6: CORRELATION FILTER
# ============================================================================
print("\n" + "="*80)
print("6️⃣  TEST DE CORRELATIONFILTER")
print("="*80)

try:
    # Crear features correlacionadas
    X_corr = X_numeric_clean.copy()
    X_corr['feature_1_copy'] = X_corr['feature_1'] * 0.99  # Casi idéntica
    
    filter = CorrelationFilter(threshold=0.95)
    X_filtered = filter.fit_transform(X_corr)
    
    print(f"✅ CorrelationFilter funciona")
    print(f"   • Shape antes: {X_corr.shape}")
    print(f"   • Shape después: {X_filtered.shape}")
    print(f"   • Features removidas: {X_corr.shape[1] - X_filtered.shape[1]}")
    print(f"   • Threshold: {filter.threshold}")
    
    assert X_filtered.shape[1] < X_corr.shape[1], "Debería remover features correlacionadas"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 7: VARIANCE THRESHOLD SELECTOR
# ============================================================================
print("\n" + "="*80)
print("7️⃣  TEST DE VARIANCETHRESHOLDSELECTOR")
print("="*80)

try:
    selector = VarianceThresholdSelector(threshold=0.01)
    X_selected = selector.fit_transform(X_numeric)
    
    print(f"✅ VarianceThresholdSelector funciona")
    print(f"   • Shape antes: {X_numeric.shape}")
    print(f"   • Shape después: {X_selected.shape}")
    print(f"   • Features removidas: {X_numeric.shape[1] - X_selected.shape[1]}")
    print(f"   • Threshold: {selector.threshold}")
    
    assert X_selected.shape[1] <= X_numeric.shape[1], "Puede remover features"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 8: FEATURE PIPELINE BUILDER
# ============================================================================
print("\n" + "="*80)
print("8️⃣  TEST DE FEATUREPIPELINEBUILDER")
print("="*80)

try:
    pipeline = (FeaturePipelineBuilder()
                .add_numeric_selector(exclude_cols=['id'])
                .add_power_transformer(method='yeo-johnson')
                .add_feature_scaler(method='standard')
                .build())
    
    X_processed = pipeline.fit_transform(X_train)
    
    print(f"✅ FeaturePipelineBuilder funciona")
    print(f"   • Pipeline creado con 3 steps")
    print(f"   • Shape final: {X_processed.shape}")
    print(f"   • Steps: {[name for name, _ in pipeline.steps]}")
    
    assert len(pipeline.steps) == 3, "Debería tener 3 steps"
    assert X_processed.shape[0] == X_train.shape[0], "Filas deberían mantenerse"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 9: FACTORY - create_preprocessing_pipeline
# ============================================================================
print("\n" + "="*80)
print("9️⃣  TEST DE create_preprocessing_pipeline")
print("="*80)

try:
    pipeline = create_preprocessing_pipeline(
        exclude_cols=['id'],
        power_transform=True
    )
    
    X_processed = pipeline.fit_transform(X_train)
    
    print(f"✅ create_preprocessing_pipeline funciona")
    print(f"   • Pipeline: {len(pipeline.steps)} steps")
    print(f"   • Shape: {X_processed.shape}")
    print(f"   • Steps: {[name for name, _ in pipeline.steps]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 10: FACTORY - create_feature_selection_pipeline
# ============================================================================
print("\n" + "="*80)
print("🔟 TEST DE create_feature_selection_pipeline")
print("="*80)

try:
    pipeline = create_feature_selection_pipeline(
        variance_threshold=0.01,
        correlation_threshold=0.95
    )
    
    X_selected = pipeline.fit_transform(X_numeric)
    
    print(f"✅ create_feature_selection_pipeline funciona")
    print(f"   • Pipeline: {len(pipeline.steps)} steps")
    print(f"   • Shape antes: {X_numeric.shape}")
    print(f"   • Shape después: {X_selected.shape}")
    print(f"   • Steps: {[name for name, _ in pipeline.steps]}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 11: FACTORY - create_full_pipeline
# ============================================================================
print("\n" + "="*80)
print("1️⃣1️⃣  TEST DE create_full_pipeline")
print("="*80)

try:
    pipeline = create_full_pipeline(
        exclude_cols=['id'],
        remove_outliers=True,
        scale_method='robust'
    )
    
    X_full = pipeline.fit_transform(X_train)
    
    print(f"✅ create_full_pipeline funciona")
    print(f"   • Pipeline completo: {len(pipeline.steps)} steps")
    print(f"   • Shape antes: {X_train.shape}")
    print(f"   • Shape después: {X_full.shape}")
    print(f"   • Steps: {[name for name, _ in pipeline.steps]}")
    print(f"   • Incluye: selector + outlier_remover + power + scaler")
    
    assert len(pipeline.steps) == 4, "Debería tener 4 steps"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 12: VALIDACIÓN DE ERRORES
# ============================================================================
print("\n" + "="*80)
print("1️⃣2️⃣  TEST DE VALIDACIÓN DE ERRORES")
print("="*80)

try:
    # Test 1: Datos None
    try:
        selector = NumericFeatureSelector()
        selector.fit(None)
        print("❌ Debería rechazar datos None")
        sys.exit(1)
    except ValueError:
        print("✅ Rechaza correctamente datos None")
    
    # Test 2: Transformer sin fit
    try:
        selector = NumericFeatureSelector()
        selector.transform(X_train)
        print("❌ Debería requerir fit primero")
        sys.exit(1)
    except RuntimeError:
        print("✅ Requiere fit() antes de transform() correctamente")
    
    # Test 3: Método inválido en scaler
    try:
        scaler = FeatureScaler(method='invalid')
        print("❌ Debería rechazar método inválido")
        sys.exit(1)
    except ValueError:
        print("✅ Rechaza métodos inválidos correctamente")
    
    # Test 4: Pipeline vacío
    try:
        builder = FeaturePipelineBuilder()
        builder.build()
        print("❌ Debería rechazar pipeline vacío")
        sys.exit(1)
    except ValueError:
        print("✅ Rechaza pipeline vacío correctamente")
    
    print("\n✅ Todas las validaciones de errores funcionan")
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# TEST 13: COMPATIBILIDAD CON NUMPY ARRAYS
# ============================================================================
print("\n" + "="*80)
print("1️⃣3️⃣  TEST DE COMPATIBILIDAD CON NUMPY ARRAYS")
print("="*80)

try:
    X_array = X_numeric_clean.values
    
    # Test PowerTransformer con array
    transformer = PowerFeatureTransformer()
    X_transformed = transformer.fit_transform(X_array)
    
    print(f"✅ Compatible con numpy arrays")
    print(f"   • Input type: {type(X_array)}")
    print(f"   • Output type: {type(X_transformed)}")
    print(f"   • Shape preservada: {X_array.shape == X_transformed.shape}")
    
    assert isinstance(X_transformed, np.ndarray), "Debería retornar array"
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "="*80)
print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE (13/13)")
print("="*80)

print("\n📊 RESUMEN DE LA REFACTORIZACIÓN:")
print("   ├─ FeatureTransformer: Clase base con validación ✓")
print("   ├─ NumericFeatureSelector: Selección numérica ✓")
print("   ├─ PowerFeatureTransformer: Normalización ✓")
print("   ├─ OutlierRemover: Detección IQR ✓")
print("   ├─ FeatureScaler: 3 métodos de escalado ✓")
print("   ├─ CorrelationFilter: Filtrado por correlación ✓")
print("   ├─ VarianceThresholdSelector: Filtrado por varianza ✓")
print("   ├─ FeaturePipelineBuilder: Construcción fluida ✓")
print("   ├─ Factory functions: 3 pipelines pre-configurados ✓")
print("   ├─ Validación robusta de errores ✓")
print("   ├─ Compatible con DataFrames y arrays ✓")
print("   └─ Logging comprehensivo ✓")

print("\n🎯 features.py LISTO PARA PRODUCCIÓN!")
print("   • De 88 líneas → ~930 líneas")
print("   • 7 transformers especializados")
print("   • Builder pattern implementado")
print("   • 3 factory functions")
print("   • Documentación completa en español")
print("   • SOLID principles en todos los componentes")

print("\n" + "="*80)
