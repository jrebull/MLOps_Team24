#!/usr/bin/env python3
"""
Reentrenamiento con Enhanced Features
======================================

Este script:
1. Carga el dataset limpio
2. Aplica feature engineering (agrega 14+ nuevas features)
3. Re-entrena el modelo Random Forest
4. Compara performance con modelo anterior
5. Guarda en MLflow con documentación completa

Basado en hallazgos de analyze_4_feature_importance.py:
- Features originales tienen Cohen's d bajo (< 0.5 mayoría)
- Solo Roughness, Eventdensity, AttackTime discriminan bien
- Necesitamos features derivadas para mejor discriminación

Autor: MLOps Team 24
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

# Importar módulos del proyecto
from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import SklearnMLPipeline

# Importar feature engineering
sys.path.append('.')
from feature_engineering import apply_feature_engineering, analyze_new_features

print("="*80)
print("🚀 REENTRENAMIENTO CON ENHANCED FEATURES")
print("="*80)
print(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# PASO 1: Cargar Dataset Original
# =============================================================================
print("\n[1/7] Cargando dataset limpio...")
dm = DatasetManager()
df_original = dm.load_processed(filename="turkish_music_emotion_v2_cleaned_full.csv")

print(f"   ✅ Dataset cargado: {df_original.shape}")
print(f"   Distribución de clases:")
for cls, count in df_original['Class'].value_counts().sort_index().items():
    print(f"      {cls}: {count}")

# =============================================================================
# PASO 2: Aplicar Feature Engineering
# =============================================================================
print("\n[2/7] Aplicando Feature Engineering...")
df_enhanced = apply_feature_engineering(df_original, verbose=True)

# Guardar dataset mejorado
output_path = Path("data/processed/turkish_music_emotion_v2_ENHANCED.csv")
df_enhanced.to_csv(output_path, index=False)
print(f"\n   💾 Dataset mejorado guardado: {output_path}")

# =============================================================================
# PASO 3: Analizar Impacto de Nuevas Features
# =============================================================================
print("\n[3/7] Analizando impacto de nuevas features...")
feature_analysis = analyze_new_features(df_original, df_enhanced)

# =============================================================================
# PASO 4: Preparar Datos para Training
# =============================================================================
print("\n[4/7] Preparando datos para training...")

# Separar features y labels
X = df_enhanced.drop('Class', axis=1)
y = df_enhanced['Class']

# Remover columna mixed_type_col si existe
if 'mixed_type_col' in X.columns:
    X = X.drop('mixed_type_col', axis=1)
    print(f"   ✅ Removida columna 'mixed_type_col'")

# Asegurar que todo es numérico
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

print(f"   Features finales: {len(X.columns)}")
print(f"   Samples: {len(X)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"   ✅ Split:")
print(f"      Train: {len(X_train)} samples")
print(f"      Test: {len(X_test)} samples")

# =============================================================================
# PASO 5: Entrenar Modelo con Enhanced Features
# =============================================================================
print("\n[5/7] Entrenando modelo con enhanced features...")

# Configuración del modelo (misma que modelo anterior para comparación justa)
pipeline = SklearnMLPipeline(
    model_type='random_forest',
    model_params={  # ✅ Parámetro correcto
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    scale_method='robust'  # ✅ Parámetro correcto
)

# Entrenar
print("   🎯 Training...")
pipeline.fit(X_train, y_train)
print("   ✅ Training completado")

# =============================================================================
# PASO 6: Evaluar Performance
# =============================================================================
print("\n[6/7] Evaluando performance...")

# Predicciones
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n   📊 ACCURACY:")
print(f"      Train: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"      Test:  {test_acc:.4f} ({test_acc*100:.2f}%)")

# Classification Report
print(f"\n   📋 Classification Report (Test):")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix para Angry específicamente
cm = confusion_matrix(y_test, y_pred_test, labels=['angry', 'happy', 'relax', 'sad'])
angry_idx = 0  # angry es el primero en labels
angry_total = cm[angry_idx].sum()
angry_correct = cm[angry_idx, angry_idx]
angry_acc = angry_correct / angry_total if angry_total > 0 else 0

print(f"\n   🔥 PERFORMANCE DE 'ANGRY' ESPECÍFICAMENTE:")
print(f"      Total samples: {angry_total}")
print(f"      Correctos: {angry_correct}")
print(f"      Accuracy: {angry_acc:.4f} ({angry_acc*100:.2f}%)")

# Comparación con modelo anterior
print(f"\n   📈 COMPARACIÓN CON MODELO ANTERIOR:")
print(f"      Modelo anterior - Test Accuracy: 84.30%")
print(f"      Modelo nuevo - Test Accuracy: {test_acc*100:.2f}%")
if test_acc > 0.8430:
    improvement = (test_acc - 0.8430) * 100
    print(f"      ✅ MEJORA: +{improvement:.2f} puntos porcentuales")
else:
    decline = (0.8430 - test_acc) * 100
    print(f"      ⚠️  DECLINE: -{decline:.2f} puntos porcentuales")

print(f"\n      Modelo anterior - Angry Accuracy: 82.8%")
print(f"      Modelo nuevo - Angry Accuracy: {angry_acc*100:.2f}%")
if angry_acc > 0.828:
    improvement = (angry_acc - 0.828) * 100
    print(f"      ✅ MEJORA EN ANGRY: +{improvement:.2f} puntos porcentuales")
else:
    decline = (0.828 - angry_acc) * 100
    print(f"      ⚠️  DECLINE EN ANGRY: -{decline:.2f} puntos porcentuales")

# =============================================================================
# PASO 7: Guardar en MLflow
# =============================================================================
print("\n[7/7] Guardando en MLflow...")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("turkish-music-emotion-recognition")

with mlflow.start_run(run_name="RandomForest_ENHANCED_Features_v3"):
    # Log params
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", "None")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("preprocessing", "power_transformer,robust_scaler")
    mlflow.log_param("feature_engineering", "enhanced_v1")
    mlflow.log_param("n_features_original", len(df_original.columns) - 1)
    mlflow.log_param("n_features_enhanced", len(X.columns))
    mlflow.log_param("n_features_new", len(X.columns) - (len(df_original.columns) - 1))
    
    # Log metrics
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("angry_accuracy", angry_acc)
    
    # Log metrics por clase
    report = classification_report(y_test, y_pred_test, output_dict=True)
    for emotion in ['angry', 'happy', 'relax', 'sad']:
        if emotion in report:
            mlflow.log_metric(f"{emotion}_precision", report[emotion]['precision'])
            mlflow.log_metric(f"{emotion}_recall", report[emotion]['recall'])
            mlflow.log_metric(f"{emotion}_f1", report[emotion]['f1-score'])
    
    # Log modelo
    mlflow.sklearn.log_model(
        pipeline,
        "model",
        input_example=X_train.iloc[:5]
    )
    
    # Log dataset mejorado como artifact
    mlflow.log_artifact(str(output_path))
    
    # Log análisis de features
    feature_analysis_df = pd.DataFrame(feature_analysis)
    feature_analysis_path = Path("feature_analysis_results.csv")
    feature_analysis_df.to_csv(feature_analysis_path, index=False)
    mlflow.log_artifact(str(feature_analysis_path))
    feature_analysis_path.unlink()  # Limpiar
    
    # Tags
    mlflow.set_tag("team", "MLOps_Team24")
    mlflow.set_tag("phase", "2")
    mlflow.set_tag("feature_engineering", "enhanced")
    mlflow.set_tag("improvement_focus", "angry_classification")
    
    run_id = mlflow.active_run().info.run_id
    print(f"\n   ✅ Guardado en MLflow")
    print(f"   Run ID: {run_id}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "="*80)
print("📊 RESUMEN FINAL")
print("="*80)

print(f"\n✅ Modelo entrenado exitosamente con enhanced features")
print(f"\nMétricas Clave:")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Angry Accuracy: {angry_acc:.4f} ({angry_acc*100:.2f}%)")
print(f"   Features totales: {len(X.columns)}")
print(f"   Features nuevas: {len(X.columns) - (len(df_original.columns) - 1)}")

print(f"\nArchivos generados:")
print(f"   ✅ {output_path}")
print(f"   ✅ MLflow Run ID: {run_id}")

print(f"\n🎯 Próximos pasos:")
print(f"   1. Comparar este modelo con el anterior en producción")
print(f"   2. Si mejora > 2%, actualizar modelo en producción")
print(f"   3. Versionar dataset mejorado con DVC")
print(f"   4. Documentar cambios en README")

print("\n" + "="*80)
print("✅ PROCESO COMPLETADO")
print("="*80)
