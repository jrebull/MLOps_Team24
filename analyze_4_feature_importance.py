"""
Script 4: Feature Importance Analysis for "Angry" Class
========================================================
Analiza qué features son más importantes para clasificar "angry"
y cómo se comparan con otras emociones.

Usage:
    python3 analyze_4_feature_importance.py
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("file:./mlruns")


def load_model_and_data(run_id="eb05c7698f12499b86ed35ca6efc15a7"):
    """Carga modelo y datos."""
    
    print("📂 Cargando modelo y datos...")
    
    # Cargar dataset
    data_path = Path("data/processed/turkish_music_emotion_v2_cleaned_full.csv")
    df = pd.read_csv(data_path)
    df['Class'] = df['Class'].str.strip().str.lower()
    
    # Separar features y labels
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split con mismo random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Cargar modelo
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    print(f"  ✅ Modelo cargado: {run_id}")
    print(f"  📊 Training: {len(X_train)}, Test: {len(X_test)}")
    
    return model, X_train, X_test, y_train, y_test, X.columns.tolist()


def analyze_random_forest_importance(model, feature_names):
    """Analiza feature importance del Random Forest."""
    
    print("\n" + "=" * 70)
    print("🌳 FEATURE IMPORTANCE (RANDOM FOREST)")
    print("=" * 70)
    
    try:
        # Obtener el modelo RandomForest del pipeline
        # El pipeline tiene: power_transformer → robust_scaler → random_forest
        rf_model = model.named_steps['random_forest']
        
        # Feature importance
        importances = rf_model.feature_importances_
        
        # Crear DataFrame
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 20 Features más importantes:")
        print("-" * 70)
        print(f"{'Rank':>6s} {'Feature':40s} {'Importance':>15s} {'Bar':20s}")
        print("-" * 70)
        
        max_importance = df_importance['importance'].max()
        
        for idx, row in df_importance.head(20).iterrows():
            rank = df_importance.index.get_loc(idx) + 1
            bar_length = int((row['importance'] / max_importance) * 20)
            bar = "█" * bar_length
            print(f"{rank:6d} {row['feature']:40s} {row['importance']:15.6f} {bar}")
        
        # Estadísticas
        print("\n📊 Estadísticas de importancia:")
        print("-" * 50)
        print(f"  Top 5 features acumulan: {df_importance.head(5)['importance'].sum():.1%} de importancia")
        print(f"  Top 10 features acumulan: {df_importance.head(10)['importance'].sum():.1%} de importancia")
        print(f"  Top 20 features acumulan: {df_importance.head(20)['importance'].sum():.1%} de importancia")
        
        return df_importance
        
    except Exception as e:
        print(f"❌ Error al extraer feature importance: {e}")
        return None


def analyze_angry_feature_distribution(X_train, y_train, X_test, y_test, feature_names, df_importance):
    """Analiza la distribución de features importantes para la clase 'angry'."""
    
    print("\n" + "=" * 70)
    print("🔥 ANÁLISIS DE FEATURES PARA CLASE 'ANGRY'")
    print("=" * 70)
    
    # Combinar train y test
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    
    # Separar por clase
    mask_angry = y_full == 'angry'
    X_angry = X_full[mask_angry]
    X_others = X_full[~mask_angry]
    
    print(f"\nSamples de 'angry': {len(X_angry)}")
    print(f"Samples de otras clases: {len(X_others)}")
    
    # Analizar top features
    print("\n🔬 Análisis de Top Features:")
    print("-" * 70)
    
    top_features = df_importance.head(20)['feature'].tolist() if df_importance is not None else feature_names[:20]
    
    print(f"\n{'Feature':40s} {'Angry Mean':>12s} {'Others Mean':>12s} {'Difference':>12s} {'Cohen-d':>10s}")
    print("-" * 90)
    
    feature_analysis = []
    
    for feature in top_features:
        # Estadísticas para angry
        angry_mean = X_angry[feature].mean()
        angry_std = X_angry[feature].std()
        
        # Estadísticas para otras clases
        others_mean = X_others[feature].mean()
        others_std = X_others[feature].std()
        
        # Diferencia absoluta
        diff = abs(angry_mean - others_mean)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(X_angry) - 1) * angry_std**2 + 
                              (len(X_others) - 1) * others_std**2) / 
                             (len(X_angry) + len(X_others) - 2))
        cohens_d = (angry_mean - others_mean) / pooled_std if pooled_std > 0 else 0
        
        feature_analysis.append({
            'feature': feature,
            'angry_mean': angry_mean,
            'others_mean': others_mean,
            'diff': diff,
            'cohens_d': abs(cohens_d)
        })
        
        # Marcar features con alta discriminación
        marker = "🎯" if abs(cohens_d) > 0.5 else "  "
        
        print(f"{marker} {feature:40s} {angry_mean:12.4f} {others_mean:12.4f} "
              f"{diff:12.4f} {cohens_d:10.3f}")
    
    # Resumen
    print("\n📊 Resumen de Discriminación:")
    print("-" * 50)
    high_discriminative = sum(1 for f in feature_analysis if f['cohens_d'] > 0.5)
    print(f"  Features con Cohen's d > 0.5: {high_discriminative}/{len(feature_analysis)}")
    print(f"  (Cohen's d > 0.5 indica diferencia notable entre angry y otras clases)")
    
    # Ordenar por Cohen's d
    feature_analysis.sort(key=lambda x: x['cohens_d'], reverse=True)
    
    print("\n🎯 Top 5 Features más discriminativos para 'angry':")
    print("-" * 50)
    for i, f in enumerate(feature_analysis[:5], 1):
        print(f"  {i}. {f['feature']:40s} (Cohen's d = {f['cohens_d']:.3f})")
    
    return feature_analysis


def analyze_angry_vs_each_emotion(X_train, y_train, X_test, y_test, feature_names):
    """Compara 'angry' con cada emoción específica."""
    
    print("\n" + "=" * 70)
    print("🎭 COMPARACIÓN: ANGRY vs CADA EMOCIÓN")
    print("=" * 70)
    
    # Combinar train y test
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    
    X_angry = X_full[y_full == 'angry']
    
    emotions = ['happy', 'sad', 'relax']
    
    for emotion in emotions:
        print(f"\n{'=' * 70}")
        print(f"🔥 ANGRY vs {emotion.upper()}")
        print('=' * 70)
        
        X_emotion = X_full[y_full == emotion]
        
        print(f"\nSamples: angry={len(X_angry)}, {emotion}={len(X_emotion)}")
        
        # Calcular features más diferentes
        differences = []
        
        for feature in feature_names[:30]:  # Top 30 features
            angry_mean = X_angry[feature].mean()
            emotion_mean = X_emotion[feature].mean()
            
            angry_std = X_angry[feature].std()
            emotion_std = X_emotion[feature].std()
            
            diff = abs(angry_mean - emotion_mean)
            
            # Cohen's d
            pooled_std = np.sqrt(((len(X_angry) - 1) * angry_std**2 + 
                                  (len(X_emotion) - 1) * emotion_std**2) / 
                                 (len(X_angry) + len(X_emotion) - 2))
            cohens_d = (angry_mean - emotion_mean) / pooled_std if pooled_std > 0 else 0
            
            differences.append({
                'feature': feature,
                'cohens_d': abs(cohens_d),
                'diff': diff
            })
        
        # Ordenar por Cohen's d
        differences.sort(key=lambda x: x['cohens_d'], reverse=True)
        
        print(f"\n🎯 Top 5 features que diferencian 'angry' de '{emotion}':")
        print("-" * 70)
        for i, d in enumerate(differences[:5], 1):
            print(f"  {i}. {d['feature']:40s} (Cohen's d = {d['cohens_d']:.3f})")
        
        # Contar features altamente discriminativas
        high_disc = sum(1 for d in differences if d['cohens_d'] > 0.5)
        print(f"\nFeatures con Cohen's d > 0.5: {high_disc}/{len(differences)}")
        
        if high_disc < 5:
            print(f"⚠️  ALERTA: Pocas features discriminan bien entre angry y {emotion}")
            print(f"   Esto puede explicar confusiones en la clasificación!")


def check_angry_audio_quality(X_train, y_train):
    """Verifica si hay problemas de calidad en samples de 'angry'."""
    
    print("\n" + "=" * 70)
    print("🔍 VERIFICACIÓN DE CALIDAD: SAMPLES 'ANGRY'")
    print("=" * 70)
    
    X_angry = X_train[y_train == 'angry']
    
    # Buscar outliers y valores anómalos
    print("\n1️⃣ Búsqueda de outliers en features:")
    print("-" * 50)
    
    outlier_counts = {}
    
    for col in X_angry.columns[:20]:  # Top 20 features
        # Usar IQR method
        Q1 = X_angry[col].quantile(0.25)
        Q3 = X_angry[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((X_angry[col] < lower_bound) | (X_angry[col] > upper_bound)).sum()
        
        if outliers > 0:
            outlier_counts[col] = outliers
    
    if outlier_counts:
        print(f"\nFeatures con outliers en 'angry' samples:")
        for feature, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / len(X_angry)) * 100
            print(f"  - {feature:40s}: {count}/{len(X_angry)} samples ({pct:.1f}%)")
    else:
        print("\n✅ No se detectaron outliers significativos")
    
    # Verificar NaN values
    print("\n2️⃣ Verificación de valores NaN:")
    print("-" * 50)
    nan_counts = X_angry.isnull().sum()
    nan_counts = nan_counts[nan_counts > 0]
    
    if len(nan_counts) > 0:
        print(f"\n⚠️  Se encontraron NaN values:")
        for col, count in nan_counts.items():
            print(f"  - {col}: {count} NaNs")
    else:
        print("\n✅ No hay valores NaN")
    
    # Estadísticas generales
    print("\n3️⃣ Estadísticas de varianza:")
    print("-" * 50)
    
    variances = X_angry.var().sort_values()
    
    print(f"\nFeatures con MENOR varianza en 'angry':")
    for i, (col, var) in enumerate(variances.head(5).items(), 1):
        print(f"  {i}. {col:40s}: {var:.6f}")
    
    if variances.head(1).values[0] < 0.001:
        print("\n⚠️  ALERTA: Algunas features tienen varianza muy baja")
        print("   Esto puede indicar que no aportan información para angry")


if __name__ == "__main__":
    print("🚀 Iniciando análisis de Feature Importance para 'Angry'")
    print("=" * 70)
    
    # 1. Cargar modelo y datos
    model, X_train, X_test, y_train, y_test, feature_names = load_model_and_data()
    
    # 2. Analizar feature importance global
    df_importance = analyze_random_forest_importance(model, feature_names)
    
    # 3. Analizar distribución de features para angry
    feature_analysis = analyze_angry_feature_distribution(
        X_train, y_train, X_test, y_test, feature_names, df_importance
    )
    
    # 4. Comparar angry con cada emoción
    analyze_angry_vs_each_emotion(X_train, y_train, X_test, y_test, feature_names)
    
    # 5. Verificar calidad de datos angry
    check_angry_audio_quality(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("✅ Análisis completado")
    print("=" * 70)
    
    # Guardar resultados
    if df_importance is not None:
        df_importance.to_csv('feature_importance_analysis.csv', index=False)
        print("\n💾 Feature importance guardado en: feature_importance_analysis.csv")
