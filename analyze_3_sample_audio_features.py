"""
Script 3: Sample Audio Feature Analysis
========================================
Extrae features de los audios en sample_audio/ y los compara
con la distribución del training set.

Usage:
    python3 analyze_3_sample_audio_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import mlflow

# Importar AudioFeatureExtractor
try:
    from acoustic_ml.features import AudioFeatureExtractor
except ImportError:
    print("❌ ERROR: No se pudo importar acoustic_ml")
    print("Asegúrate de estar en el directorio correcto y tener el paquete instalado")
    sys.exit(1)


def load_training_features():
    """Carga features del training set."""
    
    print("📂 Cargando training set...")
    data_path = Path("data/processed/turkish_music_emotion_v2_cleaned_full.csv")
    
    if not data_path.exists():
        print(f"❌ ERROR: No se encontró {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    df['Class'] = df['Class'].str.strip().str.lower()
    
    print(f"  ✅ Cargado: {len(df)} samples")
    return df


def extract_sample_audio_features():
    """Extrae features de todos los audios en sample_audio/."""
    
    print("\n" + "=" * 70)
    print("🎵 EXTRAYENDO FEATURES DE SAMPLE AUDIO")
    print("=" * 70)
    
    sample_audio_dir = Path("turkish_music_app/assets/sample_audio")
    
    if not sample_audio_dir.exists():
        print(f"❌ ERROR: No se encontró {sample_audio_dir}")
        sys.exit(1)
    
    # Inicializar extractor
    extractor = AudioFeatureExtractor()
    
    results = []
    
    # Iterar por cada emoción
    for emotion_dir in sorted(sample_audio_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue
        
        emotion = emotion_dir.name
        print(f"\n📁 Procesando: {emotion}/")
        print("-" * 50)
        
        # Procesar cada audio
        for audio_file in sorted(emotion_dir.glob("*.mp3")):
            print(f"  🎵 {audio_file.name}... ", end="")
            
            try:
                # Extraer features
                features = extractor.extract_features(str(audio_file))
                
                # Agregar metadata
                features['emotion'] = emotion
                features['filename'] = audio_file.name
                features['filepath'] = str(audio_file)
                
                results.append(features)
                print("✅")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Convertir a DataFrame
    df_samples = pd.DataFrame(results)
    
    print(f"\n✅ Extraídos features de {len(df_samples)} audios")
    return df_samples


def compare_with_training(df_training, df_samples):
    """Compara features de sample_audio con training set."""
    
    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN: SAMPLE AUDIO vs TRAINING SET")
    print("=" * 70)
    
    # Separar metadata de features
    feature_cols = [col for col in df_samples.columns 
                   if col not in ['emotion', 'filename', 'filepath']]
    
    print(f"\nFeatures a comparar: {len(feature_cols)}")
    
    # Análisis por emoción
    for emotion in sorted(df_samples['emotion'].unique()):
        print(f"\n{'=' * 70}")
        print(f"🎭 EMOCIÓN: {emotion.upper()}")
        print('=' * 70)
        
        # Samples de sample_audio/
        df_sample_emotion = df_samples[df_samples['emotion'] == emotion]
        print(f"\n📂 Sample Audio: {len(df_sample_emotion)} archivos")
        for _, row in df_sample_emotion.iterrows():
            print(f"  - {row['filename']}")
        
        # Training set
        df_train_emotion = df_training[df_training['Class'] == emotion]
        print(f"\n📊 Training Set: {len(df_train_emotion)} samples")
        
        if len(df_train_emotion) == 0:
            print(f"  ⚠️  No hay samples de '{emotion}' en training set!")
            continue
        
        # Comparar estadísticas de features
        print(f"\n🔬 Comparación de Features:")
        print("-" * 70)
        
        # Obtener features del training (excluir columna Class)
        train_features = df_train_emotion.drop('Class', axis=1, errors='ignore')
        sample_features = df_sample_emotion[feature_cols]
        
        # Calcular estadísticas
        train_means = train_features.mean()
        sample_means = sample_features.mean()
        
        train_stds = train_features.std()
        sample_stds = sample_features.std()
        
        # Calcular distancias
        print(f"\n  {'Feature':30s} {'Train Mean':>12s} {'Sample Mean':>12s} {'Diff':>10s} {'Z-score':>10s}")
        print("  " + "-" * 80)
        
        differences = []
        
        for feature in feature_cols[:15]:  # Top 15 features con mayor diferencia
            train_mean = train_means[feature]
            sample_mean = sample_means[feature]
            train_std = train_stds[feature]
            
            diff = abs(sample_mean - train_mean)
            
            # Z-score: cuántas desviaciones estándar está el sample del train
            z_score = diff / train_std if train_std > 0 else 0
            
            differences.append({
                'feature': feature,
                'train_mean': train_mean,
                'sample_mean': sample_mean,
                'diff': diff,
                'z_score': z_score
            })
        
        # Ordenar por z-score (mayor diferencia primero)
        differences.sort(key=lambda x: x['z_score'], reverse=True)
        
        for d in differences[:10]:  # Top 10
            marker = "⚠️ " if d['z_score'] > 2.0 else "   "
            print(f"{marker}{d['feature']:30s} {d['train_mean']:12.4f} {d['sample_mean']:12.4f} "
                  f"{d['diff']:10.4f} {d['z_score']:10.2f}")
        
        # Resumen
        high_z_count = sum(1 for d in differences if d['z_score'] > 2.0)
        avg_z = np.mean([d['z_score'] for d in differences])
        
        print(f"\n  📊 Resumen:")
        print(f"    - Features con Z-score > 2.0: {high_z_count}/{len(differences)}")
        print(f"    - Z-score promedio: {avg_z:.2f}")
        
        if high_z_count > 5:
            print(f"    ⚠️  ALERTA: Muchas features tienen distribución diferente!")
            print(f"    Los audios de sample_audio/{emotion}/ pueden NO ser representativos del training set")
        else:
            print(f"    ✅ Las features son razonablemente similares al training set")


def test_predictions_with_model(df_samples):
    """Prueba predicciones del modelo en sample audios."""
    
    print("\n" + "=" * 70)
    print("🎯 PREDICCIONES DEL MODELO EN SAMPLE AUDIO")
    print("=" * 70)
    
    try:
        # Cargar modelo
        run_id = "eb05c7698f12499b86ed35ca6efc15a7"
        model_uri = f"runs:/{run_id}/model"
        mlflow.set_tracking_uri("file:./mlruns")
        model = mlflow.sklearn.load_model(model_uri)
        print(f"\n✅ Modelo cargado: {run_id}")
    except Exception as e:
        print(f"\n❌ No se pudo cargar modelo: {e}")
        return
    
    # Preparar features para predicción
    feature_cols = [col for col in df_samples.columns 
                   if col not in ['emotion', 'filename', 'filepath']]
    X_samples = df_samples[feature_cols]
    
    # Hacer predicciones
    try:
        predictions = model.predict(X_samples)
        probabilities = model.predict_proba(X_samples)
        
        print("\n" + "-" * 70)
        print(f"{'Archivo':40s} {'Real':10s} {'Predicción':10s} {'Correcto':10s} {'Confianza':>10s}")
        print("-" * 70)
        
        correct = 0
        total = 0
        
        for i, row in df_samples.iterrows():
            true_emotion = row['emotion']
            pred_emotion = predictions[i]
            is_correct = true_emotion == pred_emotion
            max_prob = probabilities[i].max()
            
            marker = "✅" if is_correct else "❌"
            
            print(f"{row['filename']:40s} {true_emotion:10s} {pred_emotion:10s} "
                  f"{marker:10s} {max_prob:10.1%}")
            
            if is_correct:
                correct += 1
            total += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print("-" * 70)
        print(f"Accuracy en sample_audio: {correct}/{total} ({accuracy:.1f}%)")
        
        # Análisis por emoción
        print("\n📊 Accuracy por emoción:")
        for emotion in sorted(df_samples['emotion'].unique()):
            mask = df_samples['emotion'] == emotion
            correct_emotion = sum((df_samples[mask]['emotion'].values == predictions[mask]))
            total_emotion = sum(mask)
            acc = (correct_emotion / total_emotion * 100) if total_emotion > 0 else 0
            print(f"  {emotion:10s}: {correct_emotion}/{total_emotion} ({acc:.1f}%)")
        
    except Exception as e:
        print(f"\n❌ Error en predicciones: {e}")


if __name__ == "__main__":
    print("🚀 Iniciando análisis de Sample Audio")
    print("=" * 70)
    
    # 1. Cargar training set
    df_training = load_training_features()
    
    # 2. Extraer features de sample_audio
    df_samples = extract_sample_audio_features()
    
    # 3. Comparar distribuciones
    compare_with_training(df_training, df_samples)
    
    # 4. Probar predicciones
    test_predictions_with_model(df_samples)
    
    # 5. Guardar resultados
    output_file = "sample_audio_features_analysis.csv"
    df_samples.to_csv(output_file, index=False)
    print(f"\n💾 Features guardadas en: {output_file}")
    
    print("\n" + "=" * 70)
    print("✅ Análisis completado")
    print("=" * 70)
