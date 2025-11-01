"""
Script 2: Confusion Matrix & Error Pattern Analysis
====================================================
Genera confusion matrix detallada y analiza patrones de error,
especialmente para la clase "Angry".

Usage:
    python3 analyze_2_confusion_matrix.py
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys
import pickle

# Setup MLflow
mlflow.set_tracking_uri("file:./mlruns")

def load_model_and_data(run_id="eb05c7698f12499b86ed35ca6efc15a7"):
    """Carga el modelo desde MLflow y prepara datos de test."""
    
    print("📂 Cargando modelo y datos...")
    
    # Cargar dataset
    data_path = Path("data/processed/turkish_music_emotion_v2_cleaned_full.csv")
    if not data_path.exists():
        print(f"❌ ERROR: No se encontró {data_path}")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    df['Class'] = df['Class'].str.strip().str.lower()
    
    # Separar features y labels
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Cargar splits del modelo (si están guardados en MLflow)
    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Intentar descargar splits guardados
        artifact_path = f"mlruns/{run.info.experiment_id}/{run_id}/artifacts"
        
        print(f"  ✅ Modelo encontrado: {run_id}")
        print(f"  📊 Test accuracy: {run.data.metrics.get('test_accuracy', 'N/A')}")
        
    except Exception as e:
        print(f"  ⚠️  No se pudo acceder a MLflow: {e}")
    
    # Para este análisis, vamos a usar train_test_split con mismo random_state
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Cargar el modelo de MLflow
    try:
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"  ✅ Modelo cargado desde MLflow")
    except Exception as e:
        print(f"  ❌ ERROR al cargar modelo: {e}")
        sys.exit(1)
    
    return model, X_test, y_test, X_train, y_train


def analyze_confusion_matrix(model, X_test, y_test):
    """Genera y analiza confusion matrix detallada."""
    
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 70)
    print("🎯 CONFUSION MATRIX ANALYSIS")
    print("=" * 70)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    emotions = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=emotions)
    
    print("\n1️⃣ Confusion Matrix (Valores absolutos):")
    print("-" * 50)
    
    # Header
    print(f"\n{'':10s}", end="")
    for emotion in emotions:
        print(f"{emotion:10s}", end="")
    print()
    print("-" * (10 + 10 * len(emotions)))
    
    # Rows
    for i, true_emotion in enumerate(emotions):
        print(f"{true_emotion:10s}", end="")
        for j, pred_emotion in enumerate(emotions):
            value = cm[i, j]
            if i == j:  # Diagonal (correctos)
                print(f"✅ {value:7d}", end="")
            else:
                if value > 0:
                    print(f"❌ {value:7d}", end="")
                else:
                    print(f"   {value:7d}", end="")
        print()
    
    # Confusion matrix normalizada
    print("\n2️⃣ Confusion Matrix (Normalizada por fila - % de la clase real):")
    print("-" * 50)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Header
    print(f"\n{'':10s}", end="")
    for emotion in emotions:
        print(f"{emotion:10s}", end="")
    print()
    print("-" * (10 + 10 * len(emotions)))
    
    # Rows
    for i, true_emotion in enumerate(emotions):
        print(f"{true_emotion:10s}", end="")
        for j, pred_emotion in enumerate(emotions):
            value = cm_normalized[i, j]
            if i == j:  # Diagonal
                print(f"{value:9.1%}", end="")
            else:
                if value > 0.05:  # Más de 5% de error
                    print(f"❗{value:8.1%}", end="")
                else:
                    print(f"{value:9.1%}", end="")
        print()
    
    # Análisis de errores para cada clase
    print("\n3️⃣ Análisis de errores por clase:")
    print("-" * 50)
    
    for i, true_emotion in enumerate(emotions):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n  {true_emotion.upper()}:")
        print(f"    Total samples: {total}")
        print(f"    Correctos: {correct} ({accuracy:.1%})")
        print(f"    Incorrectos: {total - correct} ({1-accuracy:.1%})")
        
        if total > correct:
            # Mostrar confusiones más comunes
            errors = [(emotions[j], cm[i, j]) for j in range(len(emotions)) if i != j and cm[i, j] > 0]
            errors.sort(key=lambda x: x[1], reverse=True)
            
            print(f"    Confusiones más comunes:")
            for confused_with, count in errors[:3]:
                pct = (count / total) * 100
                print(f"      → Confundido con '{confused_with}': {count} veces ({pct:.1f}%)")
    
    # Focus en ANGRY
    print("\n" + "=" * 70)
    print("🔥 ANÁLISIS ESPECÍFICO DE 'ANGRY'")
    print("=" * 70)
    
    angry_idx = emotions.index('angry')
    angry_total = cm[angry_idx].sum()
    angry_correct = cm[angry_idx, angry_idx]
    
    print(f"\nTotal samples de 'angry' en test: {angry_total}")
    print(f"Correctamente clasificados: {angry_correct} ({angry_correct/angry_total:.1%})")
    print(f"Incorrectamente clasificados: {angry_total - angry_correct} ({(angry_total-angry_correct)/angry_total:.1%})")
    
    print(f"\n¿Cómo se confunde 'angry'?")
    print("-" * 50)
    for j, emotion in enumerate(emotions):
        if j != angry_idx and cm[angry_idx, j] > 0:
            count = cm[angry_idx, j]
            pct = (count / angry_total) * 100
            print(f"  '{angry_idx} → '{emotion}': {count} samples ({pct:.1f}%)")
    
    print(f"\n¿Qué se confunde COMO 'angry'?")
    print("-" * 50)
    for i, emotion in enumerate(emotions):
        if i != angry_idx and cm[i, angry_idx] > 0:
            count = cm[i, angry_idx]
            total_i = cm[i].sum()
            pct = (count / total_i) * 100
            print(f"  '{emotion}' → 'angry': {count} samples ({pct:.1f}%)")
    
    # Classification report
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION REPORT COMPLETO")
    print("=" * 70)
    print("\n", classification_report(y_test, y_pred, labels=emotions, digits=3))
    
    # Intentar crear visualización
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_angry_analysis.png', dpi=150)
        print("\n✅ Gráfico guardado: confusion_matrix_angry_analysis.png")
    except Exception as e:
        print(f"\n⚠️  No se pudo crear gráfico: {e}")
    
    return cm, cm_normalized, emotions


if __name__ == "__main__":
    # Cargar modelo y datos
    model, X_test, y_test, X_train, y_train = load_model_and_data()
    
    # Analizar confusion matrix
    cm, cm_norm, emotions = analyze_confusion_matrix(model, X_test, y_test)
    
    print("\n" + "=" * 70)
    print("✅ Análisis completado")
    print("=" * 70)
