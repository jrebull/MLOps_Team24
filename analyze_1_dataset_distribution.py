"""
Script 1: Dataset Distribution Analysis
========================================
Analiza la distribución de clases, balance, y estadísticas básicas.

Usage:
    python3 analyze_1_dataset_distribution.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def analyze_dataset_distribution():
    """Análisis completo de distribución del dataset."""
    
    # Paths
    data_path = Path("data/processed/turkish_music_emotion_v2_cleaned_full.csv")
    
    if not data_path.exists():
        print(f"❌ ERROR: No se encontró {data_path}")
        sys.exit(1)
    
    # Cargar dataset
    print("📂 Cargando dataset...")
    df = pd.read_csv(data_path)
    
    # Normalizar labels
    df['Class'] = df['Class'].str.strip().str.lower()
    
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DE DISTRIBUCIÓN DE CLASES")
    print("=" * 70)
    
    # Distribución básica
    class_counts = df['Class'].value_counts().sort_index()
    print("\n1️⃣ Conteo de samples por clase:")
    print("-" * 50)
    for emotion, count in class_counts.items():
        pct = (count / len(df)) * 100
        bar = "█" * int(pct / 2)
        print(f"  {emotion:10s}: {count:3d} samples ({pct:5.1f}%) {bar}")
    
    print(f"\n📊 Total samples: {len(df)}")
    print(f"📊 Total features: {len(df.columns) - 1}")
    
    # Balance metrics
    print("\n2️⃣ Métricas de balance:")
    print("-" * 50)
    min_class = class_counts.min()
    max_class = class_counts.max()
    balance_ratio = min_class / max_class
    
    print(f"  Clase más grande: {class_counts.idxmax()} ({max_class} samples)")
    print(f"  Clase más pequeña: {class_counts.idxmin()} ({min_class} samples)")
    print(f"  Balance ratio (min/max): {balance_ratio:.2%}")
    
    if balance_ratio < 0.70:
        print(f"  ⚠️  ADVERTENCIA: Dataset desbalanceado (ratio < 70%)")
    else:
        print(f"  ✅ Dataset razonablemente balanceado")
    
    # Estadísticas por clase
    print("\n3️⃣ Estadísticas de features por clase:")
    print("-" * 50)
    
    feature_cols = [col for col in df.columns if col != 'Class']
    
    # Filtrar NaN values antes de ordenar
    unique_classes = df['Class'].dropna().unique()
    
    # Identificar columnas numéricas solamente
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in feature_cols if col not in numeric_cols]
    
    if non_numeric_cols:
        print(f"\n⚠️  ADVERTENCIA: Se encontraron {len(non_numeric_cols)} columnas NO NUMÉRICAS:")
        for col in non_numeric_cols[:5]:
            print(f"    - {col}")
        if len(non_numeric_cols) > 5:
            print(f"    ... y {len(non_numeric_cols) - 5} más")
        print(f"\n  Usando solo {len(numeric_cols)} columnas numéricas para estadísticas")
    
    for emotion in sorted(unique_classes):
        subset = df[df['Class'] == emotion][numeric_cols]
        print(f"\n  {emotion.upper()}:")
        print(f"    Samples: {len(subset)}")
        if len(numeric_cols) > 0:
            print(f"    Features mean: {subset.mean().mean():.4f}")
            print(f"    Features std: {subset.std().mean():.4f}")
            print(f"    Features median: {subset.median().mean():.4f}")
            print(f"    Missing values: {subset.isnull().sum().sum()}")
        else:
            print(f"    ⚠️  No hay columnas numéricas para calcular estadísticas")
    
    # Verificar NaN values
    print("\n4️⃣ Verificación de integridad de datos:")
    print("-" * 50)
    total_nans = df[numeric_cols].isnull().sum().sum() if numeric_cols else 0
    if total_nans > 0:
        print(f"  ⚠️  Se encontraron {total_nans} valores NaN")
    else:
        print(f"  ✅ No hay valores NaN en features numéricas")
    
    # Duplicados
    duplicates = df.duplicated().sum()
    print(f"\n  Filas duplicadas: {duplicates}")
    if duplicates > 0:
        print(f"  ⚠️  Se encontraron {duplicates} filas duplicadas")
    else:
        print(f"  ✅ No hay filas duplicadas")
    
    print("\n" + "=" * 70)
    print("✅ Análisis completado")
    print("=" * 70)
    
    return df, class_counts

if __name__ == "__main__":
    df, class_counts = analyze_dataset_distribution()
