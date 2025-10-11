"""
Scripts para cargar y procesar datasets
"""
import pandas as pd
from pathlib import Path
from acoustic_ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TURKISH_MODIFIED, TURKISH_ORIGINAL

def load_raw_data(filename: str = "acoustic_features.csv") -> pd.DataFrame:
    """Carga datos crudos desde data/raw/"""
    filepath = RAW_DATA_DIR / filename
    return pd.read_csv(filepath)

def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Guarda datos procesados en data/processed/"""
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"✅ Datos guardados en {filepath}")

def load_turkish_original():
    """Carga el dataset ORIGINAL de música turca"""
    filepath = RAW_DATA_DIR / TURKISH_ORIGINAL
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Dataset no encontrado: {filepath}\n💡 Ejecuta: dvc pull")
    print(f"📂 Cargando: {filepath.name}")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df

def load_turkish_modified():
    """Carga el dataset MODIFICADO de música turca"""
    filepath = RAW_DATA_DIR / TURKISH_MODIFIED
    if not filepath.exists():
        raise FileNotFoundError(f"❌ Dataset no encontrado: {filepath}\n💡 Ejecuta: dvc pull")
    print(f"📂 Cargando: {filepath.name}")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df

def get_dataset_info(df: pd.DataFrame) -> None:
    """Muestra información resumida del dataset"""
    print("📊 Información del Dataset")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"Valores nulos: {df.isnull().sum().sum():,}")
    print(f"\n📋 Columnas ({len(df.columns)}):")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        print(f"   • {col:30s} | {str(dtype):10s} | Nulls: {nulls:,}")
