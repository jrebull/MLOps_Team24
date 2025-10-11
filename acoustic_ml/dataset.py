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

def load_turkish_cleaned():
    """
    Carga el dataset LIMPIO de música turca desde data/processed/
    
    Returns:
        pd.DataFrame: Dataset limpio y procesado
    """
    filepath = PROCESSED_DATA_DIR / 'turkish_music_emotion_cleaned.csv'
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"❌ Dataset limpio no encontrado: {filepath}\n"
            f"💡 Ejecuta primero el notebook de limpieza y luego: dvc pull"
        )
    
    print(f"📂 Cargando dataset limpio: {filepath.name}")
    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    
    return df

def load_processed_data(version: str = "v2") -> pd.DataFrame:
    """
    Carga el dataset procesado y versionado desde data/processed/
    
    Args:
        version (str): Versión del dataset a cargar (default: "v2")
    
    Returns:
        pd.DataFrame: Dataset procesado y limpio con la versión especificada
    """
    filename = f"turkish_music_emotion_{version}_cleaned_full.csv"
    filepath = PROCESSED_DATA_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"❌ Dataset no encontrado: {filepath}\n"
            f"💡 Asegúrate de haber ejecutado el notebook de limpieza y versionado primero.\n"
            f"   Luego ejecuta: dvc pull"
        )
    
    print(f"📂 Cargando dataset procesado ({version}): {filepath.name}")
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

# --- Nueva Función ---
def load_train_test_split():
    """
    Carga los conjuntos de entrenamiento y prueba procesados.
    
    Returns:
        tuple: Una tupla conteniendo (X_train, X_test, y_train, y_test)
    """
    # Definir rutas a los archivos usando la constante del directorio de datos procesados
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    X_test_path = PROCESSED_DATA_DIR / "X_test.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    
    # Comprobar si los archivos existen antes de intentar cargarlos
    required_files = [X_train_path, X_test_path, y_train_path, y_test_path]
    if not all(f.exists() for f in required_files):
        raise FileNotFoundError(
            f"❌ No se encontraron los archivos de train/test en {PROCESSED_DATA_DIR}\n"
            f"💡 Asegúrate de haber ejecutado el notebook de división de datos (split) primero."
        )
        
    print("📂 Cargando conjuntos de entrenamiento y prueba...")
    
    # Cargar los dataframes desde los archivos CSV
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    # Cargar los targets y extraer la columna 'Class' para obtener una Serie de pandas
    y_train = pd.read_csv(y_train_path)['Class']
    y_test = pd.read_csv(y_test_path)['Class']
    
    print("✅ Datasets de train/test cargados exitosamente:")
    print(f"   • X_train: {X_train.shape}")
    print(f"   • X_test:  {X_test.shape}")
    print(f"   • y_train: {y_train.shape}")
    print(f"   • y_test:  {y_test.shape}")
    
    return X_train, X_test, y_train, y_test