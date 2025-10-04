"""
Scripts para cargar y procesar datasets
"""
import pandas as pd
from pathlib import Path
from acoustic_ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(filename: str = "acoustic_features.csv") -> pd.DataFrame:
    """
    Carga datos crudos desde data/raw/
    
    Args:
        filename: Nombre del archivo a cargar
        
    Returns:
        DataFrame con los datos crudos
    """
    filepath = RAW_DATA_DIR / filename
    return pd.read_csv(filepath)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda datos procesados en data/processed/
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo de salida
    """
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"✅ Datos guardados en {filepath}")
