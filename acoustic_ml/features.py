"""
Feature engineering para características acústicas
"""
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features adicionales a partir del dataset
    
    Args:
        df: DataFrame con datos originales
        
    Returns:
        DataFrame con features adicionales
    """
    # Aquí va tu lógica de feature engineering
    df_features = df.copy()
    
    # Ejemplo: agregar features derivadas
    # df_features['feature_ratio'] = df_features['col1'] / df_features['col2']
    
    return df_features


def select_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Selecciona un subconjunto de features
    
    Args:
        df: DataFrame con todas las features
        feature_list: Lista de nombres de columnas a seleccionar
        
    Returns:
        DataFrame con features seleccionadas
    """
    return df[feature_list]
