"""
Feature Engineering Module - Enhanced Features for Emotion Classification
==========================================================================

Este módulo implementa feature engineering avanzado basado en análisis empírico
que identificó features con alto poder discriminativo para la clase "Angry".

Hallazgos del análisis (Script 4):
----------------------------------
Features con mejor Cohen's d para discriminar "Angry":
- Roughness_Mean: d = 0.576-0.798 (MEJOR)
- Eventdensity_Mean: d = 1.095 (EXCELENTE para angry vs sad)
- AttackTime_Mean: d = 0.919 (MUY BUENO para angry vs sad)
- Tempo_Mean: d = 0.505 (BUENO para angry vs relax)
- MFCC_Mean_6: d = 0.723 (BUENO para angry vs happy)
- MFCC_Mean_7: d = 0.739 (BUENO para angry vs happy)

Estrategia:
-----------
1. Derivar features estadísticas de las features discriminativas
2. Crear ratios y diferencias entre features relacionadas
3. Agregar features de variabilidad temporal
4. Mantener compatibilidad con pipeline existente

Autor: MLOps Team 24
Fecha: Noviembre 2025
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer que agrega features derivadas con alto poder discriminativo.
    
    Features agregadas:
    -------------------
    1. Roughness features (3 nuevas):
       - Roughness_squared
       - Roughness_log (para capturar no-linealidad)
       - Roughness_percentile_rank
    
    2. Eventdensity features (2 nuevas):
       - Eventdensity_squared
       - Eventdensity_log
    
    3. AttackTime features (2 nuevas):
       - AttackTime_squared
       - AttackTime_ratio (vs Slope)
    
    4. Tempo features (3 nuevas):
       - Tempo_squared
       - Tempo_deviation (desviación del tempo promedio 120 BPM)
       - Tempo_category (slow/medium/fast)
    
    5. MFCC features (2 nuevas):
       - MFCC_6_7_ratio
       - MFCC_6_7_interaction
    
    6. Energy features (2 nuevas):
       - RMS_Roughness_ratio
       - Energy_Attack_interaction
    
    Total: 14 nuevas features discriminativas
    
    Uso:
    ----
    >>> engineer = EnhancedFeatureEngineer()
    >>> X_enhanced = engineer.fit_transform(X)
    """
    
    def __init__(
        self,
        add_roughness_features: bool = True,
        add_eventdensity_features: bool = True,
        add_attacktime_features: bool = True,
        add_tempo_features: bool = True,
        add_mfcc_features: bool = True,
        add_interaction_features: bool = True,
        handle_missing: str = 'median',  # 'median', 'mean', 'zero'
        verbose: bool = False
    ):
        """
        Inicializa el feature engineer.
        
        Parameters:
        -----------
        add_*_features : bool
            Flags para activar/desactivar grupos de features
        handle_missing : str
            Estrategia para manejar valores NaN
        verbose : bool
            Si True, imprime logs de progreso
        """
        self.add_roughness_features = add_roughness_features
        self.add_eventdensity_features = add_eventdensity_features
        self.add_attacktime_features = add_attacktime_features
        self.add_tempo_features = add_tempo_features
        self.add_mfcc_features = add_mfcc_features
        self.add_interaction_features = add_interaction_features
        self.handle_missing = handle_missing
        self.verbose = verbose
        
        # Atributos que se calculan en fit
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None
        self.fill_values_ = {}
        self.is_fitted_ = False
    


    def fit(self, X, y=None):
        """
        Calcula estadísticas necesarias para transformación.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features originales
        y : ignored
        
        Returns:
        --------
        self
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        
        # Calcular valores para imputación de NaN (solo columnas numéricas)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if self.handle_missing == 'median':
            self.fill_values_ = X[numeric_cols].median().to_dict()
        elif self.handle_missing == 'mean':
            self.fill_values_ = X[numeric_cols].mean().to_dict()
        else:  # 'zero'
            self.fill_values_ = {col: 0 for col in numeric_cols}
        
        # Simular transformación para obtener feature_names_out
        X_temp = self._transform_internal(X.copy())
        self.feature_names_out_ = X_temp.columns.tolist()
        self.n_features_out_ = len(self.feature_names_out_)
        
        self.is_fitted_ = True
        
        if self.verbose:
            logger.info(f"EnhancedFeatureEngineer fitted:")
            logger.info(f"  Input features: {self.n_features_in_}")
            logger.info(f"  Output features: {self.n_features_out_}")
            logger.info(f"  New features added: {self.n_features_out_ - self.n_features_in_}")
        
        return self
            
    def transform(self, X):
        """
        Aplica transformaciones y agrega nuevas features.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Features originales
        
        Returns:
        --------
        X_enhanced : pd.DataFrame
            Features originales + features derivadas
        """
        if not self.is_fitted_:
            raise RuntimeError("Debe llamar fit() antes de transform()")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        
        return self._transform_internal(X.copy())
    
    def _transform_internal(self, X):
        """Implementación interna de la transformación."""
        
        # Llenar NaN values
        for col in X.columns:
            if col in self.fill_values_:
                X[col] = X[col].fillna(self.fill_values_[col])
        
        # 1. ROUGHNESS FEATURES (Cohen's d = 0.576-0.798)
        if self.add_roughness_features and '_Roughness_Mean' in X.columns:
            roughness = X['_Roughness_Mean']
            
            # Non-linear transformations
            X['Roughness_squared'] = roughness ** 2
            X['Roughness_log'] = np.log1p(np.abs(roughness))  # log(1 + |x|) para evitar log(0)
            
            # Percentile-based feature
            X['Roughness_percentile'] = roughness.rank(pct=True)
            
            if self.verbose:
                logger.info("✅ Added 3 Roughness features")
        
        # 2. EVENTDENSITY FEATURES (Cohen's d = 1.095 para angry vs sad)
        if self.add_eventdensity_features and '_Eventdensity_Mean' in X.columns:
            eventdensity = X['_Eventdensity_Mean']
            
            X['Eventdensity_squared'] = eventdensity ** 2
            X['Eventdensity_log'] = np.log1p(np.abs(eventdensity))
            
            if self.verbose:
                logger.info("✅ Added 2 Eventdensity features")
        
        # 3. ATTACKTIME FEATURES (Cohen's d = 0.919)
        if self.add_attacktime_features and '_AttackTime_Mean' in X.columns:
            attacktime = X['_AttackTime_Mean']
            
            X['AttackTime_squared'] = attacktime ** 2
            
            # Ratio con Slope si existe
            if '_AttackTime_Slope' in X.columns:
                slope = X['_AttackTime_Slope']
                # Evitar división por cero
                X['AttackTime_Slope_ratio'] = attacktime / (np.abs(slope) + 1e-10)
            
            if self.verbose:
                logger.info("✅ Added 2 AttackTime features")
        
        # 4. TEMPO FEATURES (Cohen's d = 0.505)
        if self.add_tempo_features and '_Tempo_Mean' in X.columns:
            tempo = X['_Tempo_Mean']
            
            X['Tempo_squared'] = tempo ** 2
            
            # Desviación del tempo "normal" (120 BPM)
            X['Tempo_deviation'] = np.abs(tempo - 120)
            
            # Categorización: slow (<100), medium (100-140), fast (>140)
            X['Tempo_is_fast'] = (tempo > 140).astype(int)
            X['Tempo_is_slow'] = (tempo < 100).astype(int)
            
            if self.verbose:
                logger.info("✅ Added 4 Tempo features")
        
        # 5. MFCC FEATURES (Cohen's d = 0.723-0.739 para angry vs happy)
        if self.add_mfcc_features:
            if '_MFCC_Mean_6' in X.columns and '_MFCC_Mean_7' in X.columns:
                mfcc6 = X['_MFCC_Mean_6']
                mfcc7 = X['_MFCC_Mean_7']
                
                # Ratio
                X['MFCC_6_7_ratio'] = mfcc6 / (np.abs(mfcc7) + 1e-10)
                
                # Interaction term
                X['MFCC_6_7_interaction'] = mfcc6 * mfcc7
                
                if self.verbose:
                    logger.info("✅ Added 2 MFCC interaction features")
        
        # 6. INTERACTION FEATURES (combinaciones de las mejores)
        if self.add_interaction_features:
            # RMS-Roughness interaction (energía + textura)
            if '_RMSenergy_Mean' in X.columns and '_Roughness_Mean' in X.columns:
                rms = X['_RMSenergy_Mean']
                roughness = X['_Roughness_Mean']
                
                X['RMS_Roughness_ratio'] = rms / (np.abs(roughness) + 1e-10)
                X['RMS_Roughness_product'] = rms * roughness
            
            # Energy-Attack interaction (energía + ataque)
            if '_RMSenergy_Mean' in X.columns and '_AttackTime_Mean' in X.columns:
                rms = X['_RMSenergy_Mean']
                attack = X['_AttackTime_Mean']
                
                X['Energy_Attack_interaction'] = rms * attack
            
            if self.verbose:
                logger.info("✅ Added 3 interaction features")
        
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Retorna nombres de features de salida."""
        if not self.is_fitted_:
            raise RuntimeError("Debe llamar fit() antes de get_feature_names_out()")
        return self.feature_names_out_
    
    def get_new_feature_names(self):
        """Retorna solo los nombres de las features NUEVAS agregadas."""
        if not self.is_fitted_:
            raise RuntimeError("Debe llamar fit() antes de get_new_feature_names()")
        
        original_features = set(self.feature_names_in_)
        all_features = set(self.feature_names_out_)
        new_features = list(all_features - original_features)
        
        return sorted(new_features)


def apply_feature_engineering(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Función de conveniencia para aplicar feature engineering a un DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset con features originales (debe incluir columna 'Class')
    verbose : bool
        Si True, imprime información de progreso
    
    Returns:
    --------
    df_enhanced : pd.DataFrame
        Dataset con features originales + features derivadas
    
    Usage:
    ------
    >>> df = pd.read_csv("data/processed/turkish_music_emotion_v2_cleaned_full.csv")
    >>> df_enhanced = apply_feature_engineering(df)
    >>> df_enhanced.to_csv("data/processed/turkish_music_emotion_v2_ENHANCED.csv", index=False)
    """
    
    if verbose:
        print("=" * 70)
        print("🔧 FEATURE ENGINEERING: Agregando Features Discriminativas")
        print("=" * 70)
        print(f"\n📊 Dataset original:")
        print(f"   Filas: {len(df)}")
        print(f"   Features: {len(df.columns) - 1}")  # -1 por 'Class'
    
    # Separar Class y features
    if 'Class' not in df.columns:
        raise ValueError("DataFrame debe contener columna 'Class'")
    
    y = df['Class']
    X = df.drop('Class', axis=1)
    
    # Aplicar feature engineering
    engineer = EnhancedFeatureEngineer(verbose=verbose)
    X_enhanced = engineer.fit_transform(X)
    
    # Recombinar con Class
    df_enhanced = X_enhanced.copy()
    df_enhanced['Class'] = y
    
    if verbose:
        new_features = engineer.get_new_feature_names()
        print(f"\n📊 Dataset mejorado:")
        print(f"   Filas: {len(df_enhanced)}")
        print(f"   Features: {len(df_enhanced.columns) - 1}")
        print(f"   Nuevas features: {len(new_features)}")
        
        print(f"\n✨ Features agregadas ({len(new_features)}):")
        for i, feat in enumerate(new_features, 1):
            print(f"   {i:2d}. {feat}")
        
        print("\n" + "=" * 70)
        print("✅ Feature Engineering completado")
        print("=" * 70)
    
    return df_enhanced


# ============================================================================
# ANÁLISIS DE FEATURES (para validar que las nuevas features ayudan)
# ============================================================================

def analyze_new_features(df_original: pd.DataFrame, df_enhanced: pd.DataFrame):
    """
    Analiza el impacto de las nuevas features en la discriminabilidad.
    
    Parameters:
    -----------
    df_original : pd.DataFrame
        Dataset original
    df_enhanced : pd.DataFrame
        Dataset con features agregadas
    """
    from scipy.stats import ttest_ind
    
    print("=" * 70)
    print("📊 ANÁLISIS DE IMPACTO: Nuevas Features")
    print("=" * 70)
    
    # Identificar nuevas features
    original_cols = set(df_original.columns) - {'Class'}
    enhanced_cols = set(df_enhanced.columns) - {'Class'}
    new_features = sorted(enhanced_cols - original_cols)
    
    print(f"\n🆕 Analizando {len(new_features)} nuevas features...")
    
    # Calcular Cohen's d para cada nueva feature
    results = []
    
    for feature in new_features:
        # Separar por clase
        angry = df_enhanced[df_enhanced['Class'] == 'angry'][feature]
        others = df_enhanced[df_enhanced['Class'] != 'angry'][feature]
        
        # Remover NaN
        angry = angry.dropna()
        others = others.dropna()
        
        if len(angry) > 0 and len(others) > 0:
            # Cohen's d
            mean_diff = angry.mean() - others.mean()
            pooled_std = np.sqrt(
                ((len(angry) - 1) * angry.std()**2 + 
                 (len(others) - 1) * others.std()**2) / 
                (len(angry) + len(others) - 2)
            )
            cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
            
            # T-test
            t_stat, p_value = ttest_ind(angry, others, equal_var=False)
            
            results.append({
                'feature': feature,
                'cohens_d': cohens_d,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
    
    # Ordenar por Cohen's d
    results.sort(key=lambda x: x['cohens_d'], reverse=True)
    
    print(f"\n🎯 Top 10 Nuevas Features Más Discriminativas:")
    print("-" * 70)
    print(f"{'Feature':40s} {'Cohen-d':>10s} {'P-value':>10s} {'Sig':>5s}")
    print("-" * 70)
    
    for i, r in enumerate(results[:10], 1):
        marker = "✅" if r['cohens_d'] > 0.5 else "⚠️" if r['cohens_d'] > 0.3 else "  "
        sig_marker = "***" if r['significant'] else "   "
        print(f"{marker} {r['feature']:40s} {r['cohens_d']:10.3f} {r['p_value']:10.4f} {sig_marker}")
    
    # Resumen
    high_d = sum(1 for r in results if r['cohens_d'] > 0.5)
    medium_d = sum(1 for r in results if 0.3 < r['cohens_d'] <= 0.5)
    
    print(f"\n📊 Resumen:")
    print(f"   Features con Cohen's d > 0.5 (alta discriminación): {high_d}/{len(results)}")
    print(f"   Features con Cohen's d 0.3-0.5 (media discriminación): {medium_d}/{len(results)}")
    print(f"   Features estadísticamente significativas (p < 0.05): {sum(1 for r in results if r['significant'])}/{len(results)}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    # Test del módulo
    print("🧪 Test del módulo de Feature Engineering")
    print("=" * 70)
    
    # Crear datos dummy
    np.random.seed(42)
    n_samples = 100
    
    df_test = pd.DataFrame({
        'Class': np.random.choice(['angry', 'happy', 'sad', 'relax'], n_samples),
        '_Roughness_Mean': np.random.randn(n_samples) * 100 + 500,
        '_Eventdensity_Mean': np.random.randn(n_samples) * 10 + 50,
        '_AttackTime_Mean': np.random.randn(n_samples) * 0.1 + 0.5,
        '_AttackTime_Slope': np.random.randn(n_samples) * 0.01,
        '_Tempo_Mean': np.random.randn(n_samples) * 20 + 120,
        '_MFCC_Mean_6': np.random.randn(n_samples),
        '_MFCC_Mean_7': np.random.randn(n_samples),
        '_RMSenergy_Mean': np.random.randn(n_samples) * 0.1 + 0.5,
    })
    
    # Aplicar feature engineering
    df_enhanced = apply_feature_engineering(df_test)
    
    print(f"\n✅ Test completado exitosamente!")
    print(f"   Dataset original: {df_test.shape}")
    print(f"   Dataset mejorado: {df_enhanced.shape}")
