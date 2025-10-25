#!/usr/bin/env python3
"""
Análisis de Outliers - Turkish Music Emotion Dataset

Este script analiza outliers en el dataset para tomar decisiones
informadas sobre su manejo en el pipeline de ML.

Autor: MLOps Team 24
Fecha: Octubre 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

print("=" * 70)
print("📊 ANÁLISIS DE OUTLIERS - Turkish Music Emotion Dataset")
print("=" * 70)

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ============================================================================
# 1. CARGAR DATOS
# ============================================================================
print("\n[1/5] Cargando dataset...")

data_dir = Path("data/processed")
if (data_dir / "X_train.csv").exists():
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    print(f"✅ Datos cargados: {X_train.shape}")
else:
    print("❌ No se encontraron datos procesados")
    exit(1)

# ============================================================================
# 2. DETECCIÓN DE OUTLIERS - IQR METHOD
# ============================================================================
print("\n[2/5] Detectando outliers (método IQR)...")

def detect_outliers_iqr(df, factor=1.5):
    """Detecta outliers usando el método IQR."""
    outliers_per_feature = {}
    outlier_rows = set()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_indices = df[outlier_mask].index.tolist()
        
        outliers_per_feature[col] = {
            'count': len(outlier_indices),
            'percentage': (len(outlier_indices) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'indices': outlier_indices
        }
        
        outlier_rows.update(outlier_indices)
    
    return outliers_per_feature, outlier_rows

outliers_info, outlier_rows = detect_outliers_iqr(X_train, factor=1.5)

# Estadísticas generales
total_features = len(X_train.columns)
features_with_outliers = sum(1 for v in outliers_info.values() if v['count'] > 0)
total_outlier_rows = len(outlier_rows)
outlier_percentage = (total_outlier_rows / len(X_train)) * 100

print(f"   Features totales: {total_features}")
print(f"   Features con outliers: {features_with_outliers}")
print(f"   Filas con al menos 1 outlier: {total_outlier_rows} ({outlier_percentage:.1f}%)")

# ============================================================================
# 3. ANÁLISIS POR FEATURE
# ============================================================================
print("\n[3/5] Analizando distribución de outliers por feature...")

# Ordenar features por cantidad de outliers
sorted_features = sorted(outliers_info.items(), 
                         key=lambda x: x[1]['count'], 
                         reverse=True)

print("\n   Top 10 features con más outliers:")
print("   " + "-" * 66)
print(f"   {'Feature':<30} {'Outliers':<12} {'% del total':<12}")
print("   " + "-" * 66)

for feature, info in sorted_features[:10]:
    print(f"   {feature:<30} {info['count']:<12} {info['percentage']:<12.2f}%")

# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================
print("\n[4/5] Generando visualizaciones...")

# Crear directorio de reportes si no existe
reports_dir = Path("reports/figures")
reports_dir.mkdir(parents=True, exist_ok=True)

# 4.1 Distribución de outliers por feature
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Gráfico 1: Número de outliers por feature
outlier_counts = [info['count'] for _, info in sorted_features]
feature_names = [name for name, _ in sorted_features]

axes[0].bar(range(len(outlier_counts[:20])), outlier_counts[:20], color='coral')
axes[0].set_xlabel('Features (Top 20)', fontsize=12)
axes[0].set_ylabel('Número de Outliers', fontsize=12)
axes[0].set_title('Distribución de Outliers por Feature (Top 20)', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(len(feature_names[:20])))
axes[0].set_xticklabels(feature_names[:20], rotation=90, ha='right')
axes[0].grid(axis='y', alpha=0.3)

# Gráfico 2: Porcentaje de outliers por feature
outlier_percentages = [info['percentage'] for _, info in sorted_features]
axes[1].bar(range(len(outlier_percentages[:20])), outlier_percentages[:20], color='steelblue')
axes[1].set_xlabel('Features (Top 20)', fontsize=12)
axes[1].set_ylabel('Porcentaje de Outliers (%)', fontsize=12)
axes[1].set_title('Porcentaje de Outliers por Feature (Top 20)', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(len(feature_names[:20])))
axes[1].set_xticklabels(feature_names[:20], rotation=90, ha='right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=10, color='red', linestyle='--', label='10% threshold', alpha=0.7)
axes[1].legend()

plt.tight_layout()
plt.savefig(reports_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Gráfico guardado: {reports_dir / 'outlier_analysis.png'}")

# 4.2 Boxplots de features con más outliers
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, (feature, _) in enumerate(sorted_features[:9]):
    axes[idx].boxplot(X_train[feature].dropna(), vert=True)
    axes[idx].set_title(feature, fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Valor', fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(reports_dir / 'outlier_boxplots.png', dpi=300, bbox_inches='tight')
print(f"   ✅ Boxplots guardados: {reports_dir / 'outlier_boxplots.png'}")

plt.close('all')

# ============================================================================
# 5. RECOMENDACIONES
# ============================================================================
print("\n[5/5] Generando recomendaciones...")
print("\n" + "=" * 70)
print("🎯 RECOMENDACIONES BASADAS EN ANÁLISIS")
print("=" * 70)

# Decisión basada en análisis
if outlier_percentage > 50:
    recommendation = "RobustScaler"
    reason = "Más del 50% de filas tienen outliers - usar escalador robusto"
    color = "🟡"
elif outlier_percentage > 30:
    recommendation = "RobustScaler"
    reason = "30-50% de filas con outliers - RobustScaler recomendado"
    color = "🟢"
else:
    recommendation = "StandardScaler o RobustScaler"
    reason = "Menos del 30% outliers - ambos métodos son válidos"
    color = "🟢"

print(f"\n{color} DECISIÓN: Usar {recommendation}")
print(f"   Razón: {reason}")

print("\n📋 JUSTIFICACIÓN TÉCNICA:")
print("   1. Random Forest es naturalmente robusto a outliers")
print("   2. Outliers pueden contener información valiosa (no son errores)")
print(f"   3. {outlier_percentage:.1f}% de datos afectados por al menos 1 outlier")
print("   4. MLOps best practice: Transformar, no eliminar")

print("\n🔧 IMPLEMENTACIÓN:")
print("   → No usar OutlierRemover (elimina filas, rompe pipeline)")
if outlier_percentage > 30:
    print("   → Usar RobustScaler (usa mediana e IQR)")
else:
    print("   → StandardScaler o RobustScaler son válidos")
    print("   → RobustScaler es más conservador")

print("\n💡 ALTERNATIVAS CONSIDERADAS:")
print("   ❌ Eliminar outliers: Rompe sklearn pipeline (cambio de # filas)")
print("   ❌ Imputar outliers: Pérdida de información potencialmente valiosa")
print("   ✅ RobustScaler: Maneja outliers sin eliminar información")
print("   ✅ StandardScaler: Válido si outliers < 30%")

# Guardar reporte en archivo
report_file = reports_dir / 'outlier_analysis_report.txt'
with open(report_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("REPORTE DE ANÁLISIS DE OUTLIERS\n")
    f.write("Turkish Music Emotion Recognition Project\n")
    f.write("MLOps Team 24\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("ESTADÍSTICAS GENERALES:\n")
    f.write(f"  - Features totales: {total_features}\n")
    f.write(f"  - Features con outliers: {features_with_outliers}\n")
    f.write(f"  - Filas con outliers: {total_outlier_rows} ({outlier_percentage:.1f}%)\n\n")
    
    f.write("DECISIÓN:\n")
    f.write(f"  - Método recomendado: {recommendation}\n")
    f.write(f"  - Razón: {reason}\n\n")
    
    f.write("TOP 10 FEATURES CON MÁS OUTLIERS:\n")
    for feature, info in sorted_features[:10]:
        f.write(f"  - {feature}: {info['count']} outliers ({info['percentage']:.2f}%)\n")

print(f"\n✅ Reporte guardado: {report_file}")

print("\n" + "=" * 70)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 70)
print("\n💾 Archivos generados:")
print(f"   - {reports_dir / 'outlier_analysis.png'}")
print(f"   - {reports_dir / 'outlier_boxplots.png'}")
print(f"   - {report_file}")
print("\n🎯 Próximo paso: Actualizar sklearn_pipeline.py con la decisión")
