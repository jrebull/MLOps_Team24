"""
Visualizaciones para análisis y reportes
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from acoustic_ml.config import REPORTS_DIR


def save_figure(fig, filename: str, subfolder: str = "figures") -> None:
    """
    Guarda una figura en reports/figures/
    
    Args:
        fig: Figura de matplotlib
        filename: Nombre del archivo
        subfolder: Subcarpeta dentro de reports
    """
    save_path = REPORTS_DIR / subfolder / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Figura guardada en {save_path}")


def plot_feature_importance(feature_importance: dict, top_n: int = 20) -> plt.Figure:
    """
    Visualiza importancia de features
    
    Args:
        feature_importance: Diccionario {feature_name: importance}
        top_n: Número de features principales a mostrar
        
    Returns:
        Figura de matplotlib
    """
    # Ordenar features por importancia
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(features, importances)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    return fig
