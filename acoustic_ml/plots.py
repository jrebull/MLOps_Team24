"""
Módulo de visualizaciones adaptado para tests y MLOps.
Incluye:
- PlotManager
- BasePlotter
- FeatureImportancePlotter
- Funciones legacy compatibles
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import warnings

from acoustic_ml.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# ------------------- Plot Manager -------------------
class PlotManager:
    def __init__(self, reports_dir: Path = REPORTS_DIR, default_subfolder: str = "figures",
                 dpi: int = 300, style: str = "whitegrid"):
        self.reports_dir = reports_dir
        self.default_subfolder = default_subfolder
        self.dpi = dpi
        self.style = style
        sns.set_style(self.style)
        logger.info(f"PlotManager inicializado con reports_dir={reports_dir}")

    def save_figure(self, fig: plt.Figure, filename: str,
                    subfolder: Optional[str] = None, dpi: Optional[int] = None) -> Path:
        if fig is None:
            raise ValueError("La figura no puede ser None")
        subfolder = subfolder or self.default_subfolder
        dpi = dpi or self.dpi
        save_path = self.reports_dir / subfolder / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
        logger.info(f"Figura guardada en {save_path}")
        print(f"✅ Figura guardada en {save_path}")
        return save_path

    def create_subplot_grid(self, nrows: int = 1, ncols: int = 1,
                            figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        figsize = figsize or (10 * ncols, 8 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        return fig, axes

# ------------------- Base Plotter -------------------
class BasePlotter(ABC):
    def __init__(self, manager: PlotManager):
        self.manager = manager

    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        pass

    def _validate_data(self, data, expected_type=None):
        if data is None:
            raise ValueError("Los datos no pueden ser None")
        if expected_type and not isinstance(data, expected_type):
            raise ValueError(f"Tipo inválido. Esperado: {expected_type}, recibido: {type(data)}")

# ------------------- Feature Importance Plotter -------------------
class FeatureImportancePlotter(BasePlotter):
    def __init__(self, manager: PlotManager, default_top_n: int = 20, figsize: Tuple[int,int]=(10,8)):
        super().__init__(manager)
        self.default_top_n = default_top_n
        self.figsize = figsize

    def plot(self, feature_importance: Dict[str,float], top_n: Optional[int] = None,
             title: Optional[str] = None, xlabel: str = 'Importance', color: str='steelblue') -> plt.Figure:
        self._validate_data(feature_importance, dict)
        if not feature_importance:
            raise ValueError("Diccionario vacío de feature importance")
        top_n = top_n or self.default_top_n
        title = title or f'Top {top_n} Feature Importance'
        sorted_features = sorted(feature_importance.items(), key=lambda x:x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.barh(features, importances, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Features')
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def plot_and_save(self, feature_importance: Dict[str,float], filename: str="feature_importance.png", **kwargs) -> Path:
        fig = self.plot(feature_importance, **kwargs)
        save_path = self.manager.save_figure(fig, filename)
        plt.close(fig)
        return save_path

# ------------------- Funciones Legacy -------------------
def save_figure(fig, filename: str, subfolder: str = "figures") -> None:
    warnings.warn("save_figure() deprecated. Use PlotManager.save_figure() instead.", DeprecationWarning)
    manager = PlotManager()
    manager.save_figure(fig, filename, subfolder)

def plot_feature_importance(feature_importance: dict, top_n: int = 20) -> plt.Figure:
    warnings.warn("plot_feature_importance() deprecated. Use FeatureImportancePlotter.plot() instead.", DeprecationWarning)
    manager = PlotManager()
    plotter = FeatureImportancePlotter(manager)
    return plotter.plot(feature_importance, top_n)

# ------------------- Outlier Analysis Plotter -------------------
class OutlierAnalysisPlotter(BasePlotter):
    """
    Plotter para análisis de outliers por feature.
    Genera boxplots para cada feature numérica y permite guardar la figura.
    Maneja validaciones y escenarios problemáticos.
    """
    def __init__(self, manager: PlotManager, figsize: Tuple[int,int]=(12,8)):
        super().__init__(manager)
        self.figsize = figsize

    def plot(self, df, title: Optional[str]=None, show: bool=False) -> plt.Figure:
        """
        Crea boxplots para todas las columnas numéricas del dataframe.

        Args:
            df: DataFrame con features a analizar
            title: Título del gráfico
            show: Si True, muestra la figura en pantalla
        
        Returns:
            Figura matplotlib
        """
        # Validar datos
        self._validate_data(df, expected_type=pd.DataFrame)
        if df.empty:
            raise ValueError("El DataFrame está vacío")

        numeric_cols = df.select_dtypes(include=['float', 'int']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No hay columnas numéricas para analizar outliers")
        
        # Crear figura
        fig, ax = plt.subplots(figsize=self.figsize)
        df[numeric_cols].boxplot(ax=ax, vert=False)
        ax.set_title(title or "Outlier Analysis", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if show:
            plt.show()
        
        return fig

    def plot_and_save(
        self,
        df,
        filename: str="outlier_analysis.png",
        title: Optional[str]=None,
        show: bool=False
    ) -> Path:
        """
        Crea y guarda automáticamente el plot de outliers.
        Validaciones incluidas.
        """
        # Validar ruta de guardado
        if not isinstance(filename, str) or not filename.endswith((".png", ".jpg", ".pdf")):
            raise ValueError("filename debe ser string y terminar en .png, .jpg o .pdf")
        
        fig = self.plot(df, title, show=show)

        # Guardar figura
        save_path = self.manager.save_figure(fig, filename)
        plt.close(fig)  # Cerrar figura para liberar memoria
        return save_path