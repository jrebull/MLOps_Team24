"""
Módulo de visualizaciones para análisis y reportes.

Este módulo implementa clases para generar y gestionar visualizaciones
del proyecto de clasificación de emociones en música turca.

Implementa principios SOLID:
- Single Responsibility: Cada plotter tiene una responsabilidad específica
- Open/Closed: Extensible mediante herencia, cerrado a modificación
- Liskov Substitution: Todos los plotters son intercambiables
- Interface Segregation: Interfaces mínimas y específicas
- Dependency Inversion: Depende de abstracciones, no implementaciones

Clases:
    PlotManager: Gestor principal para guardar y configurar figuras
    BasePlotter: Clase base abstracta para todos los plotters
    FeatureImportancePlotter: Visualización de importancia de features

Ejemplo:
    >>> manager = PlotManager()
    >>> plotter = FeatureImportancePlotter(manager)
    >>> fig = plotter.plot(feature_importance_dict, top_n=15)
    >>> manager.save_figure(fig, "feature_importance.png")
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging
import warnings

from acoustic_ml.config import REPORTS_DIR

# Configuración de logging
logger = logging.getLogger(__name__)


class PlotManager:
    """
    Gestor principal para guardar y configurar figuras.
    
    Centraliza la lógica de guardado de figuras y configuración
    de estilos visuales. Sigue el principio de Single Responsibility.
    
    Attributes:
        reports_dir: Directorio raíz para reportes
        default_subfolder: Subcarpeta por defecto para figuras
        dpi: Resolución por defecto para guardar figuras
        style: Estilo de seaborn a utilizar
    
    Ejemplo:
        >>> manager = PlotManager(dpi=300)
        >>> manager.save_figure(fig, "output.png")
        ✅ Figura guardada en reports/figures/output.png
    """
    
    def __init__(
        self,
        reports_dir: Path = REPORTS_DIR,
        default_subfolder: str = "figures",
        dpi: int = 300,
        style: str = "whitegrid"
    ):
        """
        Inicializa el gestor de plots.
        
        Args:
            reports_dir: Directorio raíz para guardar reportes
            default_subfolder: Subcarpeta por defecto dentro de reports
            dpi: Resolución para guardar figuras
            style: Estilo de seaborn ('whitegrid', 'darkgrid', 'white', etc.)
        """
        self.reports_dir = reports_dir
        self.default_subfolder = default_subfolder
        self.dpi = dpi
        self.style = style
        
        # Configurar estilo de seaborn
        sns.set_style(self.style)
        
        logger.info(f"PlotManager inicializado con reports_dir={reports_dir}")
    
    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        subfolder: Optional[str] = None,
        dpi: Optional[int] = None
    ) -> Path:
        """
        Guarda una figura en el directorio de reportes.
        
        Args:
            fig: Figura de matplotlib a guardar
            filename: Nombre del archivo (con extensión)
            subfolder: Subcarpeta específica (usa default si None)
            dpi: Resolución específica (usa default si None)
        
        Returns:
            Path del archivo guardado
        
        Raises:
            ValueError: Si la figura es None
            OSError: Si no se puede crear el directorio o guardar
        """
        if fig is None:
            raise ValueError("La figura no puede ser None")
        
        # Usar valores por defecto si no se especifican
        subfolder = subfolder or self.default_subfolder
        dpi = dpi or self.dpi
        
        # Construir ruta completa
        save_path = self.reports_dir / subfolder / filename
        
        # Crear directorio si no existe
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar figura
        try:
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            logger.info(f"Figura guardada en {save_path}")
            print(f"✅ Figura guardada en {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error al guardar figura: {e}")
            raise
    
    def create_subplot_grid(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Crea una grilla de subplots configurada.
        
        Args:
            nrows: Número de filas
            ncols: Número de columnas
            figsize: Tamaño de la figura (ancho, alto)
        
        Returns:
            Tupla (figura, axes)
        """
        figsize = figsize or (10 * ncols, 8 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        logger.debug(f"Grilla creada: {nrows}x{ncols}")
        return fig, axes


class BasePlotter(ABC):
    """
    Clase base abstracta para todos los plotters.
    
    Define la interfaz común para todos los plotters siguiendo
    el principio de Interface Segregation. Cada plotter específico
    debe implementar el método plot().
    
    Attributes:
        manager: Instancia de PlotManager para gestión de figuras
    """
    
    def __init__(self, manager: PlotManager):
        """
        Inicializa el plotter base.
        
        Args:
            manager: Gestor de plots para guardar figuras
        """
        self.manager = manager
        logger.debug(f"{self.__class__.__name__} inicializado")
    
    @abstractmethod
    def plot(self, *args, **kwargs) -> plt.Figure:
        """
        Método abstracto para crear visualización.
        
        Las subclases deben implementar este método con su
        lógica específica de visualización.
        
        Returns:
            Figura de matplotlib generada
        """
        pass
    
    def _validate_data(self, data, expected_type=None) -> None:
        """
        Valida que los datos no sean None y opcionalmente el tipo.
        
        Args:
            data: Datos a validar
            expected_type: Tipo esperado (opcional)
        
        Raises:
            ValueError: Si los datos son inválidos
        """
        if data is None:
            raise ValueError("Los datos no pueden ser None")
        
        if expected_type and not isinstance(data, expected_type):
            raise ValueError(
                f"Tipo de datos inválido. "
                f"Esperado: {expected_type}, Recibido: {type(data)}"
            )


class FeatureImportancePlotter(BasePlotter):
    """
    Plotter especializado para visualizar importancia de features.
    
    Crea gráficos de barras horizontales mostrando las features
    más importantes del modelo. Sigue el principio de Single Responsibility.
    
    Ejemplo:
        >>> manager = PlotManager()
        >>> plotter = FeatureImportancePlotter(manager)
        >>> importance = {'feature_1': 0.5, 'feature_2': 0.3}
        >>> fig = plotter.plot(importance, top_n=15)
    """
    
    def __init__(
        self,
        manager: PlotManager,
        default_top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Inicializa el plotter de feature importance.
        
        Args:
            manager: Gestor de plots
            default_top_n: Número por defecto de features a mostrar
            figsize: Tamaño por defecto de la figura
        """
        super().__init__(manager)
        self.default_top_n = default_top_n
        self.figsize = figsize
        logger.info("FeatureImportancePlotter inicializado")
    
    def plot(
        self,
        feature_importance: Dict[str, float],
        top_n: Optional[int] = None,
        title: Optional[str] = None,
        xlabel: str = 'Importance',
        color: str = 'steelblue'
    ) -> plt.Figure:
        """
        Crea visualización de importancia de features.
        
        Args:
            feature_importance: Diccionario {nombre_feature: importancia}
            top_n: Número de features principales (usa default si None)
            title: Título personalizado del gráfico
            xlabel: Etiqueta del eje X
            color: Color de las barras
        
        Returns:
            Figura de matplotlib
        
        Raises:
            ValueError: Si feature_importance está vacío o es inválido
        """
        # Validar datos
        self._validate_data(feature_importance, dict)
        
        if not feature_importance:
            raise ValueError("El diccionario de feature importance está vacío")
        
        # Usar valores por defecto si no se especifican
        top_n = top_n or self.default_top_n
        title = title or f'Top {top_n} Feature Importance'
        
        # Ordenar features por importancia descendente
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Desempaquetar features e importancias
        features, importances = zip(*sorted_features)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Crear barras horizontales
        ax.barh(features, importances, color=color)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Invertir eje Y para mostrar el más importante arriba
        ax.invert_yaxis()
        
        # Ajustar layout
        plt.tight_layout()
        
        logger.info(f"Plot de feature importance creado con top {top_n} features")
        
        return fig
    
    def plot_and_save(
        self,
        feature_importance: Dict[str, float],
        filename: str = "feature_importance.png",
        **plot_kwargs
    ) -> Path:
        """
        Crea y guarda automáticamente el plot de feature importance.
        
        Método de conveniencia que combina plot() y save_figure().
        
        Args:
            feature_importance: Diccionario de importancias
            filename: Nombre del archivo a guardar
            **plot_kwargs: Argumentos adicionales para plot()
        
        Returns:
            Path del archivo guardado
        """
        fig = self.plot(feature_importance, **plot_kwargs)
        save_path = self.manager.save_figure(fig, filename)
        plt.close(fig)  # Cerrar figura para liberar memoria
        return save_path


# ============================================================================
# FUNCIONES LEGACY (DEPRECATED) - Solo para compatibilidad backward
# ============================================================================
# Estas funciones se mantienen para no romper código existente, pero no
# deben usarse en código nuevo. Usar las clases PlotManager y plotters.
# ============================================================================

def save_figure(fig, filename: str, subfolder: str = "figures") -> None:
    """
    Función legacy para guardar figuras (mantiene compatibilidad).
    
    .. deprecated:: 2024.10
        Usar :class:`PlotManager.save_figure` en su lugar.
        Esta función será removida en versión futura.
    
    Args:
        fig: Figura de matplotlib
        filename: Nombre del archivo
        subfolder: Subcarpeta dentro de reports
    
    Ejemplo de migración:
        >>> # Código viejo (deprecated)
        >>> save_figure(fig, "output.png")
        
        >>> # Código nuevo (recomendado)
        >>> manager = PlotManager()
        >>> manager.save_figure(fig, "output.png")
    """
    warnings.warn(
        "save_figure() está deprecated y será removido en versión futura. "
        "Usar PlotManager.save_figure() en su lugar.",
        DeprecationWarning,
        stacklevel=2
    )
    manager = PlotManager()
    manager.save_figure(fig, filename, subfolder)


def plot_feature_importance(feature_importance: dict, top_n: int = 20) -> plt.Figure:
    """
    Función legacy para plot de feature importance (mantiene compatibilidad).
    
    .. deprecated:: 2024.10
        Usar :class:`FeatureImportancePlotter.plot` en su lugar.
        Esta función será removida en versión futura.
    
    Args:
        feature_importance: Diccionario {feature_name: importance}
        top_n: Número de features principales a mostrar
    
    Returns:
        Figura de matplotlib
    
    Ejemplo de migración:
        >>> # Código viejo (deprecated)
        >>> fig = plot_feature_importance(importance_dict, top_n=15)
        
        >>> # Código nuevo (recomendado)
        >>> manager = PlotManager()
        >>> plotter = FeatureImportancePlotter(manager)
        >>> fig = plotter.plot(importance_dict, top_n=15)
    """
    warnings.warn(
        "plot_feature_importance() está deprecated y será removido en versión futura. "
        "Usar FeatureImportancePlotter.plot() en su lugar.",
        DeprecationWarning,
        stacklevel=2
    )
    manager = PlotManager()
    plotter = FeatureImportancePlotter(manager)
    return plotter.plot(feature_importance, top_n)
