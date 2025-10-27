import pytest
from pathlib import Path
import seaborn as sns

# Importamos la clase que vamos a probar
from acoustic_ml.plots import PlotManager

def test_plot_manager_init():
    """
    Verifica que la clase PlotManager se inicializa correctamente
    y sus atributos tienen los valores esperados.
    """
    # 1. Arrange (Preparación): Definimos los valores de entrada.
    test_dpi = 150
    test_style = "whitegrid"
    
    # 2. Act (Acción): Creamos una instancia de la clase.
    manager = PlotManager(dpi=test_dpi, style=test_style)
    
    # 3. Assert (Verificación): Usamos 'assert' para comprobar los resultados.
    # Pytest se encarga de todo si una de estas condiciones no se cumple.
    assert manager.dpi == test_dpi
    assert manager.style == test_style
    assert manager.default_subfolder == "figures"
    
    # También podemos verificar efectos secundarios, como la configuración de seaborn.
    # Nota: Este test asume que ningún otro test ha cambiado el estilo global.
    assert sns.axes_style() == sns.axes_style(test_style)
