import pytest
from pathlib import Path
import matplotlib.pyplot as plt
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

def test_plot_manager_save_figure(tmp_path):
    """
    Verifica que PlotManager.save_figure guarda una figura correctamente.
    Usa el fixture tmp_path de pytest para crear un directorio temporal.
    """
    # 1. Arrange (Preparación)
    # Usamos tmp_path como el directorio de reportes para el test.
    # Esto asegura que no dejamos archivos basura en nuestro proyecto.
    manager = PlotManager(reports_dir=tmp_path)
    
    # Creamos una figura simple de matplotlib para el test.
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    
    filename = "test_plot.png"
    
    # 2. Act (Acción)
    # Llamamos al método que queremos probar.
    saved_path = manager.save_figure(fig, filename)
    
    # 3. Assert (Verificación)
    # Construimos la ruta esperada. El manager debe crear la subcarpeta 'figures'.
    expected_path = tmp_path / "figures" / filename
    
    # Verificamos que la ruta devuelta es la correcta.
    assert saved_path == expected_path
    # Verificamos que el archivo fue realmente creado en el disco.
    assert expected_path.exists()
    # Verificamos que el archivo no está vacío (que se guardó algo).
    assert expected_path.stat().st_size > 0
    
    # Es una buena práctica cerrar la figura para liberar memoria.
    plt.close(fig)
