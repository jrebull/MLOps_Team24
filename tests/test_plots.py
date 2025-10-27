import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo antes de importar pyplot
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importamos la clase que vamos a probar
from acoustic_ml.plots import PlotManager, FeatureImportancePlotter, BasePlotter

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

def test_save_figure_raises_error_for_none_fig(tmp_path):
    """Verifica que save_figure lanza un ValueError si la figura es None."""
    # 1. Arrange
    manager = PlotManager(reports_dir=tmp_path)
    
    # 2. Act & 3. Assert
    # pytest.raises es un "context manager" que espera una excepción.
    # El test pasa si se lanza un ValueError con el mensaje esperado.
    # El test falla si no se lanza ninguna excepción o si se lanza una diferente.
    with pytest.raises(ValueError, match="La figura no puede ser None"):
        manager.save_figure(None, "some_file.png")

def test_plot_manager_create_subplot_grid():
    """Verifica que create_subplot_grid devuelve una figura y ejes correctos para una grilla."""
    # 1. Arrange
    manager = PlotManager()
    
    # 2. Act
    fig, axes = manager.create_subplot_grid(nrows=2, ncols=2, figsize=(10, 8))
    
    # 3. Assert
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)  # Para 2x2, axes es un array de Axes
    assert axes.shape == (2, 2)
    assert fig.get_size_inches().tolist() == [10.0, 8.0]  # Verificar figsize
    
    plt.close(fig)

def test_plot_manager_create_subplot_grid_single_plot():
    """Verifica create_subplot_grid para un solo plot."""
    manager = PlotManager()
    fig, ax = manager.create_subplot_grid(nrows=1, ncols=1)
    
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)  # Para 1x1, ax es un solo objeto Axes
    
    plt.close(fig)

# --- FeatureImportancePlotter Tests ---

@pytest.fixture
def sample_feature_importance_data():
    """Proporciona un diccionario de prueba con importancias de features."""
    return {
        'spectral_centroid_mean': 0.25,
        'mfcc_1_mean': 0.20,
        'rms_mean': 0.15,
        'zero_crossing_rate_mean': 0.12,
        'spectral_rolloff_mean': 0.10,
        'tempo': 0.08,
        'chroma_stft_mean': 0.05,
        'spectral_bandwidth_mean': 0.03,
        'mfcc_2_mean': 0.02
    }

def test_feature_importance_plotter_init():
    """Verifica que FeatureImportancePlotter se inicializa correctamente."""
    # 1. Arrange
    manager = PlotManager() # Necesitamos una instancia de PlotManager
    
    # 2. Act
    plotter = FeatureImportancePlotter(manager, default_top_n=10, figsize=(12, 10))
    
    # 3. Assert
    assert plotter.manager is manager
    assert plotter.default_top_n == 10
    assert plotter.figsize == (12, 10)

def test_feature_importance_plotter_plot_basic(sample_feature_importance_data):
    """Verifica que el método plot genera una figura con las propiedades correctas."""
    # 1. Arrange
    manager = PlotManager()
    plotter = FeatureImportancePlotter(manager)
    
    # 2. Act
    fig = plotter.plot(sample_feature_importance_data, top_n=5, title="Test Plot", xlabel="Score")
    
    # 3. Assert
    # Verificamos que se devuelve un objeto de figura
    assert isinstance(fig, plt.Figure)
    
    # Obtenemos los ejes para inspeccionar el gráfico
    ax = fig.get_axes()[0]
    assert ax.get_title() == "Test Plot"
    assert ax.get_xlabel() == "Score"
    
    # Verificamos que se dibujaron el número correcto de barras (top_n=5)
    assert len(ax.patches) == 5
    
    # Verificamos que las features están ordenadas correctamente (de mayor a menor importancia)
    expected_features = ['spectral_centroid_mean', 'mfcc_1_mean', 'rms_mean', 'zero_crossing_rate_mean', 'spectral_rolloff_mean']
    actual_features = [label.get_text() for label in ax.get_yticklabels()]
    assert actual_features == expected_features
    
    plt.close(fig)

def test_feature_importance_plotter_plot_empty_data_raises_error():
    """Verifica que plot lanza ValueError con un diccionario vacío."""
    # 1. Arrange
    manager = PlotManager()
    plotter = FeatureImportancePlotter(manager)
    
    # 2. Act & 3. Assert
    with pytest.raises(ValueError, match="El diccionario de feature importance está vacío"):
        plotter.plot({})

def test_feature_importance_plotter_plot_none_data_raises_error():
    """Verifica que plot lanza ValueError con datos None."""
    # 1. Arrange
    manager = PlotManager()
    plotter = FeatureImportancePlotter(manager)
    
    # 2. Act & 3. Assert
    with pytest.raises(ValueError, match="Los datos no pueden ser None"):
        plotter.plot(None)

def test_feature_importance_plotter_plot_and_save(tmp_path, sample_feature_importance_data):
    """Verifica que plot_and_save genera y guarda la figura correctamente."""
    # 1. Arrange
    # Inicializamos PlotManager con tmp_path para que guarde en un directorio temporal
    manager = PlotManager(reports_dir=tmp_path)
    plotter = FeatureImportancePlotter(manager)
    
    filename = "feature_importance_test.png"
    
    # 2. Act
    # Llamamos al método que queremos probar.
    # Le pasamos top_n=3 para que el plot sea pequeño y rápido.
    saved_path = plotter.plot_and_save(sample_feature_importance_data, filename, top_n=3)
    
    # 3. Assert
    # Construimos la ruta esperada (PlotManager guarda en 'figures' por defecto)
    expected_path = tmp_path / "figures" / filename
    
    # Verificamos que la ruta devuelta es la correcta
    assert saved_path == expected_path
    # Verificamos que el archivo existe y no está vacío
    assert expected_path.exists()
    assert expected_path.stat().st_size > 0
    
    # Nota: El método plot_and_save ya cierra la figura internamente,
    # por lo que no necesitamos plt.close(fig) aquí.
