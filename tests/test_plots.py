import matplotlib
matplotlib.use('Agg')  # Configurar backend no interactivo antes de importar pyplot
import pytest
from pathlib import Path
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import seaborn as sns
import numpy as np

# Importamos la clase que vamos a probar
from acoustic_ml.plots import PlotManager, FeatureImportancePlotter, save_figure, plot_feature_importance


@pytest.fixture
def manager(tmp_path):
    """Fixture que proporciona una instancia de PlotManager con un directorio temporal."""
    return PlotManager(reports_dir=tmp_path)


class TestPlotManager:
    """Grupo de tests para la clase PlotManager."""

    def test_init(self):
        """Verifica que PlotManager se inicializa con los valores correctos."""
        # Arrange
        test_dpi = 150
        test_style = "whitegrid"
        
        # Act
        manager_instance = PlotManager(dpi=test_dpi, style=test_style)
        
        # Assert
        assert manager_instance.dpi == test_dpi
        assert manager_instance.style == test_style
        assert manager_instance.default_subfolder == "figures"
        assert sns.axes_style() == sns.axes_style(test_style)

    def test_save_figure(self, manager):
        """Verifica que save_figure guarda una figura correctamente."""
        # Arrange
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        filename = "test_plot.png"
        
        # Act
        saved_path = manager.save_figure(fig, filename)
        
        # Assert
        expected_path = manager.reports_dir / "figures" / filename
        assert saved_path == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0
        
        plt.close(fig)

    def test_save_figure_raises_error_for_none_fig(self, manager):
        """Verifica que save_figure lanza un ValueError si la figura es None."""
        with pytest.raises(ValueError, match="La figura no puede ser None"):
            manager.save_figure(None, "some_file.png")

    def test_create_subplot_grid(self):
        """Verifica que create_subplot_grid devuelve una grilla de subplots."""
        # Arrange
        manager_instance = PlotManager()
        
        # Act
        fig, axes = manager_instance.create_subplot_grid(nrows=2, ncols=2, figsize=(10, 8))
        
        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)
        assert fig.get_size_inches().tolist() == [10.0, 8.0]
        
        plt.close(fig)

    def test_create_subplot_grid_single_plot(self):
        """Verifica create_subplot_grid para un solo plot."""
        # Arrange
        manager_instance = PlotManager()
        
        # Act
        fig, ax = manager_instance.create_subplot_grid(nrows=1, ncols=1)
        
        # Assert
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
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

@pytest.fixture
def plotter(manager):
    """Fixture que proporciona una instancia de FeatureImportancePlotter."""
    return FeatureImportancePlotter(manager)


class TestFeatureImportancePlotter:
    """Grupo de tests para la clase FeatureImportancePlotter."""

    def test_init(self, manager):
        """Verifica que FeatureImportancePlotter se inicializa correctamente."""
        # Act
        plotter_instance = FeatureImportancePlotter(manager, default_top_n=10, figsize=(12, 10))
        
        # Assert
        assert plotter_instance.manager is manager
        assert plotter_instance.default_top_n == 10
        assert plotter_instance.figsize == (12, 10)

    def test_plot_basic(self, plotter, sample_feature_importance_data):
        """Verifica que el método plot genera una figura con las propiedades correctas."""
        # Act
        fig = plotter.plot(sample_feature_importance_data, top_n=5, title="Test Plot", xlabel="Score")
        
        # Assert
        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        assert ax.get_title() == "Test Plot"
        assert ax.get_xlabel() == "Score"
        assert len(ax.patches) == 5
        
        expected_features = ['spectral_centroid_mean', 'mfcc_1_mean', 'rms_mean', 'zero_crossing_rate_mean', 'spectral_rolloff_mean']
        actual_features = [label.get_text() for label in ax.get_yticklabels()]
        assert actual_features == expected_features
        
        plt.close(fig)

    def test_plot_empty_data_raises_error(self, plotter):
        """Verifica que plot lanza ValueError con un diccionario vacío."""
        with pytest.raises(ValueError, match="El diccionario de feature importance está vacío"):
            plotter.plot({})

    def test_plot_none_data_raises_error(self, plotter):
        """Verifica que plot lanza ValueError con datos None."""
        with pytest.raises(ValueError, match="Los datos no pueden ser None"):
            plotter.plot(None)

    def test_plot_and_save(self, plotter, sample_feature_importance_data):
        """Verifica que plot_and_save genera y guarda la figura correctamente."""
        # Arrange
        filename = "feature_importance_test.png"
        
        # Act
        saved_path = plotter.plot_and_save(sample_feature_importance_data, filename, top_n=3)
        
        # Assert
        expected_path = plotter.manager.reports_dir / "figures" / filename
        assert saved_path == expected_path
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0


# --- Legacy Functions Tests ---

@pytest.mark.filterwarnings("always::DeprecationWarning")
def test_legacy_save_figure_warns_and_calls_new_method(tmp_path):
    """Verifica que la función legacy save_figure emite un DeprecationWarning y llama a PlotManager.save_figure."""
    # Arrange
    mock_manager_instance = MagicMock(spec=PlotManager)
    
    # "Parcheamos" la clase PlotManager. Cuando la función legacy la llame, obtendrá nuestro mock.
    with patch('acoustic_ml.plots.PlotManager', return_value=mock_manager_instance) as MockPlotManagerClass:
        # Verificamos que se emite la advertencia correcta.
        with pytest.warns(DeprecationWarning, match=r"save_figure\(\) está deprecated"):
            # 2. Act
            fig, ax = plt.subplots()
            filename = "legacy_test.png"
            save_figure(fig, filename) # Llamamos a la función legacy
            
            # 3. Assert
            # Verificamos que PlotManager() fue llamado una vez.
            MockPlotManagerClass.assert_called_once()
            # Verificamos que el método save_figure de nuestro mock fue llamado con los argumentos correctos.
            mock_manager_instance.save_figure.assert_called_once_with(fig, filename, "figures")
            
            plt.close(fig)

@pytest.mark.filterwarnings("always::DeprecationWarning")
def test_legacy_plot_feature_importance_warns_and_calls_new_method(sample_feature_importance_data):
    """Verifica que la función legacy plot_feature_importance emite un DeprecationWarning y llama a FeatureImportancePlotter.plot."""
    # Arrange
    mock_manager_instance = MagicMock(spec=PlotManager)
    mock_plotter_instance = MagicMock(spec=FeatureImportancePlotter)
    
    # Arrange: Configurar el mock para que devuelva una figura real para poder cerrarla.
    real_fig_for_mock, _ = plt.subplots()
    mock_plotter_instance.plot.return_value = real_fig_for_mock

    # Parcheamos ambas clases
    with patch('acoustic_ml.plots.PlotManager', return_value=mock_manager_instance) as MockPlotManagerClass:
        with patch('acoustic_ml.plots.FeatureImportancePlotter', return_value=mock_plotter_instance) as MockFeatureImportancePlotterClass:
            with pytest.warns(DeprecationWarning, match=r"plot_feature_importance\(\) está deprecated"):
                # Act
                fig_returned = plot_feature_importance(sample_feature_importance_data, top_n=3)
                
                # Assert
                MockPlotManagerClass.assert_called_once()
                MockFeatureImportancePlotterClass.assert_called_once_with(mock_manager_instance)
                mock_plotter_instance.plot.assert_called_once_with(sample_feature_importance_data, 3)
                assert fig_returned is real_fig_for_mock
                plt.close(fig_returned)
