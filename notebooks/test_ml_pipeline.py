# test_ml_pipeline.py
import pandas as pd
import numpy as np
import pytest
from scipy.stats import skew
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_pipeline import DataLoader, Preprocessor, ModelTrainer


@pytest.fixture
def sample_df():
    """Crea un DataFrame de prueba para usar en los tests"""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=0, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["Class"] = y
    return df


def test_data_loader_reads_csv(tmp_path):
    """Verifica que el DataLoader carga correctamente un archivo CSV"""
    test_file = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "Class": [0, 1]})
    df.to_csv(test_file, index=False)

    loader = DataLoader(str(test_file))
    loaded_df = loader.load_csv()
    pd.testing.assert_frame_equal(loaded_df, df)


def test_encode_labels(sample_df):
    """Verifica que la codificación de etiquetas se aplique correctamente"""
    preprocessor = Preprocessor(sample_df)
    preprocessor.encode_labels()
    encoded = preprocessor.get_clean_df()
    assert set(encoded["Class"].unique()).issubset({0, 1})


def test_drop_duplicates_removes_dupes():
    """Verifica que se eliminan filas duplicadas"""
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2], "Class": [0, 0]})
    preprocessor = Preprocessor(df)
    preprocessor.drop_duplicates()
    cleaned = preprocessor.get_clean_df()
    assert cleaned.shape[0] == 1


def test_drop_constant_columns_removes_constants():
    """Verifica que se eliminan columnas con varianza cero"""
    df = pd.DataFrame({
        "a": [1, 1, 1],
        "b": [2, 3, 4],
        "Class": [0, 1, 0]
    })
    preprocessor = Preprocessor(df)
    preprocessor.drop_constant_columns()
    cleaned = preprocessor.get_clean_df()
    assert "a" not in cleaned.columns


def test_detect_outliers_does_not_crash(sample_df):
    """Verifica que la detección de outliers por z-score no lanza errores"""
    preprocessor = Preprocessor(sample_df)
    preprocessor.detect_outliers()  # No hace falta un assert, solo que no crashee


def test_iqr_outliers_does_not_crash(sample_df):
    """Verifica que la detección de outliers por IQR no lanza errores"""
    preprocessor = Preprocessor(sample_df)
    preprocessor.iqr_outliers()  # No hace falta un assert, solo que no crashee

def test_normalize_skewed_runs():
    """Verifica que normalize_skewed transforma columnas sesgadas correctamente."""
    # Crea un DataFrame con una variable muy sesgada
    df = pd.DataFrame({
        "x1": np.random.exponential(scale=5, size=200),  # sesgo positivo
        "x2": np.random.normal(0, 1, 200),               # casi normal
        "Class": np.random.choice(["happy", "sad"], 200)
    })
    pre = Preprocessor(df)
    cols_trans = pre.normalize_skewed(threshold=0.5)

    # Verifica que devolvió una lista
    assert isinstance(cols_trans, list)
    # Verifica que haya al menos una columna transformada
    assert len(cols_trans) > 0

    # Y que el sesgo absoluto haya disminuido
    before = abs(skew(df[cols_trans[0]]))
    after = abs(skew(pre.df[cols_trans[0]]))
    assert after < before


def test_split_and_scale_shapes(sample_df):
    """Verifica que los tamaños de train/test son correctos"""
    trainer = ModelTrainer(sample_df)
    trainer.split_and_scale(test_size=0.2)
    X_train, X_test, Y_train, Y_test = trainer.get_split_data()

    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20
    assert Y_train.shape[0] == 80
    assert Y_test.shape[0] == 20


def test_train_logistic_runs(sample_df, capsys):
    """Verifica que el entrenamiento del modelo logístico se ejecuta correctamente"""
    trainer = ModelTrainer(sample_df)
    trainer.split_and_scale()
    trainer.train_logistic()
    captured = capsys.readouterr()
    assert "Accuracy" in captured.out


def test_train_knn_runs(sample_df, capsys):
    """Verifica que el modelo KNN se entrena sin errores"""
    trainer = ModelTrainer(sample_df)
    trainer.split_and_scale()
    trainer.train_knn()
    captured = capsys.readouterr()
    assert "KNN" in captured.out