import joblib
from pathlib import Path
from typing import Any

# Importaciones de módulos locales que contienen la lógica del pipeline de ML
from src.ml_pipeline import DataLoader, Preprocessor, ModelTrainer
from app.core.config import settings
from app.core.logger import logger

# Directorio donde se guardarán los modelos entrenados.
MODEL_DIR = Path(settings.MODEL_PATH).parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ModelService:
    """
    Clase de servicio responsable de gestionar el ciclo de vida del modelo
    de Machine Learning: entrenamiento, carga y predicción.
    """
    def __init__(self, model_path: str = settings.MODEL_PATH):
        """
        Inicializa el ModelService.

        Args:
            model_path (str): Ruta predeterminada donde se espera encontrar el modelo.
        """
        self.model_path = model_path
        self.model = None

    def train(self, data_path: str, model_name: str = "model_v1"):
        """
        Ejecuta el pipeline completo de entrenamiento del modelo.

        1. Carga los datos.
        2. Aplica el preprocesamiento (codificación, limpieza, normalización).
        3. Entrena el modelo (Regresión Logística por defecto).
        4. Guarda el modelo serializado usando joblib.

        Args:
            data_path (str): Ruta al archivo de datos de entrenamiento (e.g., CSV).
            model_name (str): Nombre base para el archivo del modelo (e.g., "model_v1").

        Returns:
            str: La ruta completa donde se guardó el modelo entrenado.
        """
        logger.info("Starting training pipeline")

        # 1. Carga de datos
        loader = DataLoader(data_path)
        df = loader.load_csv()

        # 2. Preprocesamiento de datos
        pre = Preprocessor(df)
        pre.encode_labels()           # Codifica variables categóricas.
        pre.drop_duplicates()         # Elimina filas duplicadas.
        pre.drop_constant_columns()   # Elimina columnas con un único valor.
        pre.detect_outliers()         # Detecta posibles outliers (e.g., usa el método IQR).
        pre.iqr_outliers()            # Limpia/trata los outliers según la estrategia IQR.
        pre.normalize_skewed()        # Normaliza las distribuciones asimétricas de las características.
        cleaned = pre.get_clean_df()  # Obtiene el DataFrame preprocesado.

        # 3. Entrenamiento del modelo
        trainer = ModelTrainer(cleaned)
        trainer.split_and_scale()     # Divide datos y aplica escalamiento (e.g., StandardScaler).
        trainer.train_logistic()      # Entrena el modelo (asumiendo Logistic Regression).
        model = trainer.get_trained_model()  # Obtiene el objeto del modelo entrenado.
        
        # 4. Guardar el modelo
        path = MODEL_DIR / f"{model_name}.joblib"
        joblib.dump(model, path) # Serializa y guarda el modelo a disco.
        logger.info("Saved model to %s", path)
        return str(path)

    def load(self, model_path: str | None = None):
        """
        Carga el modelo serializado desde la ruta especificada.

        Args:
            model_path (str | None): Ruta al archivo .joblib. Usa self.model_path
                                     si no se especifica.

        Returns:
            Any: El objeto del modelo (e.g., un clasificador de sklearn).
        """
        path = model_path or self.model_path
        logger.info("Loading model from %s", path)
        # Deserializa y carga el modelo.
        self.model = joblib.load(path)
        return self.model

    def predict(self, features: list[float]):
        """
        Realiza una predicción utilizando el modelo cargado.

        Si el modelo aún no está cargado, lo carga automáticamente.
        Calcula la clase predicha y las probabilidades (si el modelo lo soporta).

        Args:
            features (list[float]): Lista de características (una única muestra).

        Returns:
            dict[str, Any]: Un diccionario con la predicción de clase y las
                            probabilidades asociadas.
        """
        # Carga el modelo si aún no se ha hecho.
        if self.model is None:
            self.load()

        import numpy as np
        # Convierte la lista de features a un array de numpy y lo reformatea para sklearn (1 fila, N columnas).
        X = np.array(features).reshape(1, -1)

        # Realiza la predicción de clase. El [0] obtiene el primer (y único) resultado.
        pred = int(self.model.predict(X)[0])
        # Comprueba si el modelo tiene el método 'predict_proba' (e.g., para clasificadores).
        probs = None
        if hasattr(self.model, "predict_proba"):
            # Obtiene las probabilidades y las convierte a lista.
            probs = self.model.predict_proba(X).tolist()[0]
        return {"prediction": pred, "probabilities": probs}
