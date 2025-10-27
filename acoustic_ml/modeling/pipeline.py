"""
Pipeline completo de ML: orquestación de todo el flujo de entrenamiento y evaluación.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
from sklearn.base import BaseEstimator

from acoustic_ml.config import MODELS_DIR
from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.train import BaseModelTrainer, ModelConfig, TrainingConfig
from acoustic_ml.modeling.predict import ModelPredictor
from acoustic_ml.modeling.evaluate import ModelEvaluator

logger = logging.getLogger(__name__)


class MLPipeline:

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        model_filename: str = "baseline_model.pkl"
    ):
      
        self.model_config = model_config
        self.training_config = training_config
        self.model_filename = model_filename
        

        self.dataset_manager = DatasetManager()
        self.trainer = None
        self.model = None
        self.predictor = None
        self.evaluator = None
        

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        logger.info(f"🚀 MLPipeline inicializado: {model_config.name}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        logger.info("📊 Paso 1/5: Cargando datos...")
        
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = \
                self.dataset_manager.load_train_test_split()
            
            logger.info(f"✅ Datos cargados correctamente")
            logger.info(f"   - Train: {self.X_train.shape}")
            logger.info(f"   - Test: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        
        except Exception as e:
            logger.error(f"❌ Error al cargar datos: {e}")
            raise
    
    def train_model(self) -> BaseEstimator:
        """
        Entrenar el modelo con los datos cargados.
        
        Returns:
            Modelo entrenado
        
        Raises:
            ValueError: Si los datos no han sido cargados
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Datos no cargados. Ejecute load_data() primero.")
        
        logger.info("🎯 Paso 2/5: Entrenando modelo...")
        
        try:
            self.trainer = BaseModelTrainer(self.model_config, self.training_config)
            self.model = self.trainer.train(self.X_train, self.y_train)
            
            # Obtener métricas de validación cruzada
            cv_metrics = self.trainer.evaluate(self.X_train, self.y_train)
            
            logger.info(f"✅ Modelo entrenado exitosamente")
            logger.info(f"   - CV Accuracy (mean): {cv_metrics['accuracy_mean']:.4f}")
            logger.info(f"   - CV Accuracy (std): {cv_metrics['accuracy_std']:.4f}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"❌ Error al entrenar modelo: {e}")
            raise
    
    def save_model(self, custom_path: Optional[Path] = None) -> Path:
        """
        Guardar el modelo entrenado en disco.
        
        Args:
            custom_path: Ruta personalizada (opcional)
        
        Returns:
            Path del modelo guardado
        
        Raises:
            ValueError: Si el modelo no ha sido entrenado
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute train_model() primero.")
        
        logger.info("💾 Paso 3/5: Guardando modelo...")
        
        try:
            model_path = custom_path or (MODELS_DIR / self.model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            logger.info(f"✅ Modelo guardado en: {model_path}")
            
            return model_path
        
        except Exception as e:
            logger.error(f"❌ Error al guardar modelo: {e}")
            raise
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Realizar predicciones con el modelo entrenado.
        
        Args:
            X: Features para predecir (usa X_test si es None)
        
        Returns:
            Predicciones
        
        Raises:
            ValueError: Si el modelo no ha sido entrenado
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute train_model() primero.")
        
        logger.info("🔮 Paso 4/5: Realizando predicciones...")
        
        try:
            # Usar X_test si no se proporciona X
            X_pred = X if X is not None else self.X_test
            
            if X_pred is None:
                raise ValueError("No hay datos para predecir. Proporcione X o cargue datos.")
            
            # Inicializar predictor si no existe
            if self.predictor is None:
                self.predictor = ModelPredictor(self.model_filename)
                self.predictor.model = self.model  # Usar modelo en memoria
            
            predictions = self.predictor.predict(X_pred)
            
            logger.info(f"✅ Predicciones generadas: {len(predictions)} muestras")
            
            return predictions
        
        except Exception as e:
            logger.error(f"❌ Error al predecir: {e}")
            raise
    
    def evaluate(self, save_artifacts: bool = True) -> Dict:
        """
        Evaluar el modelo en el conjunto de prueba.
        
        Args:
            save_artifacts: Si guardar reportes y gráficas
        
        Returns:
            Diccionario con métricas de evaluación
        
        Raises:
            ValueError: Si el modelo no ha sido entrenado o datos no cargados
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado. Ejecute train_model() primero.")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Datos de prueba no cargados. Ejecute load_data() primero.")
        
        logger.info("📈 Paso 5/5: Evaluando modelo...")
        
        try:
            self.evaluator = ModelEvaluator(self.model_filename)
            self.evaluator.predictor.model = self.model  # Usar modelo en memoria
            
            # Generar reporte completo
            results = self.evaluator.generate_full_report(
                self.X_test, 
                self.y_test,
                save_artifacts=save_artifacts
            )
            
            metrics = results['metrics']
            
            logger.info(f"✅ Evaluación completada")
            logger.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"   - Precision (macro): {metrics['precision_macro']:.4f}")
            logger.info(f"   - Recall (macro): {metrics['recall_macro']:.4f}")
            logger.info(f"   - F1 (macro): {metrics['f1_macro']:.4f}")
            
            if save_artifacts:
                logger.info(f"   - Reportes guardados en: {self.evaluator.report_path}")
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Error al evaluar: {e}")
            raise
    
    def run(
        self, 
        save_model: bool = True,
        save_artifacts: bool = True,
        log_mlflow: bool = False
    ) -> Dict:
        """
        Ejecutar el pipeline completo de principio a fin.
        
        Args:
            save_model: Si guardar el modelo entrenado
            save_artifacts: Si guardar reportes y visualizaciones
            log_mlflow: Si registrar en MLflow
        
        Returns:
            Diccionario con resultados completos del pipeline
        """
        logger.info("=" * 70)
        logger.info("🚀 INICIANDO PIPELINE COMPLETO DE ML")
        logger.info("=" * 70)
        
        try:
            # 1. Cargar datos
            self.load_data()
            
            # 2. Entrenar modelo
            self.train_model()
            
            # 3. Guardar modelo
            if save_model:
                model_path = self.save_model()
            else:
                model_path = None
            
            # 4. Predecir
            predictions = self.predict()
            
            # 5. Evaluar
            eval_results = self.evaluate(save_artifacts=save_artifacts)
            
            # 6. Log a MLflow (opcional)
            if log_mlflow and self.evaluator:
                self.evaluator.log_to_mlflow(run_name=self.model_config.name)
            
            # Resultados finales
            results = {
                'model': self.model,
                'model_path': model_path,
                'predictions': predictions,
                'evaluation': eval_results,
                'config': {
                    'model_config': self.model_config,
                    'training_config': self.training_config
                }
            }
            
            logger.info("=" * 70)
            logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info("=" * 70)
            
            return results
        
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"❌ PIPELINE FALLÓ: {e}")
            logger.error("=" * 70)
            raise


def run_baseline_pipeline(
    mlflow_enabled: bool = False,
    save_artifacts: bool = True
) -> Dict:
    """
    Función de conveniencia para ejecutar pipeline baseline.
    
    Args:
        mlflow_enabled: Si habilitar logging a MLflow
        save_artifacts: Si guardar reportes y modelos
    
    Returns:
        Resultados del pipeline
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Configuración del modelo baseline
    model_config = ModelConfig(
        name="RandomForest_Baseline",
        model_class=RandomForestClassifier,
        hyperparameters={
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42,
            "n_jobs": -1
        }
    )
    
    # Configuración de entrenamiento
    training_config = TrainingConfig(
        cv_folds=5,
        scoring="accuracy",
        mlflow_tracking_uri="http://127.0.0.1:5001" if mlflow_enabled else None,
        mlflow_experiment="Equipo24-MER"
    )
    
    # Crear y ejecutar pipeline
    pipeline = MLPipeline(
        model_config=model_config,
        training_config=training_config,
        model_filename="baseline_model.pkl"
    )
    
    results = pipeline.run(
        save_model=save_artifacts,
        save_artifacts=save_artifacts,
        log_mlflow=mlflow_enabled
    )
    
    return results
