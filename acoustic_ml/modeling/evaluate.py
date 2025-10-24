"""
Módulo de evaluación de modelos con métricas completas e integración con MLflow.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from acoustic_ml.config import MODELS_DIR, REPORTS_DIR
from acoustic_ml.modeling.predict import ModelPredictor

# Configurar logging
logger = logging.getLogger(__name__)

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:
    """
    Evalúa modelos entrenados con métricas completas y visualizaciones.
    
    Atributos:
        predictor (ModelPredictor): Predictor instance for inference
        metrics (Dict): Computed evaluation metrics
        report_path (Path): Path to save evaluation reports
    """
    
    def __init__(
        self, 
        model_name: str = "baseline_model.pkl",
        report_dir: Path = REPORTS_DIR
    ):
        """
        Inicializar evaluador con modelo y directorio de reportes.
        
        Argumentos:
            model_name: Name of the model file to evaluate
            report_dir: Directory to save evaluation reports
        """
        self.predictor = ModelPredictor(model_name)
        self.predictor.load_model()
        self.metrics = {}
        self.report_path = report_dir / f"{model_name.replace('.pkl', '')}_evaluation"
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelEvaluator initialized for {model_name}")
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluar modelo con datos de prueba.
        
        Argumentos:
            X_test: Test features
            y_test: True labels
        
        Retorna:
            Dictionary with evaluation metrics
        
        Lanza:
            ValueError: If inputs are invalid
        """
        if X_test.empty or y_test.empty:
            raise ValueError("Los datos de prueba no pueden estar vacíos")
        
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test ({len(X_test)}) and y_test ({len(y_test)}) deben tener la misma longitud"
            )
        
        logger.info(f"Iniciando evaluación en {len(X_test)} test muestras")
        
        # Get predictions
        y_pred = self.predictor.predict(X_test)
        
        # Calculate metrics
        self.metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
            'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'n_muestras': len(X_test)
        }
        
        logger.info(f"✅ Evaluación completa - Accuracy: {self.metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def generate_classification_report(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True
    ) -> str:
        """
        Generar reporte de clasificación detallado.
        
        Argumentos:
            X_test: Test features
            y_test: True labels
            save: Whether to save report to file
        
        Retorna:
            Classification report as string
        """
        y_pred = self.predictor.predict(X_test)
        
        report = classification_report(y_test, y_pred)
        
        if save:
            report_file = self.report_path / "classification_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"📄 Reporte de clasificación guardado en {report_file}")
        
        return report
    
    def plot_confusion_matrix(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Graficar y opcionalmente guardar matriz de confusión.
        
        Argumentos:
            X_test: Test features
            y_test: True labels
            save: Whether to save plot to file
            figsize: Figure size
        
        Retorna:
            Matplotlib figure object
        """
        y_pred = self.predictor.predict(X_test)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Get class labels
        labels = sorted(y_test.unique())
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plot_file = self.report_path / "confusion_matrix.png"
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Matriz de confusión guardada en {plot_file}")
        
        return fig
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        Guardar métricas computadas en archivo JSON.
        
        Argumentos:
            filename: Name of the output file
        """
        if not self.metrics:
            logger.warning("No hay métricas para guardar. Ejecute evaluate() primero.")
            return
        
        metrics_file = self.report_path / filename
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"💾 Métricas guardadas en {metrics_file}")
    
    def log_to_mlflow(self, run_name: Optional[str] = None) -> None:
        """
        Registrar métricas en MLflow.
        
        Argumentos:
            run_name: Optional name for the MLflow run
        """
        try:
            import mlflow
            
            if not self.metrics:
                logger.warning("No hay métricas para registrar. Ejecute evaluate() primero.")
                return
            
            if run_name:
                mlflow.set_tag("run_name", run_name)
            
            # Log all metrics
            for metric_name, metric_value in self.metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)
            
            # Log artifacts if they exist
            if self.report_path.exists():
                mlflow.log_artifacts(str(self.report_path))
            
            logger.info("✅ Métricas registradas en MLflow")
        
        except ImportError:
            logger.warning("MLflow no disponible. Omitiendo registro en MLflow.")
        except Exception as e:
            logger.error(f"❌ Falló al registrar en MLflow: {e}")
    
    def generate_full_report(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save_artifacts: bool = True
    ) -> Dict:
        """
        Generar reporte completo de evaluación con todas las métricas y visualizaciones.
        
        Argumentos:
            X_test: Test features
            y_test: True labels
            save_artifacts: Whether to save all artifacts to disk
        
        Retorna:
            Dictionary with all evaluation results
        """
        logger.info("Generando reporte completo de evaluación...")
        
        # Calculate metrics
        metrics = self.evaluate(X_test, y_test)
        
        # Generate classification report
        clf_report = self.generate_classification_report(X_test, y_test, save=save_artifacts)
        
        # Generate confusion matrix
        self.plot_confusion_matrix(X_test, y_test, save=save_artifacts)
        
        # Save metrics
        if save_artifacts:
            self.save_metrics()
        
        results = {
            'metrics': metrics,
            'classification_report': clf_report,
            'report_path': str(self.report_path) if save_artifacts else None
        }
        
        logger.info("✅ Reporte completo de evaluación generado")
        
        return results


# Función de conveniencia para evaluación rápida
def evaluate_model(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_artifacts: bool = True,
    log_mlflow: bool = False
) -> Dict:
    """
    Evaluación rápida de modelo con todas las métricas y visualizaciones.
    
    Argumentos:
        model_name: Name of the model file
        X_test: Test features
        y_test: True labels
        save_artifacts: Whether to save reports and plots
        log_mlflow: Whether to log to MLflow
    
    Retorna:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator(model_name)
    results = evaluator.generate_full_report(X_test, y_test, save_artifacts)
    
    if log_mlflow:
        evaluator.log_to_mlflow()
    
    return results
