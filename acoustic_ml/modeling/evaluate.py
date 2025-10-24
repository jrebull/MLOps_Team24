"""
Model evaluation module with comprehensive metrics and MLflow integration.
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

# Configure logging
logger = logging.getLogger(__name__)

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:
    """
    Evaluates trained models with comprehensive metrics and visualizations.
    
    Attributes:
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
        Initialize evaluator with model and report directory.
        
        Args:
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
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: True labels
        
        Returns:
            Dictionary with evaluation metrics
        
        Raises:
            ValueError: If inputs are invalid
        """
        if X_test.empty or y_test.empty:
            raise ValueError("Test data cannot be empty")
        
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test ({len(X_test)}) and y_test ({len(y_test)}) must have same length"
            )
        
        logger.info(f"Starting evaluation on {len(X_test)} test samples")
        
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
            'n_samples': len(X_test)
        }
        
        logger.info(f"✅ Evaluation complete - Accuracy: {self.metrics['accuracy']:.4f}")
        
        return self.metrics
    
    def generate_classification_report(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            X_test: Test features
            y_test: True labels
            save: Whether to save report to file
        
        Returns:
            Classification report as string
        """
        y_pred = self.predictor.predict(X_test)
        
        report = classification_report(y_test, y_pred)
        
        if save:
            report_file = self.report_path / "classification_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"📄 Classification report saved to {report_file}")
        
        return report
    
    def plot_confusion_matrix(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save: bool = True,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot and optionally save confusion matrix.
        
        Args:
            X_test: Test features
            y_test: True labels
            save: Whether to save plot to file
            figsize: Figure size
        
        Returns:
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
            logger.info(f"📊 Confusion matrix saved to {plot_file}")
        
        return fig
    
    def save_metrics(self, filename: str = "metrics.json") -> None:
        """
        Save computed metrics to JSON file.
        
        Args:
            filename: Name of the output file
        """
        if not self.metrics:
            logger.warning("No metrics to save. Run evaluate() first.")
            return
        
        metrics_file = self.report_path / filename
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        logger.info(f"💾 Metrics saved to {metrics_file}")
    
    def log_to_mlflow(self, run_name: Optional[str] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            run_name: Optional name for the MLflow run
        """
        try:
            import mlflow
            
            if not self.metrics:
                logger.warning("No metrics to log. Run evaluate() first.")
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
            
            logger.info("✅ Metrics logged to MLflow")
        
        except ImportError:
            logger.warning("MLflow not available. Skipping MLflow logging.")
        except Exception as e:
            logger.error(f"❌ Failed to log to MLflow: {e}")
    
    def generate_full_report(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        save_artifacts: bool = True
    ) -> Dict:
        """
        Generate complete evaluation report with all metrics and visualizations.
        
        Args:
            X_test: Test features
            y_test: True labels
            save_artifacts: Whether to save all artifacts to disk
        
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Generating full evaluation report...")
        
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
        
        logger.info("✅ Full evaluation report generated")
        
        return results


# Convenience function for quick evaluation
def evaluate_model(
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_artifacts: bool = True,
    log_mlflow: bool = False
) -> Dict:
    """
    Quick model evaluation with all metrics and visualizations.
    
    Args:
        model_name: Name of the model file
        X_test: Test features
        y_test: True labels
        save_artifacts: Whether to save reports and plots
        log_mlflow: Whether to log to MLflow
    
    Returns:
        Dictionary with evaluation results
    """
    evaluator = ModelEvaluator(model_name)
    results = evaluator.generate_full_report(X_test, y_test, save_artifacts)
    
    if log_mlflow:
        evaluator.log_to_mlflow()
    
    return results
