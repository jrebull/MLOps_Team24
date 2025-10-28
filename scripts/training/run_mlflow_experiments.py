"""
MLflow Experiments Runner for Turkish Music Emotion Recognition
================================================================

Clean implementation using existing acoustic_ml infrastructure.
No external data loading - uses pre-processed splits from DatasetManager.

Author: MLOps Team 24
Date: November 2024
Phase: 2 - Task 4

Following Cookiecutter Data Science structure and MLOps best practices.
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Import existing acoustic_ml infrastructure
from acoustic_ml.dataset import DatasetManager
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline


class MLflowExperimentRunner:
    """
    Manages MLflow experiments for Turkish Music Emotion Recognition.
    
    Uses existing acoustic_ml infrastructure:
    - DatasetManager for loading pre-split data
    - create_sklearn_pipeline for model pipelines
    """
    
    def __init__(
        self,
        experiment_name: str = "turkish-music-emotion-recognition",
        tracking_uri: str = "./mlruns"
    ):
        """
        Initialize experiment runner.
        
        Args:
            experiment_name: Name for MLflow experiment
            tracking_uri: MLflow tracking server URI (local filesystem)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        print(f"🎯 MLflow Tracking URI: {tracking_uri}")
        print(f"📊 Experiment: {experiment_name}")
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load pre-split dataset using existing DatasetManager.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n📂 Loading pre-processed dataset splits...")
        
        # Use existing DatasetManager infrastructure
        dataset_manager = DatasetManager()
        X_train, X_test, y_train, y_test = dataset_manager.load_train_test_split(
            validate=True
        )
        
        print(f"   ✓ Train set: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"   ✓ Test set:  {len(X_test)} samples, {X_test.shape[1]} features")
        print(f"   ✓ Classes: {sorted(y_train.unique())}")
        print(f"   ✓ Using validated pre-processed splits")
        
        return X_train, X_test, y_train, y_test
    
    def get_model_configs(self) -> List[Dict]:
        """
        Define model configurations to test.
        Uses acoustic_ml's create_sklearn_pipeline API.
        
        Returns:
            List of model configuration dictionaries with model_type and model_params
        """
        configs = [
            {
                'name': 'Random_Forest_Current_Best',
                'description': 'Current production model (80.17% baseline)',
                'model_type': 'random_forest',
                'model_params': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            {
                'name': 'Random_Forest_Deep',
                'description': 'Deeper RF with more trees',
                'model_type': 'random_forest',
                'model_params': {
                    'n_estimators': 300,
                    'max_depth': 25,
                    'min_samples_split': 3,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            {
                'name': 'Random_Forest_Simple',
                'description': 'Simpler RF to prevent overfitting',
                'model_type': 'random_forest',
                'model_params': {
                    'n_estimators': 150,
                    'max_depth': 15,
                    'min_samples_split': 8,
                    'min_samples_leaf': 4,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            {
                'name': 'Gradient_Boosting',
                'description': 'Gradient Boosting ensemble method',
                'model_type': 'gradient_boosting',
                'model_params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            {
                'name': 'Gradient_Boosting_Conservative',
                'description': 'GB with lower learning rate',
                'model_type': 'gradient_boosting',
                'model_params': {
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'random_state': 42
                }
            },
            {
                'name': 'Logistic_Regression_Baseline',
                'description': 'Linear baseline model',
                'model_type': 'logistic_regression',
                'model_params': {
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            {
                'name': 'SVM_RBF',
                'description': 'Support Vector Machine with RBF kernel',
                'model_type': 'svm',
                'model_params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'random_state': 42
                }
            }
        ]
        
        return configs
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        labels: List[str],
        save_path: Path
    ) -> np.ndarray:
        """
        Create and save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of class labels
            save_path: Path to save the plot
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Happy', 'Sad', 'Angry', 'Relax'],
            yticklabels=['Happy', 'Sad', 'Angry', 'Relax']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def run_experiment(
        self,
        config: Dict,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        run_number: int
    ) -> Dict:
        """
        Run a single MLflow experiment.
        
        Args:
            config: Model configuration dictionary
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            run_number: Experiment run number
            
        Returns:
            Dictionary with experiment results
        """
        model_name = f"exp_{run_number:02d}_{config['name']}"
        
        print(f"\n{'='*70}")
        print(f"🚀 Experiment {run_number}: {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"{'='*70}")
        
        with mlflow.start_run(run_name=model_name):
            # ===== LOG PARAMETERS =====
            print(f"   📝 Logging parameters...")
            mlflow.log_param("model_type", config['model_type'])
            mlflow.log_param("scale_method", "robust")
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            
            # Log all model hyperparameters
            for param_name, param_value in config['model_params'].items():
                mlflow.log_param(f"model_{param_name}", param_value)
            
            # ===== TRAIN MODEL =====
            print(f"   🎯 Training {config['model_type']} pipeline...")
            
            # Create pipeline using existing infrastructure
            pipeline = create_sklearn_pipeline(
                model_type=config['model_type'],
                model_params=config['model_params'],
                scale_method="robust"
            )
            
            # Train with timing
            start_time = datetime.now()
            pipeline.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   ✅ Training completed in {training_time:.2f}s")
            
            # ===== PREDICTIONS =====
            print(f"   🔮 Making predictions...")
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)
            
            # ===== CALCULATE METRICS =====
            print(f"   📊 Calculating metrics...")
            
            # Training metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
            train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
            train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
            
            # Test metrics
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            # Overfitting indicator
            overfitting_gap = train_accuracy - test_accuracy
            
            # ===== LOG METRICS =====
            print(f"   💾 Logging metrics to MLflow...")
            
            # Training metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)
            
            # Test metrics
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)
            
            # Other metrics
            mlflow.log_metric("overfitting_gap", overfitting_gap)
            mlflow.log_metric("training_time_seconds", training_time)
            
            print(f"\n   📈 Results:")
            print(f"      Train Accuracy: {train_accuracy:.4f}")
            print(f"      Test Accuracy:  {test_accuracy:.4f}")
            print(f"      Test F1-Score:  {test_f1:.4f}")
            print(f"      Overfitting:    {overfitting_gap:.4f}")
            
            # ===== CREATE ARTIFACTS =====
            print(f"   🎨 Creating visualization artifacts...")
            
            # Create artifacts directory
            artifacts_dir = Path("mlflow_artifacts") / model_name
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            cm_path = artifacts_dir / "confusion_matrix.png"
            self.plot_confusion_matrix(
                y_test,
                y_test_pred,
                labels=sorted(y_test.unique()),
                save_path=cm_path
            )
            mlflow.log_artifact(str(cm_path))
            
            # Classification report
            report = classification_report(y_test, y_test_pred)
            report_path = artifacts_dir / "classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(f"Classification Report - {model_name}\n")
                f.write(f"{'='*70}\n")
                f.write(f"Description: {config['description']}\n")
                f.write(f"Model Type: {config['model_type']}\n")
                f.write(f"Scaler: RobustScaler (acoustic_ml default)\n")
                f.write(f"{'='*70}\n\n")
                f.write(report)
                f.write(f"\n\nOverfitting Gap: {overfitting_gap:.4f}\n")
                f.write(f"Training Time: {training_time:.2f}s\n")
            mlflow.log_artifact(str(report_path))
            
            # Save metrics summary
            metrics_path = artifacts_dir / "metrics_summary.txt"
            with open(metrics_path, 'w') as f:
                f.write(f"Metrics Summary - {model_name}\n")
                f.write(f"{'='*70}\n\n")
                f.write(f"Training Metrics:\n")
                f.write(f"  Accuracy:  {train_accuracy:.4f}\n")
                f.write(f"  Precision: {train_precision:.4f}\n")
                f.write(f"  Recall:    {train_recall:.4f}\n")
                f.write(f"  F1-Score:  {train_f1:.4f}\n\n")
                f.write(f"Test Metrics:\n")
                f.write(f"  Accuracy:  {test_accuracy:.4f}\n")
                f.write(f"  Precision: {test_precision:.4f}\n")
                f.write(f"  Recall:    {test_recall:.4f}\n")
                f.write(f"  F1-Score:  {test_f1:.4f}\n\n")
                f.write(f"Overfitting Gap: {overfitting_gap:.4f}\n")
                f.write(f"Training Time: {training_time:.2f}s\n")
            mlflow.log_artifact(str(metrics_path))
            
            # ===== LOG MODEL (MLflow 3.4.0 compatible) =====
            print(f"   💾 Logging model to MLflow...")
            
            try:
                # Prepare signature and input example
                sample_size = min(5, len(X_train))
                if hasattr(X_train, 'iloc'):  # DataFrame
                    X_sample = X_train.iloc[:sample_size]
                else:  # numpy array
                    X_sample = X_train[:sample_size]
                
                # Infer signature
                predictions = pipeline.predict(X_sample)
                signature = infer_signature(X_sample, predictions)
                
                # MLflow 3.4.0: Use save_model + log_artifacts for robust saving
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = Path(tmpdir) / "model"
                    
                    # Save model to temporary directory
                    mlflow.sklearn.save_model(
                        sk_model=pipeline,
                        path=str(model_path),
                        signature=signature,
                        input_example=X_sample
                    )
                    
                    # Log entire model directory as artifact
                    mlflow.log_artifacts(str(model_path), artifact_path="model")
                    
                print(f"   ✅ Model saved successfully as artifact 'model/'")
                
            except Exception as e:
                print(f"   ⚠️  Warning: Model logging failed - {str(e)}")
                print(f"   ℹ️  Experiment metrics still logged, continuing...")
            
            # ===== LOG TAGS =====
            mlflow.set_tag("team", "MLOps Team 24")
            mlflow.set_tag("project", "Turkish Music Emotion Recognition")
            mlflow.set_tag("institution", "ITESM")
            mlflow.set_tag("phase", "2")
            mlflow.set_tag("task", "4")
            mlflow.set_tag("scaler", "robust")
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            mlflow.set_tag("model_family", config['model_type'])
            mlflow.set_tag("infrastructure", "acoustic_ml")
            
            print(f"   ✅ Experiment logged successfully!")
            
            return {
                'run_number': run_number,
                'model_name': model_name,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'overfitting_gap': overfitting_gap,
                'training_time': training_time
            }
    
    def run_all_experiments(self) -> List[Dict]:
        """
        Run all experiments and return results summary.
        
        Returns:
            List of experiment results
        """
        print("\n" + "="*70)
        print("🚀 STARTING MLFLOW EXPERIMENTS")
        print("   Turkish Music Emotion Recognition - Phase 2 Task 4")
        print("="*70)
        
        # Load data once using existing infrastructure
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Get model configurations
        configs = self.get_model_configs()
        
        print(f"\n📋 Total experiments to run: {len(configs)}")
        print(f"📊 Tracking URI: {self.tracking_uri}")
        print(f"🏷️  Experiment name: {self.experiment_name}")
        
        # Run experiments
        results = []
        for i, config in enumerate(configs, 1):
            try:
                result = self.run_experiment(
                    config, X_train, X_test, y_train, y_test, i
                )
                results.append(result)
            except Exception as e:
                print(f"\n   ❌ Error in experiment {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # ===== GENERATE SUMMARY =====
        print("\n" + "="*70)
        print("📊 EXPERIMENTS SUMMARY")
        print("="*70)
        
        if results:
            # Create results dataframe
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('test_accuracy', ascending=False)
            
            # Display summary table
            print("\n" + results_df.to_string(index=False))
            
            # Highlight best model
            best = results_df.iloc[0]
            print(f"\n{'='*70}")
            print(f"🏆 BEST MODEL: {best['model_name']}")
            print(f"{'='*70}")
            print(f"   Test Accuracy:  {best['test_accuracy']:.4f}")
            print(f"   Test F1-Score:  {best['test_f1']:.4f}")
            print(f"   Test Precision: {best['test_precision']:.4f}")
            print(f"   Test Recall:    {best['test_recall']:.4f}")
            print(f"   Overfitting:    {best['overfitting_gap']:.4f}")
            print(f"   Training Time:  {best['training_time']:.2f}s")
            print(f"{'='*70}")
            
            # Save summary to CSV
            summary_path = Path("mlflow_artifacts") / "experiments_summary.csv"
            results_df.to_csv(summary_path, index=False)
            print(f"\n💾 Summary saved to: {summary_path}")
            
            # Save detailed report
            report_path = Path("mlflow_artifacts") / "experiments_report.txt"
            with open(report_path, 'w') as f:
                f.write("MLflow Experiments Report\n")
                f.write("Turkish Music Emotion Recognition - Phase 2 Task 4\n")
                f.write("="*70 + "\n\n")
                f.write(f"Team: MLOps Team 24\n")
                f.write(f"Institution: ITESM\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Experiments: {len(results)}\n")
                f.write(f"Infrastructure: acoustic_ml\n\n")
                f.write("="*70 + "\n\n")
                f.write("RESULTS SUMMARY (sorted by accuracy):\n\n")
                f.write(results_df.to_string(index=False))
                f.write(f"\n\n{'='*70}\n")
                f.write(f"BEST MODEL: {best['model_name']}\n")
                f.write(f"{'='*70}\n")
                f.write(f"Test Accuracy:  {best['test_accuracy']:.4f}\n")
                f.write(f"Test F1-Score:  {best['test_f1']:.4f}\n")
                f.write(f"Test Precision: {best['test_precision']:.4f}\n")
                f.write(f"Test Recall:    {best['test_recall']:.4f}\n")
                f.write(f"Overfitting:    {best['overfitting_gap']:.4f}\n")
                f.write(f"Training Time:  {best['training_time']:.2f}s\n")
            
            print(f"📄 Detailed report saved to: {report_path}")
        
        print("\n" + "="*70)
        print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\n🌐 View results in MLflow UI:")
        print(f"   1. Run: mlflow ui --port 5001")
        print(f"   2. Open: http://localhost:5001")
        print(f"   3. Select experiment: {self.experiment_name}")
        print("="*70 + "\n")
        
        return results


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("🎯 MLflow Experiments Runner")
    print("   Turkish Music Emotion Recognition - Phase 2 Task 4")
    print("   MLOps Team 24 - ITESM")
    print("="*70)
    print("\n💡 Using existing acoustic_ml infrastructure:")
    print("   • DatasetManager for pre-processed splits")
    print("   • create_sklearn_pipeline for model pipelines")
    print("   • Validated and tested components")
    print("="*70)
    
    # Initialize runner with local filesystem tracking
    runner = MLflowExperimentRunner(
        experiment_name="turkish-music-emotion-recognition",
        tracking_uri="./mlruns"
    )
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    if results:
        print("\n🎉 Experiments completed successfully!")
        print("\n📝 Next Steps:")
        print("   1. Start MLflow UI: mlflow ui --port 5001")
        print("   2. Open browser: http://localhost:5001")
        print("   3. Compare experiments visually")
        print("   4. Register best model in Model Registry")
        print("   5. Take screenshots for documentation")
        print("   6. Update project README with results")
    else:
        print("\n⚠️  No experiments completed successfully.")
        print("   Check error messages above.")
    

if __name__ == "__main__":
    main()