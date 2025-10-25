#!/usr/bin/env python3
"""
Agregar logging de modelo a MLflow en train.py
"""

import sys
from pathlib import Path

file_path = Path("acoustic_ml/modeling/train.py")

if not file_path.exists():
    print(f"❌ Error: No se encuentra {file_path}")
    sys.exit(1)

content = file_path.read_text()

# Agregar mlflow.sklearn.log_model después de loguear métricas
old_method = """    def _log_to_mlflow(self, X, y):
        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.training_config.mlflow_experiment)
        with mlflow.start_run(run_name=self.model_config.name):
            for k, v in self.model_config.hyperparameters.items():
                mlflow.log_param(k, v if v is not None else "None")
            metrics = self.evaluate(X, y)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            logger.info(f"MLflow: {metrics}")"""

new_method = """    def _log_to_mlflow(self, X, y):
        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.training_config.mlflow_experiment)
        with mlflow.start_run(run_name=self.model_config.name):
            # Log parameters
            for k, v in self.model_config.hyperparameters.items():
                mlflow.log_param(k, v if v is not None else "None")
            
            # Log metrics
            metrics = self.evaluate(X, y)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log model as artifact
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=None  # No auto-register
            )
            
            logger.info(f"MLflow: {metrics}")
            logger.info(f"Model logged to MLflow artifacts")"""

if old_method in content:
    content = content.replace(old_method, new_method)
    file_path.write_text(content)
    print("✅ Model logging agregado a train.py")
    print("   → mlflow.sklearn.log_model() agregado")
    print("   → Modelo se guardará en artifacts/model/")
    print("   → Incluye: model.pkl, requirements.txt, MLmodel")
else:
    print("❌ No se encontró el patrón exacto")
    sys.exit(1)

print("\n📦 Artifacts que se guardarán:")
print("   • model/model.pkl - Modelo serializado")
print("   • model/requirements.txt - Dependencies")
print("   • model/MLmodel - Metadata")
print("   • model/conda.yaml - Conda env")
print("\n🎯 Próximo paso: Ejecutar nuevo experimento para ver artifacts")
