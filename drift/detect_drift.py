#!/usr/bin/env python3
from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

class DriftDetector:
    def __init__(self, 
                 train_path: str = "data/processed/X_train.csv",
                 prod_path: str = "data/processed/data_drift_prod.csv",
                 model_path: str = "app/models/model.joblib"):
        self.train_path = Path(train_path)
        self.prod_path = Path(prod_path)
        self.model_path = Path(model_path)
        self.train_df = None
        self.prod_df = None
        self.model = None
        self.scenarios = {}
        self.results = {}
        
    def load_data(self):
        print("\n📦 Cargando datos...")
        self.train_df = pd.read_csv(self.train_path)
        self.prod_df = pd.read_csv(self.prod_path)
        print(f"✓ Datos cargados: Train {self.train_df.shape}, Prod {self.prod_df.shape}")
    
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            print(f"✓ Modelo: {type(self.model).__name__}")
        except:
            self.model = None
    
    def extract_scenarios(self):
        if '__drift_scenario__' in self.prod_df.columns:
            for scenario in self.prod_df['__drift_scenario__'].unique():
                self.scenarios[scenario] = self.prod_df[
                    self.prod_df['__drift_scenario__'] == scenario
                ].drop(columns=['__drift_scenario__'])
            print(f"✓ Escenarios: {list(self.scenarios.keys())}")
    
    def generate_drift_report(self, scenario_name: str, scenario_data: pd.DataFrame):
        print(f"\n📊 {scenario_name.upper()}...", end="", flush=True)
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(current_data=scenario_data, reference_data=self.train_df)
        
        output_dir = Path("reports/drift")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Predicciones
        if self.model:
            preds = self.model.predict(scenario_data)
            unique, counts = np.unique(preds, return_counts=True)
            metrics = {
                'predictions': preds.tolist(),
                'unique_classes': unique.tolist(),
                'class_dist': dict(zip(unique.tolist(), counts.tolist())),
                'mean_pred': float(np.mean(preds))
            }
        else:
            metrics = {}
        
        # Guardar JSON
        json_path = output_dir / f"drift_{scenario_name}.json"
        with open(json_path, "w") as f:
            json.dump({'scenario': scenario_name, 'metrics': metrics, 'timestamp': datetime.now().isoformat()}, f)
        
        self.results[scenario_name] = metrics
        print(" ✓")
    
    def generate_all_reports(self):
        print("\n" + "="*70)
        print("🔥 DRIFT DETECTION")
        print("="*70)
        self.extract_scenarios()
        for name, data in self.scenarios.items():
            self.generate_drift_report(name, data)
    
    def print_summary(self):
        print("\n" + "="*70)
        print("✅ COMPLETADO")
        print("="*70)
        print(f"Reportes guardados en: reports/drift/")
    
    def run(self):
        print("="*70)
        print("🔥 DRIFT PIPELINE")
        print("="*70)
        self.load_data()
        self.load_model()
        self.generate_all_reports()
        self.print_summary()

if __name__ == "__main__":
    DriftDetector().run()
