from evidently import Report
from evidently.presets import DataDriftPreset
import pandas as pd
from pathlib import Path
from .drift_config import DriftConfig

class DriftDetector:
    def __init__(self, train_path=DriftConfig.TRAIN_PATH, prod_path=DriftConfig.PROD_PATH):
        self.train_path = train_path
        self.prod_path = prod_path
        self.train_df = None
        self.prod_df = None

    def load_data(self):
        self.train_df = pd.read_csv(self.train_path)
        self.prod_df = pd.read_csv(self.prod_path)
        print(f"Datos cargados: train {self.train_df.shape}, prod {self.prod_df.shape}")

    def generate_report(self, output_json=DriftConfig.DRIFT_JSON_REPORT):
        if self.train_df is None or self.prod_df is None:
            self.load_data()
        report = Report(metrics=[DataDriftPreset()])
        result = report.run(current_data=self.prod_df, reference_data=self.train_df)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        result.save_json(output_json)
        print(f"Reporte de drift guardado en JSON: {output_json}")
