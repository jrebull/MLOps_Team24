from pathlib import Path

class DriftConfig:
    TRAIN_PATH = Path("data/processed/X_train.csv")
    PROD_PATH = Path("data/processed/data_drift_prod.csv")
    DRIFT_JSON_REPORT = Path("reports/drift/drift_report.json")
