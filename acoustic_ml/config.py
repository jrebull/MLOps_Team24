from pathlib import Path

TRAIN_PATH = Path("data/processed/X_train.csv")
PROD_PATH = Path("data/processed/X_prod.csv")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
DRIFT_REPORT_PATH = REPORTS_DIR / "data_drift_report.json"

# Configuración de test estadístico
ALPHA = 0.05  # Nivel de significancia para KS test
