import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_csv(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.path}")
        
        df = pd.read_csv(self.path)
        logger.info("Dataset cargado: %s (%d filas × %d columnas)", self.path.name, df.shape[0], df.shape[1])
        return df
