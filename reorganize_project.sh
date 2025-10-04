#!/bin/bash

# Script para reorganizar MLOps_Team24 con estructura Cookiecutter Data Science
# Autor: Asistente Claude
# Fecha: 2025-10-03

set -e  # Salir si hay algún error

echo "🚀 Reorganizando proyecto MLOps_Team24 con estructura Cookiecutter..."
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -f "dvc.yaml" ] || [ ! -d ".dvc" ]; then
    echo "❌ Error: Este script debe ejecutarse desde /Users/haowei/Documents/MLOps/MNA_Team24/MLOps_Team24"
    exit 1
fi

echo "📁 Creando estructura de directorios..."

# Crear estructura de datos
mkdir -p data/external
mkdir -p data/interim
mkdir -p data/processed

# Crear estructura de notebooks
mkdir -p notebooks

# Crear estructura de reports
mkdir -p reports/figures

# Crear estructura de references
mkdir -p references

# Crear módulo Python principal
mkdir -p acoustic_ml/modeling

echo "✅ Directorios creados"
echo ""

echo "📦 Moviendo notebooks a carpeta notebooks/..."
if [ -f "Fase1_equipo24.ipynb" ]; then
    mv Fase1_equipo24.ipynb notebooks/
    echo "  ✓ Fase1_equipo24.ipynb movido"
fi

if [ -f "NoteBook Testing.ipynb" ]; then
    mv "NoteBook Testing.ipynb" notebooks/
    echo "  ✓ NoteBook Testing.ipynb movido"
fi

echo ""
echo "🐍 Creando módulo Python acoustic_ml..."

# Crear __init__.py principal
cat > acoustic_ml/__init__.py << 'EOF'
"""
Acoustic ML - MLOps Team 24
Proyecto de Machine Learning para análisis de características acústicas
"""

__version__ = "0.1.0"
EOF

# Crear config.py
cat > acoustic_ml/config.py << 'EOF'
"""
Configuración del proyecto
"""
import os
from pathlib import Path

# Directorios base
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

# Subdirectorios de datos
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Configuración de MLflow
MLFLOW_TRACKING_URI = "file:///" + str(PROJECT_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "acoustic_ml_experiments"

# Configuración de DVC
DVC_REMOTE_NAME = "local"
DVC_REMOTE_URL = str(PROJECT_DIR / "dvcstore")
EOF

# Crear dataset.py
cat > acoustic_ml/dataset.py << 'EOF'
"""
Scripts para cargar y procesar datasets
"""
import pandas as pd
from pathlib import Path
from acoustic_ml.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_raw_data(filename: str = "acoustic_features.csv") -> pd.DataFrame:
    """
    Carga datos crudos desde data/raw/
    
    Args:
        filename: Nombre del archivo a cargar
        
    Returns:
        DataFrame con los datos crudos
    """
    filepath = RAW_DATA_DIR / filename
    return pd.read_csv(filepath)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """
    Guarda datos procesados en data/processed/
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo de salida
    """
    filepath = PROCESSED_DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"✅ Datos guardados en {filepath}")
EOF

# Crear features.py
cat > acoustic_ml/features.py << 'EOF'
"""
Feature engineering para características acústicas
"""
import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features adicionales a partir del dataset
    
    Args:
        df: DataFrame con datos originales
        
    Returns:
        DataFrame con features adicionales
    """
    # Aquí va tu lógica de feature engineering
    df_features = df.copy()
    
    # Ejemplo: agregar features derivadas
    # df_features['feature_ratio'] = df_features['col1'] / df_features['col2']
    
    return df_features


def select_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """
    Selecciona un subconjunto de features
    
    Args:
        df: DataFrame con todas las features
        feature_list: Lista de nombres de columnas a seleccionar
        
    Returns:
        DataFrame con features seleccionadas
    """
    return df[feature_list]
EOF

# Crear modeling/__init__.py
cat > acoustic_ml/modeling/__init__.py << 'EOF'
"""
Módulo de modelado
"""
EOF

# Mover train_baseline.py al módulo y crear train.py
if [ -f "scripts/train_baseline.py" ]; then
    cp scripts/train_baseline.py acoustic_ml/modeling/train.py
    echo "  ✓ train_baseline.py copiado a acoustic_ml/modeling/train.py"
fi

# Crear predict.py
cat > acoustic_ml/modeling/predict.py << 'EOF'
"""
Inference con modelos entrenados
"""
import pickle
import pandas as pd
from pathlib import Path
from acoustic_ml.config import MODELS_DIR


def load_model(model_name: str = "baseline_model.pkl"):
    """
    Carga un modelo entrenado
    
    Args:
        model_name: Nombre del archivo del modelo
        
    Returns:
        Modelo cargado
    """
    model_path = MODELS_DIR / model_name
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza predicciones con un modelo
    
    Args:
        model: Modelo entrenado
        X: Features para predicción
        
    Returns:
        DataFrame con predicciones
    """
    predictions = model.predict(X)
    return pd.DataFrame({'predictions': predictions})
EOF

# Crear plots.py
cat > acoustic_ml/plots.py << 'EOF'
"""
Visualizaciones para análisis y reportes
"""
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from acoustic_ml.config import REPORTS_DIR


def save_figure(fig, filename: str, subfolder: str = "figures") -> None:
    """
    Guarda una figura en reports/figures/
    
    Args:
        fig: Figura de matplotlib
        filename: Nombre del archivo
        subfolder: Subcarpeta dentro de reports
    """
    save_path = REPORTS_DIR / subfolder / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Figura guardada en {save_path}")


def plot_feature_importance(feature_importance: dict, top_n: int = 20) -> plt.Figure:
    """
    Visualiza importancia de features
    
    Args:
        feature_importance: Diccionario {feature_name: importance}
        top_n: Número de features principales a mostrar
        
    Returns:
        Figura de matplotlib
    """
    # Ordenar features por importancia
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(features, importances)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    return fig
EOF

echo "  ✓ Módulo acoustic_ml creado con estructura completa"
echo ""

echo "📄 Actualizando .gitignore..."
cat >> .gitignore << 'EOF'

# Datos procesados
data/processed/*
data/interim/*
data/external/*
!data/processed/.gitkeep
!data/interim/.gitkeep
!data/external/.gitkeep

# Notebooks checkpoints
notebooks/.ipynb_checkpoints/

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyCharm
.idea/

# VS Code
.vscode/

EOF

echo "✅ .gitignore actualizado"
echo ""

echo "📄 Creando .gitkeep en carpetas vacías..."
touch data/external/.gitkeep
touch data/interim/.gitkeep
touch data/processed/.gitkeep
touch references/.gitkeep
touch reports/figures/.gitkeep

echo "✅ Archivos .gitkeep creados"
echo ""

echo "📝 Actualizando README.md..."
cat > README_NEW.md << 'EOF'
# Acoustic ML - MLOps Team 24

Proyecto de Machine Learning para análisis de características acústicas usando MLOps best practices.

## 🏗️ Estructura del Proyecto

```
├── LICENSE            <- Licencia del proyecto
├── Makefile           <- Comandos útiles (make data, make train, etc.)
├── README.md          <- Este archivo
│
├── data
│   ├── external       <- Datos de fuentes externas
│   ├── interim        <- Datos intermedios transformados
│   ├── processed      <- Datasets finales para modelado
│   └── raw            <- Datos originales inmutables
│
├── models             <- Modelos entrenados y serializados
│
├── notebooks          <- Jupyter notebooks para exploración
│                         Convención: número-iniciales-descripción
│                         Ej: 1.0-hw-exploratory-analysis.ipynb
│
├── reports            <- Análisis generados (HTML, PDF, etc.)
│   └── figures        <- Gráficas y figuras para reportes
│
├── references         <- Diccionarios de datos, manuales, etc.
│
├── requirements.txt   <- Dependencias del proyecto
│
├── scripts            <- Scripts auxiliares
│
├── acoustic_ml        <- Código fuente del proyecto
│   ├── __init__.py    
│   ├── config.py      <- Configuración y variables
│   ├── dataset.py     <- Scripts para cargar/generar datos
│   ├── features.py    <- Feature engineering
│   ├── plots.py       <- Visualizaciones
│   └── modeling       
│       ├── __init__.py
│       ├── train.py   <- Entrenamiento de modelos
│       └── predict.py <- Inferencia con modelos
│
├── mlruns             <- Experimentos de MLflow
├── mlartifacts        <- Artifacts de MLflow
├── dvcstore           <- Almacenamiento local de DVC
├── .dvc               <- Configuración de DVC
└── .git               <- Control de versiones Git
```

## 🚀 Quick Start

### 1. Configurar entorno

```bash
# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar DVC

```bash
# Pull de datos desde DVC
dvc pull
```

### 3. Entrenar modelo

```bash
# Usando Makefile
make train

# O directamente
python -m acoustic_ml.modeling.train
```

### 4. Ver experimentos en MLflow

```bash
mlflow ui
# Abrir http://localhost:5000
```

## 🛠️ Comandos Makefile

```bash
make data          # Procesar datos
make train         # Entrenar modelo
make predict       # Hacer predicciones
make clean         # Limpiar archivos temporales
```

## 📊 Tracking de Experimentos

Este proyecto usa:
- **DVC**: Versionado de datos y modelos
- **MLflow**: Tracking de experimentos y métricas
- **Git**: Control de versiones de código

## 👥 Equipo

MLOps Team 24

## 📝 Licencia

[Especificar licencia]
EOF

echo "✅ README_NEW.md creado (revísalo y renómbralo si te gusta)"
echo ""

echo "📄 Creando pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "acoustic_ml"
version = "0.1.0"
description = "MLOps project for acoustic features analysis"
authors = [
    {name = "Team 24", email = "team24@example.com"}
]
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "scikit-learn>=1.2.0",
    "matplotlib>=3.6.0",
    "seaborn>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "jupyter>=1.0.0",
]

[tool.black]
line-length = 88
target-version = ['py39', 'py310', 'py311', 'py312']

[tool.flake8]
max-line-length = 88
extend-ignore = ["E203", "W503"]
EOF

echo "✅ pyproject.toml creado"
echo ""

echo "🎉 ¡Reorganización completada!"
echo ""
echo "📋 Resumen de cambios:"
echo "  ✓ Estructura de carpetas Cookiecutter creada"
echo "  ✓ Notebooks movidos a notebooks/"
echo "  ✓ Módulo Python acoustic_ml creado"
echo "  ✓ Archivos de configuración actualizados"
echo "  ✓ .gitignore mejorado"
echo "  ✓ README_NEW.md generado"
echo ""
echo "📝 Próximos pasos recomendados:"
echo "  1. Revisar README_NEW.md y renombrarlo si te gusta:"
echo "     mv README_NEW.md README.md"
echo ""
echo "  2. Hacer commit de los cambios:"
echo "     git add ."
echo "     git commit -m \"Reorganizar proyecto con estructura Cookiecutter\""
echo ""
echo "  3. Actualizar imports en tus notebooks para usar el módulo acoustic_ml:"
echo "     from acoustic_ml.dataset import load_raw_data"
echo "     from acoustic_ml.modeling.train import train_model"
echo ""
echo "  4. Mover datos procesados a data/processed/ cuando los generes"
echo ""
echo "✨ ¡Tu proyecto ahora sigue las mejores prácticas de MLOps!"
