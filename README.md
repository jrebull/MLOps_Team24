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
