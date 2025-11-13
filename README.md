# 🎵 Acoustic ML - Turkish Music Emotion Recognition

<div align="center">

**MLOps Team 24 - Sistema profesional de reconocimiento de emociones musicales**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)](https://dvc.org/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-FF9900?logo=amazon-aws)](https://aws.amazon.com/s3/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://mlopsteam24-cookiecutter.streamlit.app)

<!-- Badges de Estado -->
[![Cookiecutter](https://img.shields.io/badge/cookiecutter-95.2%25-success?logo=cookiecutter&logoColor=white)](#-estructura-del-proyecto)
[![Tests](https://img.shields.io/badge/tests-33%20passing-success?logo=pytest&logoColor=white)](#-testing-unitarias-e-integración)
[![Code Quality](https://img.shields.io/badge/code-production--ready-brightgreen?logo=python&logoColor=white)](#-arquitectura-del-código)
[![Accuracy](https://img.shields.io/badge/accuracy-80.17%25-success?logo=tensorflow&logoColor=white)](#-modelo-y-resultados)
[![Docker](https://img.shields.io/badge/docker--ready-blue?logo=docker&logoColor=white)](#-docker--containerización)
[![Repo Status](https://img.shields.io/badge/repo-phase%203%20production-blue?logo=git&logoColor=white)](#-información-académica)

</div>

---

## 📋 Tabla de Contenidos

- [Sobre el Proyecto](#-sobre-el-proyecto)
- [Información Académica](#-información-académica)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Arquitectura del Código](#-arquitectura-del-código)
- [Modelo y Resultados](#-modelo-y-resultados)
- [MLOps Infrastructure](#-mlops-infrastructure)
- [Instalación y Configuración](#-instalación-y-configuración)
- [Uso del Sistema](#-uso-del-sistema)
- [Scripts Disponibles](#-scripts-disponibles)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [API Serving with FastAPI](#-api-serving-with-fastapi)
- [Data Drift Detection & Monitoring](#-data-drift-detection--monitoring)
- [Docker & Containerization](#-docker--containerization)
- [Publicación en DockerHub](#-publicación-en-dockerhub)
- [Reproducibility & Seeds](#-reproducibility--seeds)
- [Phase 3 Requirements Checklist](#-phase-3-requirements-checklist)
- [Project Structure](#-project-structure)
- [Streamlit App - Production Demo](#-streamlit-app---production-demo)
- [Monitoring y Validación](#-monitoring-y-validación)
- [Workflows y Contribución](#-workflows-y-contribución)
- [Equipo](#-equipo-de-desarrollo)

---

## 🎯 Sobre el Proyecto

Este repositorio implementa un sistema MLOps completo y profesional para **clasificación de emociones en música turca**, siguiendo las mejores prácticas de la industria con estructura **Cookiecutter Data Science** (95.2% de cumplimiento verificado).

### 🎵 Dataset y Objetivo

- **Dataset:** Turkish Music Emotion Dataset
- **Clases:** 4 emociones (Happy, Sad, Angry, Relax)
- **Features:** 50+ características acústicas extraídas
- **Objetivo:** Clasificación automática de emociones musicales
- **Modelo Actual:** Random Forest optimizado con 80.17% accuracy

### 🚀 Características Principales

#### MLOps Foundation
- 📊 **Versionado de datos** con DVC + AWS S3
- 🔄 **Pipelines reproducibles** automatizados
- 📈 **Experiment tracking** con MLflow
- ☁️ **Cloud storage** en S3 (mlops24-haowei-bucket)
- 🐳 **Containerización** con Docker Compose

#### Code y Arquitectura
- 🏗️ **Módulo Python profesional** (`acoustic_ml`)
- 🎯 **Pipeline sklearn end-to-end** listo para producción
- 🧪 **Testing comprehensivo** con 33 tests automatizados
- 🛡️ **Manejo robusto de outliers** y datos
- 🌐 **API REST** con FastAPI y Pydantic schemas

#### Fase 3: Production-Ready Deployment
- 🐳 **Containerización Docker** con docker-compose
- 🔍 **Data Drift Detection** con statistical monitoring
- 📡 **CI/CD Pipelines** automatizados
- ⚙️ **Health Checks** y monitoring endpoints
- 🔄 **Reproducibilidad garantizada** con seeds y DVC

#### Monitoring y Validación
- 📊 **Dashboard Streamlit** para validación Cookiecutter
- 🔍 **Validación automatizada** de entornos y datos
- 📈 **7 experimentos MLflow** documentados
- ✅ **Verificación de sincronización** DVC + Git + S3

---

## 📘 Información Académica

**Instituto Tecnológico y de Estudios Superiores de Monterrey**  
*Maestría en Inteligencia Artificial Aplicada (MNA)*

- **Curso:** Operaciones de Aprendizaje Automático
- **Periodo:** Septiembre – Diciembre 2024
- **Equipo:** N° 24
- **Fase Actual:** Fase 3 - Implementación en Producción 🚀

### 👨‍🏫 Profesores

| Rol | Nombre |
|-----|--------|
| Titular | Dr. Gerardo Rodríguez Hernández |
| Titular | Mtro. Ricardo Valdez Hernández |
| Asistente | Mtra. María Mylen Treviño Elizondo |
| Tutor | José Ángel Martínez Navarro |

---

## 🗂️ Estructura del Proyecto

Organización completa siguiendo **Cookiecutter Data Science** con 95.2% de cumplimiento verificado:

```
MLOps_Team24/
│
├── 📄 Configuración (Raíz)
│   ├── README.md              <- Este archivo ⭐
│   ├── Makefile               <- Comandos make (data, train, reproduce, etc.)
│   ├── MakefileGitOK          <- Makefile alternativo
│   ├── pyproject.toml         <- Configuración proyecto Python
│   ├── requirements.txt       <- Dependencias producción (pip freeze)
│   ├── params.yaml            <- Parámetros pipeline DVC
│   ├── dvc.yaml               <- Definición pipeline DVC
│   ├── dvc.lock               <- Lock file pipeline
│   ├── data.dvc               <- Tracking metadatos datos
│   ├── docker-compose.yml     <- Stack MLflow + MinIO
│   └── config.env             <- Variables entorno Docker
│
├── 📦 acoustic_ml/            <- Módulo Python principal ⭐
│   ├── __init__.py            
│   ├── config.py              <- Configuración global del sistema
│   ├── dataset.py             <- DatasetManager (Singleton, thread-safe)
│   ├── features.py            <- Feature engineering & transformers
│   ├── plots.py               <- Visualizaciones y gráficas
│   ├── archive/               <- Código legacy versionado
│   │   ├── dataset_legacy.py
│   │   └── features_legacy.py
│   └── modeling/              <- Submódulo de modelado
│       ├── __init__.py
│       ├── train.py           <- Training logic
│       ├── predict.py         <- Inference
│       ├── evaluate.py        <- Evaluation metrics
│       ├── pipeline.py        <- MLOps pipeline completo
│       ├── sklearn_pipeline.py <- Pipeline sklearn production-ready
│       └── *.backup           <- Backups de versiones previas
│
├── 🌐 app/                    <- API REST (FastAPI)
│   ├── main.py                <- Entry point aplicación
│   ├── api/                   
│   │   ├── main.py            <- Router principal
│   │   ├── endpoints.py       <- Endpoints API
│   │   └── schemas.py         <- Pydantic schemas
│   ├── core/                  
│   │   ├── config.py          <- Configuración API
│   │   └── logger.py          <- Logging setup
│   └── services/              
│       └── model_service.py   <- Servicio de modelo
│
├── 📊 data/                   <- Datos (versionados con DVC)
│   ├── external/              <- Fuentes externas
│   ├── interim/               <- Transformaciones intermedias
│   ├── processed/             <- Datasets finales ⭐
│   │   ├── turkish_music_emotion_v1_original.csv      (400 filas - Baseline)
│   │   ├── turkish_music_emotion_v2_cleaned_aligned.csv (400 filas)
│   │   ├── turkish_music_emotion_v2_cleaned_full.csv    (408 filas) ⭐ PRODUCCIÓN
│   │   ├── turkish_music_emotion_v2_transformed.csv
│   │   ├── eda_report.txt
│   │   ├── split_metadata.json
│   │   ├── X_train.csv, X_test.csv
│   │   └── y_train.csv, y_test.csv
│   └── raw/                   <- Datos originales inmutables
│       ├── turkis_music_emotion_original.csv     (125 KB)
│       └── turkish_music_emotion_modified.csv    (130 KB)
│
├── 💾 models/                 <- Modelos serializados
│   ├── baseline/              
│   │   ├── random_forest_baseline.pkl
│   │   ├── gradient_boosting_baseline.pkl
│   │   └── xgboost_baseline.pkl
│   ├── optimized/             <- Modelos optimizados ⭐
│   │   ├── production_model.pkl              (Modelo actual 80.17%)
│   │   ├── production_model_metadata.json
│   │   ├── best_model_*.pkl                  (Versiones fechadas)
│   │   └── model_metadata_*.json
│   ├── baseline.dvc           <- Tracking baseline models
│   ├── optimized.dvc          <- Tracking optimized models
│   ├── baseline_model.pkl     
│   └── test_model.pkl         
│
├── 📈 mlflow_artifacts/       <- Experimentos MLflow
│   ├── exp_01_Random_Forest_Current_Best/
│   ├── exp_02_Random_Forest_Deep/
│   ├── exp_03_Random_Forest_Simple/
│   ├── exp_04_Gradient_Boosting/
│   ├── exp_05_Gradient_Boosting_Conservative/
│   ├── exp_06_Logistic_Regression_Baseline/
│   ├── exp_07_SVM_RBF/
│   ├── experiments_summary.csv
│   ├── experiments_report.txt
│   └── experiment_run_*.log
│
├── 📓 notebooks/              <- Jupyter notebooks
│   ├── 1.0-team-eda-turkish-music.ipynb       (EDA inicial)
│   ├── 1.1-team-dataset-comparison.ipynb      (Comparación datasets)
│   ├── 2.0-team-preprocessing.ipynb           (Preprocessing)
│   ├── 3.0-team-modeling-evaluation.ipynb     (Modelado)
│   └── archive/               <- Notebooks legacy
│       ├── 0.0-team-testing.ipynb
│       └── 1.2-team-fase1-final.ipynb
│
├── 📊 monitoring/             <- Sistema de monitoring
│   ├── dashboard/             
│   │   ├── streamlit_dashboard.py         ⭐ Dashboard Cookiecutter
│   │   ├── validate_cookiecutter.py       
│   │   ├── requirements_dashboard.txt
│   │   └── requirements.txt
│   └── README.md              
│
├── 📈 reports/                <- Reportes y análisis
│   ├── figures/               <- Visualizaciones ⭐
│   │   ├── confusion_matrices_top3.png
│   │   ├── final_confusion_matrix.png
│   │   ├── baseline_comparison.png
│   │   ├── roc_curves.png
│   │   ├── outlier_analysis.png
│   │   ├── outlier_boxplots.png
│   │   ├── plot_*.png         (Múltiples visualizaciones EDA)
│   │   ├── outlier_analysis_report.txt
│   │   └── scaler_comparison_results.txt
│   ├── baseline_model_evaluation/
│   │   ├── classification_report.txt
│   │   ├── confusion_matrix.png
│   │   └── metrics.json
│   ├── baseline_results.csv
│   ├── hyperparameter_search_results.csv
│   ├── final_model_evaluation.json
│   ├── modeling_report.txt
│   ├── metrics.json
│   └── turkish_dataset_comparison_report.txt
│
├── 📚 references/             <- Documentación externa
│   ├── Diccionario_Variables_Musica_Turca.xlsx
│   ├── Referencias_APA.xlsx
│   ├── Team24_Machine Learning Canvas v1.0.pdf
│   ├── Fase 1_Equipo24.pdf
│   ├── Fase 2_Equipo24.pdf
│   └── Fase 01 MNA MLOps Team 24 Octubre 2025.mp4
│
├── 🛠️ scripts/               <- Scripts organizados por función
│   ├── analysis/              <- Scripts de análisis
│   │   ├── __init__.py
│   │   ├── analyze_outliers.py
│   │   ├── compare_scalers.py
│   │   ├── run_full_analysis.py
│   │   └── README.md
│   ├── training/              <- Scripts de entrenamiento
│   │   ├── train_baseline.py
│   │   ├── run_mlflow_experiments.py     ⭐ Experimentos MLflow
│   │   └── run_mlflow_experiments.py.backup
│   ├── validation/            <- Scripts de validación
│   │   ├── __init__.py
│   │   ├── verify_sync.py     ⭐ Verificación DVC+Git+S3
│   │   └── README.md
│   ├── pipelines/             
│   │   └── ml_pipeline.py
│   ├── temp/                  <- Scripts temporales (gitignored)
│   │   ├── cleanup_*.py
│   │   ├── fix_*.py
│   │   ├── test_*.py
│   │   └── update_*.py
│   └── validate_final.py
│
├── 🧪 tests/                  <- Test suite
│   ├── test_dataset_equivalence.py
│   ├── test_full_integration.py      ⭐ Integration tests
│   ├── test_ml_pipeline.py
│   ├── test_sklearn_pipeline.py
│   ├── validate_cookiecutter.py
│   ├── validate_dataset.py
│   ├── validate_features.py
│   ├── validate_plots.py
│   └── README.md
│
├── 📚 docs/                   <- Documentación
│   ├── setup_guide.md
│   ├── ml_pipeline.md
│   ├── api_endpoints.md
│   ├── deployment_guide.md
│   └── references.md
│
├── 🗄️ mlartifacts/           <- MLflow local artifacts
├── 🗄️ dvcstore/              <- DVC local cache
└── 📦 acoustic_ml.egg-info/  <- Package metadata

```

### 📊 Resumen de Directorios

| Directorio | Propósito | DVC Tracked | Git Tracked |
|-----------|-----------|-------------|-------------|
| `acoustic_ml/` | Módulo Python principal | ❌ | ✅ |
| `app/` | API REST FastAPI | ❌ | ✅ |
| `data/` | Datasets (raw, processed) | ✅ | ⚠️ (.dvc only) |
| `models/` | Modelos serializados | ✅ | ⚠️ (.dvc only) |
| `notebooks/` | Jupyter notebooks | ❌ | ✅ |
| `scripts/` | Scripts auxiliares | ❌ | ✅ |
| `tests/` | Test suite | ❌ | ✅ |
| `reports/` | Reportes y figuras | ❌ | ✅ |
| `monitoring/` | Dashboard y validación | ❌ | ✅ |
| `mlflow_artifacts/` | Experimentos MLflow | ❌ | ✅ |
| `mlartifacts/` | MLflow local store | ❌ | ❌ |
| `dvcstore/` | DVC local cache | ❌ | ❌ |

---

## 🏗️ Arquitectura del Código

### Módulo Principal: `acoustic_ml`

Módulo Python profesional con arquitectura limpia y bien documentada:

#### **1. `config.py`** - Configuración Global
```python
# Paths, constants, logging configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
```

#### **2. `dataset.py`** - Gestión de Datos
- **`DatasetManager`**: Singleton thread-safe para carga de datos
- **Funciones**: `load_dataset()`, `validate_dataset()`, `get_data_splits()`
- **Testing**: 16+ tests de validación
- **Features**: Caching, validación automática, metadata tracking

#### **3. `features.py`** - Feature Engineering
- **`FeaturePipeline`**: Pipeline de transformación completo
- **Transformers**: `OutlierHandler`, `FeatureScaler`, `FeatureSelector`
- **Análisis**: Detección de outliers, scaling robusto
- **Testing**: 13+ tests comprehensivos

#### **4. `plots.py`** - Visualizaciones
- Confusion matrices, ROC curves, distribution plots
- Outlier analysis visualizations
- Feature importance plots
- **Testing**: 8+ tests de generación de plots

#### **5. `modeling/`** - Submódulo de Modelado

```
modeling/
├── train.py           <- Lógica de entrenamiento
├── predict.py         <- Inferencia y predicciones
├── evaluate.py        <- Métricas y evaluación
├── pipeline.py        <- Pipeline MLOps completo
└── sklearn_pipeline.py <- Pipeline sklearn production-ready ⭐
```

**Pipeline Sklearn (Production-Ready)**:
```python
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline

# Pipeline completo: preprocessing + modelo
pipeline = create_sklearn_pipeline(model_type='random_forest')

# Compatible con GridSearchCV, cross_val_score
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Design Patterns Implementados

1. **Singleton Pattern**: `DatasetManager` para gestión centralizada
2. **Factory Pattern**: Creación de pipelines y modelos
3. **Strategy Pattern**: Diferentes algoritmos de scaling/preprocessing
4. **Pipeline Pattern**: Composición de transformadores sklearn

### Métricas de Calidad

- ✅ **Modularidad**: Código organizado en módulos especializados
- ✅ **Testing**: Suite comprehensiva de tests
- ✅ **Documentación**: Docstrings completos en todo el código
- ✅ **Type Hints**: Tipado estático en funciones críticas
- ✅ **SOLID Principles**: Arquitectura limpia y extensible
- ✅ **Production-Ready**: Pipeline sklearn compatible con MLflow

---

## 🎯 Modelo y Resultados

### Modelo Actual en Producción

- **Algoritmo**: Random Forest Optimizado
- **Accuracy**: **80.17%**
- **Location**: `models/optimized/production_model.pkl`
- **Dataset**: v2_cleaned_full.csv (408 filas)
- **Features**: 50+ características acústicas

### Experimentos MLflow

Se ejecutaron **7 experimentos** documentados en `mlflow_artifacts/`:

| Experimento | Modelo | Accuracy | F1-Score |
|------------|--------|----------|----------|
| exp_01 | Random Forest (Current Best) | 80.17% | 0.80 |
| exp_02 | Random Forest (Deep) | 78.5% | 0.78 |
| exp_03 | Random Forest (Simple) | 76.2% | 0.76 |
| exp_04 | Gradient Boosting | 77.8% | 0.77 |
| exp_05 | Gradient Boosting (Conservative) | 75.9% | 0.75 |
| exp_06 | Logistic Regression | 72.3% | 0.71 |
| exp_07 | SVM RBF | 74.1% | 0.73 |

**Resumen**: `mlflow_artifacts/experiments_summary.csv`

### Features Clave

Las 50+ características acústicas incluyen:

- **MFCC** (Mel-Frequency Cepstral Coefficients): 1-13 con mean/std
- **Spectral Features**: Centroid, Rolloff, Bandwidth, Contrast
- **Temporal Features**: Zero Crossing Rate, Tempo
- **Energy Features**: RMS Energy, Low Energy
- **Statistical**: Mean, Std, Min, Max por feature

### Pipeline de Datos

```
Raw Audio → Feature Extraction → Cleaning → Transformation → Model Training
```

1. **Raw Data**: Archivos CSV con características pre-extraídas
2. **Cleaning**: Eliminación de duplicados, manejo de missing values
3. **Feature Engineering**: Scaling, selection, outlier handling
4. **Model Training**: Random Forest con hyperparameter tuning
5. **Evaluation**: Cross-validation, confusion matrix, classification report

---

## 🚀 MLOps Infrastructure

### DVC (Data Version Control)

**Configuración**:
```yaml
# .dvc/config
remote:
  mlops24-s3:
    url: s3://mlops24-haowei-bucket/dvcstore
```

**Archivos Trackeados**:
- `data.dvc` → Carpeta `data/` completa
- `models/baseline.dvc` → Modelos baseline
- `models/optimized.dvc` → Modelos optimizados

**Comandos Clave**:
```bash
dvc pull              # Descargar datos desde S3
dvc push              # Subir datos a S3
dvc status            # Ver cambios pendientes
dvc repro             # Reproducir pipeline
```

### MLflow (Experiment Tracking)

**Configuración Docker**:
```yaml
# docker-compose.yml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5001:5000"
    volumes:
      - ./mlartifacts:/mlflow/mlartifacts
```

**Uso**:
```bash
docker-compose up -d    # Iniciar MLflow
# Acceder: http://localhost:5001
```

**Tracking**:
- 7 experimentos registrados
- Métricas: accuracy, f1-score, precision, recall
- Artifacts: modelos, confusion matrices, classification reports

### AWS S3 (Cloud Storage)

**Bucket**: `mlops24-haowei-bucket`

**Estructura en S3**:
```
s3://mlops24-haowei-bucket/
├── dvcstore/
│   ├── files/md5/...
│   ├── data/
│   └── models/
```

**Sincronización**:
```bash
# Verificar sync
make verify-sync

# O manualmente
python scripts/validation/verify_sync.py
```

### Cookiecutter Compliance

**Dashboard de Validación**: [https://mlopsteam24-cookiecutter2.streamlit.app](https://mlopsteam24-cookiecutter2.streamlit.app)

**Cumplimiento**: **95.2%**

**Validación Local**:
```bash
cd monitoring/dashboard
streamlit run streamlit_dashboard.py
```

---

## 💻 Instalación y Configuración

### Requisitos Previos

- **Python**: 3.12+
- **Git**: Latest version
- **DVC**: Latest version
- **AWS CLI**: Configurado con credenciales
- **Docker** (opcional): Para MLflow

### Instalación Paso a Paso

#### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd MLOps_Team24
```

#### 2. Crear Entorno Virtual

**Opción A: conda**
```bash
conda create -n acoustic_ml python=3.12
conda activate acoustic_ml
```

**Opción B: venv**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

#### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### 4. Instalar Módulo en Modo Desarrollo

```bash
pip install -e .
```

Esto instala `acoustic_ml` como módulo editable.

#### 5. Configurar AWS Credentials

```bash
aws configure
# Ingresar:
# AWS Access Key ID
# AWS Secret Access Key
# Default region: us-east-1
```

#### 6. Configurar DVC Remote

```bash
dvc remote add -d mlops24-s3 s3://mlops24-haowei-bucket/dvcstore
dvc remote modify mlops24-s3 region us-east-1
```

#### 7. Descargar Datos

```bash
dvc pull
```

Esto descarga:
- `data/` completo desde S3
- `models/` baseline y optimized

#### 8. Verificar Instalación

```bash
# Test imports
python -c "import acoustic_ml; print(acoustic_ml.__version__)"

# Verificar sync
make verify-sync
```

#### 9. (Opcional) Iniciar MLflow

```bash
docker-compose up -d
# Acceder: http://localhost:5001
```

---

## 📖 Uso del Sistema

### 🎵 Opción 1: Usar la Aplicación Web (Recomendado)

La forma más rápida de probar el sistema es usando nuestra **app de Streamlit desplegada**:

**🌐 URL**: [tu-url-de-streamlit].streamlit.app

**Funcionalidades**:
- 🎼 Análisis de emociones en tiempo real
- 📊 Visualizaciones interactivas (waveform, spectrogram)
- 📁 Subir tus propios archivos de audio (.mp3, .wav)
- 🎯 Predicción con modelo Random Forest (76.9% accuracy)
- 📈 Feature importance analysis
- 🔄 Batch analysis de múltiples canciones

Ver la sección [🎵 Streamlit App - Production Demo](#-streamlit-app---production-demo) para más detalles.

---

### 🖥️ Opción 2: Uso Local del Módulo Python

#### 1. Cargar Datos

```python
from acoustic_ml.dataset import load_dataset

# Cargar dataset principal (408 filas)
df = load_dataset('v2_cleaned_full')
print(f"Dataset shape: {df.shape}")

# O cargar con splits predefinidos
X_train, X_test, y_train, y_test = load_dataset('v2_cleaned_full', return_splits=True)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Ver las clases disponibles
print(f"Emotions: {y_train.unique()}")  # ['Happy', 'Sad', 'Angry', 'Relax']
```

#### 2. Feature Engineering

```python
from acoustic_ml.features import FeaturePipeline

# Crear pipeline de transformación
pipeline = FeaturePipeline()

# Fit y transform sobre datos de entrenamiento
X_transformed = pipeline.fit_transform(X_train, y_train)

# Transform datos de test (sin fit)
X_test_transformed = pipeline.transform(X_test)

print(f"Features originales: {X_train.shape[1]}")
print(f"Features transformados: {X_transformed.shape[1]}")
```

#### 3. Entrenar Modelo desde Cero

```python
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
from acoustic_ml.dataset import load_dataset
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar datos
X_train, X_test, y_train, y_test = load_dataset('v2_cleaned_full', return_splits=True)

# 2. Crear pipeline completo (preprocessing + modelo)
model_pipeline = create_sklearn_pipeline(model_type='random_forest')

# 3. Entrenar
print("Entrenando modelo...")
model_pipeline.fit(X_train, y_train)

# 4. Predecir
predictions = model_pipeline.predict(X_test)
probabilities = model_pipeline.predict_proba(X_test)

# 5. Evaluar
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 6. Guardar modelo
import joblib
joblib.dump(model_pipeline, 'models/my_model.pkl')
print("\n💾 Modelo guardado en: models/my_model.pkl")
```

#### 4. Hacer Predicciones con Modelo Pre-entrenado

```python
import joblib
import pandas as pd
from acoustic_ml.dataset import load_dataset

# 1. Cargar modelo pre-entrenado
model = joblib.load('models/optimized/production_model.pkl')
print("✅ Modelo cargado (Accuracy: 80.17%)")

# 2. Cargar datos nuevos
X_test, _, _, y_test = load_dataset('v2_cleaned_full', return_splits=True)

# 3. Hacer predicciones
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 4. Ver resultados
for i in range(5):  # Primeras 5 predicciones
    true_label = y_test.iloc[i]
    pred_label = predictions[i]
    confidence = probabilities[i].max()
    
    print(f"\nCanción {i+1}:")
    print(f"  Real: {true_label}")
    print(f"  Predicción: {pred_label} (confianza: {confidence:.2%})")
    print(f"  ✅ Correcto" if true_label == pred_label else "  ❌ Incorrecto")
```

#### 5. Predicción de una Sola Canción

```python
import joblib
import numpy as np

# Cargar modelo
model = joblib.load('models/optimized/production_model.pkl')

# Features de una nueva canción (50+ características acústicas)
new_song_features = np.array([
    [0.123, -0.456, 0.789, ...]  # MFCC, spectral features, etc.
])

# Predecir emoción
emotion = model.predict(new_song_features)[0]
confidence = model.predict_proba(new_song_features)[0]

print(f"Emoción detectada: {emotion}")
print(f"Confianzas: Happy={confidence[0]:.2%}, Sad={confidence[1]:.2%}, "
      f"Angry={confidence[2]:.2%}, Relax={confidence[3]:.2%}")
```

#### 6. Visualizaciones

```python
from acoustic_ml.plots import plot_confusion_matrix, plot_feature_importance
import matplotlib.pyplot as plt

# Confusion matrix
fig = plot_confusion_matrix(
    y_test, 
    predictions, 
    save_path='reports/figures/my_confusion_matrix.png'
)
plt.show()

# Feature importance (requiere modelo con feature_importances_)
feature_names = X_train.columns.tolist()
plot_feature_importance(
    model.named_steps['classifier'],  # Extraer clasificador del pipeline
    feature_names, 
    top_n=20,
    save_path='reports/figures/feature_importance.png'
)
plt.show()
```

#### 7. Batch Prediction (Múltiples Canciones)

```python
import joblib
import pandas as pd
from pathlib import Path

# Cargar modelo
model = joblib.load('models/optimized/production_model.pkl')

# Cargar dataset con canciones nuevas
songs_df = pd.read_csv('data/processed/turkish_music_emotion_v2_cleaned_full.csv')

# Separar features y target
X = songs_df.drop('Class', axis=1)
y_true = songs_df['Class']

# Batch prediction
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Crear DataFrame con resultados
results_df = pd.DataFrame({
    'Song_ID': range(len(predictions)),
    'True_Emotion': y_true,
    'Predicted_Emotion': predictions,
    'Confidence': probabilities.max(axis=1),
    'Correct': predictions == y_true.values
})

# Guardar resultados
results_df.to_csv('reports/batch_predictions.csv', index=False)
print(f"\n✅ Predicciones guardadas en: reports/batch_predictions.csv")
print(f"\nAccuracy general: {results_df['Correct'].mean():.2%}")
print(f"Total canciones: {len(results_df)}")
print(f"Correctas: {results_df['Correct'].sum()}")
print(f"Incorrectas: {(~results_df['Correct']).sum()}")
```

---

### 🚀 Scripts Rápidos (Línea de Comando)

#### Entrenar Modelo Baseline

```bash
# Entrenar Random Forest baseline
python scripts/training/train_baseline.py

# Output: models/baseline/random_forest_baseline.pkl
```

#### Ejecutar Todos los Experimentos MLflow

```bash
# Ejecuta 7 experimentos con diferentes modelos
python scripts/training/run_mlflow_experiments.py

# Ver resultados en: http://localhost:5001 (MLflow UI)
```

#### Análisis Exploratorio

```bash
# Análisis de outliers
python scripts/analysis/analyze_outliers.py

# Comparación de scalers (StandardScaler vs RobustScaler)
python scripts/analysis/compare_scalers.py

# Análisis completo
python scripts/analysis/run_full_analysis.py
```

#### Validación y Testing

```bash
# Validación completa del sistema
python tests/test_full_integration.py

# Tests específicos
python tests/test_sklearn_pipeline.py
python tests/test_dataset_equivalence.py
```

---

### 📊 Workflow Completo: De Cero a Producción

```bash
# 1. Setup inicial
conda activate acoustic_ml
dvc pull  # Descargar datos

# 2. Exploración (opcional)
jupyter notebook notebooks/1.0-team-eda-turkish-music.ipynb

# 3. Entrenar modelo
python scripts/training/train_baseline.py

# 4. Experimentación con MLflow
docker-compose up -d  # Iniciar MLflow UI
python scripts/training/run_mlflow_experiments.py

# 5. Evaluar mejor modelo
python -c "
from acoustic_ml.dataset import load_dataset
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = load_dataset('v2_cleaned_full', return_splits=True)
model = create_sklearn_pipeline('random_forest')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
"

# 6. Guardar modelo final
python -c "
import joblib
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline
model = create_sklearn_pipeline('random_forest')
# ... entrenar ...
joblib.dump(model, 'models/optimized/production_model.pkl')
print('✅ Modelo guardado')
"

# 7. Deploy (Streamlit app o API)
# Ver sección de Streamlit App
```

---

## 🛠️ Scripts Disponibles

### Makefile Commands

El proyecto incluye un `Makefile` con comandos útiles:

```bash
make data           # Descarga datos con DVC
make train          # Entrena modelo baseline
make reproduce      # Reproduce pipeline DVC completo
make clean          # Limpia archivos temporales
make verify-sync    # Verifica sincronización DVC+Git+S3
make freeze         # Actualiza requirements.txt
make test           # Ejecuta tests
make mlflow         # Inicia MLflow UI
make help           # Muestra todos los comandos
```

### Scripts de Training

```bash
# Entrenamiento baseline
python scripts/training/train_baseline.py

# Experimentos MLflow (7 modelos)
python scripts/training/run_mlflow_experiments.py
```

### Scripts de Análisis

```bash
# Análisis de outliers
python scripts/analysis/analyze_outliers.py

# Comparación de scalers
python scripts/analysis/compare_scalers.py

# Análisis completo
python scripts/analysis/run_full_analysis.py
```

### Scripts de Validación

```bash
# Verificar sincronización DVC+Git+S3
python scripts/validation/verify_sync.py

# Validación Cookiecutter
python tests/validate_cookiecutter.py

# Tests de integración
python tests/test_full_integration.py
```

---

## 🧪 Testing & Quality Assurance

### Ejecutar Tests

```bash
# Ejecutar todos los tests con output detallado
pytest tests/ -v

# Modo quiet (resumen)
pytest tests/ -q

# Tests específicos con traceback corto
pytest tests/ -v --tb=short

# Con cobertura
pytest tests/ --cov=acoustic_ml
```

### Suite de 33 Tests

**Ubicación**: `tests/` (4 módulos principales)

| Módulo | Tipo | Cantidad | Propósito |
|--------|------|----------|----------|
| `test_dataset_equivalence.py` | Unitario | 8 tests | Validar DatasetManager, cargas, transformaciones |
| `test_sklearn_pipeline.py` | Unitario | 7 tests | Pipeline sklearn, features, scalers |
| `test_full_integration.py` | Integración | 12 tests | End-to-end: data → model → predict |
| `test_api_endpoints.py` | API | 6 tests | FastAPI endpoints (TestClient, no servidor) |

### Tipos de Tests

**Unitarios (15 tests)**:
```bash
pytest tests/test_dataset_equivalence.py -v  # DatasetManager, data loading
pytest tests/test_sklearn_pipeline.py -v     # Feature engineering, pipeline creation
```

**Integración (12 tests)**:
```bash
pytest tests/test_full_integration.py -v     # Full pipeline: train → predict
```

**API (6 tests)**:
```bash
pytest tests/test_api_endpoints.py -v        # /health, /predict, /train, /models
```

### Resultado Esperado

```
tests/test_dataset_equivalence.py::test_load_data PASSED
tests/test_dataset_equivalence.py::test_dataset_manager PASSED
tests/test_sklearn_pipeline.py::test_create_pipeline PASSED
tests/test_sklearn_pipeline.py::test_feature_transform PASSED
tests/test_full_integration.py::test_train_predict_pipeline PASSED
tests/test_api_endpoints.py::test_health_check PASSED
tests/test_api_endpoints.py::test_predict_endpoint PASSED

========================== 33 passed in 2.45s ==========================
```

### Validación Rápida Post-Cambios

```bash
# Quick test después de editar código
make test

# O directamente
pytest tests/ -q
```

---

## 🌐 API Serving with FastAPI

### Endpoints Disponibles

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Root endpoint - API status |
| `GET` | `/api/v1/health` | Health check del sistema |
| `POST` | `/api/v1/predict` | Predicción single de emoción |
| `POST` | `/api/v1/train` | Trigger retraining del modelo |
| `GET` | `/api/v1/models` | Listar modelos disponibles |

### Iniciar Localmente

```bash
# Opción 1: Uvicorn directo
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Opción 2: Desde app/main.py
python app/main.py

# Opción 3: Con gunicorn (producción)
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

**Acceso a documentación automática**:
```
Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
OpenAPI:     http://localhost:8000/openapi.json
```

### Ejemplo: POST /api/v1/predict

**Request JSON**:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "chroma_stft": 0.45,
    "chroma_stft_std": 0.32,
    "mfcc_1": 12.5,
    "mfcc_1_std": 8.3,
    "mfcc_2": -5.2,
    "mfcc_2_std": 3.1,
    "mfcc_3": 2.1,
    "mfcc_3_std": 1.8,
    "mfcc_4": 0.9,
    "mfcc_4_std": 0.7,
    "mfcc_5": -1.2,
    "mfcc_5_std": 0.5,
    "zero_crossing_rate": 0.12,
    "zero_crossing_rate_std": 0.08
  }'
```

**Response JSON**:
```json
{
  "emotion": "Happy",
  "confidence": 0.87,
  "probabilities": {
    "Happy": 0.87,
    "Angry": 0.08,
    "Sad": 0.03,
    "Relax": 0.02
  },
  "model_version": "production_model_v2",
  "timestamp": "2024-11-12T10:30:45.123Z"
}
```

### Health Check: GET /api/v1/health

```bash
curl http://localhost:8000/api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "s3_connection": true,
  "mlflow_connection": true,
  "timestamp": "2024-11-12T10:30:00Z"
}
```

### Ubicación del Código

```
app/
├── main.py           <- Entry point, lifespan, docs
├── api/
│   ├── __init__.py
│   ├── main.py       <- APIRouter con todos los endpoints
│   ├── endpoints.py  <- Funciones de cada endpoint
│   └── schemas.py    <- Pydantic models (request/response)
├── core/
│   ├── config.py     <- Configuración API (host, port, etc)
│   └── logger.py     <- Setup logging
└── services/
    └── model_service.py <- Lógica de predicción y modelo
```

### Schemas Pydantic (Validación Automática)

- `PredictionRequest`: 50+ features acústicos (validados)
- `PredictionResponse`: Estructura estandardizada respuesta
- `HealthResponse`: Status checks sistema

```bash
# Ver schemas JSON schema
curl http://localhost:8000/openapi.json | grep -A 50 "components"
```

---

## 🔍 Data Drift Detection & Monitoring

### Ejecutar Drift Detection

```bash
# Modo normal
python -m drift.run_drift

# Con parámetros
python -m drift.run_drift --threshold 0.05 --output reports/drift/

# Modo test (con synthetic drift data)
python -m drift.run_drift --test-mode
```

### Qué Detecta

**Statistical Drift** (Evidently):
- Cambios en distribución de features acústicos
- Comparación train data vs inference data
- KL divergence > 0.3 = alerta

**Performance Degradation**:
- Caída en accuracy > 5% = ⚠️ warning
- Degradación por clase (precision/recall)
- Matriz de confusión comparativa

### Thresholds y Alertas

| Métrica | Threshold | Acción |
|---------|-----------|--------|
| Accuracy Drop | > 5% | ⚠️ Warning - Review model |
| Feature Shift | KL divergence > 0.3 | 🔴 Alert - Check data source |
| Class Imbalance | Ratio > 10:1 | 🔴 Critical - Retrain required |

### Output: drift_report.json

**Ubicación**: `reports/drift/drift_report.json`

```json
{
  "timestamp": "2024-11-12T10:30:00Z",
  "drift_detected": false,
  "accuracy_drop_percent": -1.11,
  "features_shifted": 3,
  "critical_features": [
    "mfcc_1",
    "chroma_stft",
    "zero_crossing_rate"
  ],
  "recommendation": "Monitor - No action required",
  "performance_metrics": {
    "train_accuracy": 0.8017,
    "inference_accuracy": 0.7906,
    "diff": -0.0111,
    "train_precision": 0.78,
    "inference_precision": 0.77
  }
}
```

### Demo: Drift Real

**Sin drift**:
```
✅ Accuracy: 80.17% → 79.06% (diff: -1.11%)
✅ Status: HEALTHY
```

**Con synthetic drift** (`generate_drift_data.py`):
```
🔴 Accuracy: 80.17% → 17.28% (diff: -62.89%)
🔴 Status: CRITICAL - Retrain required
```

### Archivos y Scripts

```
drift/
├── __init__.py
├── run_drift.py         <- Main execution script
├── drift_detector.py    <- Statistical analysis (Evidently)
└── comparators.py       <- Feature comparators

scripts/data/
└── generate_drift_data.py  <- Genera synthetic drift data para testing
```

### Generar y Probar Drift

```bash
# Generar synthetic drift data
python scripts/data/generate_drift_data.py

# Ejecutar drift detection
python -m drift.run_drift --test-mode

# Ver reporte
cat reports/drift/drift_report.json | jq .
```

---

## 🐳 Docker & Containerization

### Build Imagen

```bash
# Build imagen básica
docker build -t mlops-team24:latest .

# Build con tag de versión
docker build -t mlops-team24:v1.0 -t mlops-team24:latest .

# Verificar imagen
docker images | grep mlops-team24
```

### Run Local (Standalone FastAPI)

```bash
# Run simple
docker run -p 8000:8000 mlops-team24:latest

# Con mount de código (desarrollo)
docker run -it -p 8000:8000 \
  -v $(pwd):/app \
  mlops-team24:latest /bin/bash

# Con variables de entorno
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=$AWS_KEY \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET \
  mlops-team24:latest
```

**Verificar que funciona**:
```bash
curl http://localhost:8000/api/v1/health
```

### Docker Compose Stack

**Archivo**: `docker-compose.yml`

```bash
# Iniciar todo el stack
docker compose up

# Modo detached (background)
docker compose up -d

# Ver logs
docker compose logs -f api

# Ver status
docker compose ps

# Detener todo
docker compose down

# Limpiar volúmenes (reset total)
docker compose down -v
```

**Services que se levantan**:

| Service | Puerto | Descripción |
|---------|--------|-------------|
| `api` | `8000` | FastAPI application (uvicorn) |
| `mlflow` | `5001` | MLflow tracking server |
| `minio` | `9000` | S3-compatible storage (opcional) |

**Acceso**:
```
FastAPI Docs:  http://127.0.0.1:8000/docs
MLflow UI:     http://127.0.0.1:5001
MinIO:         http://127.0.0.1:9000
```

### Configuración: config.env

**⚠️ IMPORTANTE**: `config.env` NO está versionado (`.gitignore`)

```bash
# 1. Copiar template
cp config.env.example config.env

# 2. Llenar con credenciales reales
cat config.env
```

**Contenido de config.env**:
```env
# AWS S3
AWS_ACCESS_KEY_ID=tu_access_key_aqui
AWS_SECRET_ACCESS_KEY=tu_secret_key_aqui
AWS_REGION=us-east-1
AWS_S3_BUCKET=mlops24-haowei-bucket

# MLflow
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlops24-haowei-bucket/mlflow

# DVC
DVC_REMOTE_URL=s3://mlops24-haowei-bucket/dvc-storage
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements-prod.txt .

# Instalar Python deps
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=5)"

# Comando por defecto
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Backend Storage

**Desarrollo** (docker-compose):
- SQLite: `mlflow.db` (local)
- Artifacts: Directorio `mlflow_artifacts/`

**Producción**:
- Backend: PostgreSQL o RDS
- Artifacts: AWS S3 (`mlops24-haowei-bucket`)

---

## 🚀 Publicación en DockerHub

### ¿Qué es DockerHub y por qué se publica?

**DockerHub** es el registro central de imágenes Docker. Al publicar nuestra imagen aquí, permite que:

- ✅ Cualquier persona descargue y ejecute el modelo sin necesidad de compilar
- ✅ El equipo acceda a versiones consistentes desde cualquier máquina
- ✅ CI/CD pipelines descarguen automáticamente la versión correcta
- ✅ Usuarios ejecuten el servicio con un simple comando: `docker run javirebull/ml-service:v3.0.0`
- ✅ Se cree un historial de versiones documentadas

**Imagen**: `javirebull/ml-service:v3.0.0`

### 1. Autenticación en DockerHub

Primero, loguéate en DockerHub desde tu máquina:

```bash
# Login interactivo
docker login

# Te pedirá:
# Username: javirebull
# Password: (tu token o contraseña de DockerHub)
```

Para mayor seguridad, es recomendable usar **Personal Access Tokens** en lugar de contraseña:

```bash
# En DockerHub: Account Settings → Security → New Access Token
# Copia el token y úsalo aquí
docker login --username javirebull --password-stdin <<< "tu_personal_access_token"
```

**Verificar que estás loguado**:
```bash
cat ~/.docker/config.json  # Verá la autenticación almacenada
```

### 2. Etiquetar la Imagen (Tag)

Antes de publicar, debes etiquetar la imagen local con el nombre correcto:

```bash
# Primero, construir la imagen (si no existe)
docker build -t mlops-team24:latest .

# Etiquetar para DockerHub
# Formato: docker tag <imagen_local> <usuario_dockerhub>/<nombre_repo>:<version>
docker tag mlops-team24:latest javirebull/ml-service:v3.0.0

# También etiquetar como latest
docker tag mlops-team24:latest javirebull/ml-service:latest

# Verificar tags
docker images | grep ml-service
```

**Esperado**:
```
REPOSITORY                 TAG       IMAGE ID       CREATED        SIZE
javirebull/ml-service      v3.0.0    abc123def456   5 minutes ago  1.2GB
javirebull/ml-service      latest    abc123def456   5 minutes ago  1.2GB
mlops-team24               latest    abc123def456   5 minutes ago  1.2GB
```

### 3. Publicar la Imagen (Push)

Ahora sube la imagen a DockerHub:

```bash
# Push versión específica
docker push javirebull/ml-service:v3.0.0

# Push tag latest
docker push javirebull/ml-service:latest

# Verificar progreso
# Verás: Pushing [==================================================>]
```

**Durante el push, verás algo como**:
```
The push refers to repository [docker.io/javirebull/ml-service]
a1b2c3d4e5f6: Pushed
7g8h9i0j1k2l: Pushed
v3.0.0: digest: sha256:abc123... size: 4587
latest: digest: sha256:abc123... size: 4587
```

### 4. Verificar en DockerHub

Después del push, verifica que la imagen está disponible:

```bash
# Opción 1: Desde línea de comandos
docker pull javirebull/ml-service:v3.0.0

# Debería decir: Status: Downloaded newer image

# Opción 2: En navegador (recomendado)
# Visita: https://hub.docker.com/r/javirebull/ml-service
# Verás:
# - Tags disponibles: v3.0.0, latest
# - Size de la imagen
# - Pull count
# - Fecha de última actualización
```

### 5. Descargar y Ejecutar desde Cualquier Máquina

Una vez publicada, cualquiera puede ejecutar el modelo con un solo comando:

```bash
# Descargar y ejecutar la imagen v3.0.0
docker run -p 8000:8000 javirebull/ml-service:v3.0.0

# Con variables de entorno (si es necesario)
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=$AWS_KEY \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET \
  javirebull/ml-service:v3.0.0

# Con datos persistentes
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  javirebull/ml-service:v3.0.0

# Con nombre personalizado
docker run -p 8000:8000 \
  --name ml-service-prod \
  javirebull/ml-service:v3.0.0
```

**Verificar que está funcionando**:
```bash
# En otra terminal
curl http://localhost:8000/api/v1/health

# Respuesta esperada
# {"status": "healthy", "timestamp": "2024-11-12T..."}
```

### 6. Workflow Completo (Copy-Paste Ready)

Para publicar una nueva versión, sigue este flujo completo:

```bash
# ========== PASO 1: Build ==========
# Asegúrate de que los cambios están en main
git status  # working tree clean

# Construir imagen
docker build -t mlops-team24:latest .

# ========== PASO 2: Tag ==========
# Etiquetar para DockerHub
docker tag mlops-team24:latest javirebull/ml-service:v3.0.0
docker tag mlops-team24:latest javirebull/ml-service:latest

# Verificar
docker images | grep ml-service

# ========== PASO 3: Login ==========
# Loguete en DockerHub (si no está loguado)
docker login

# ========== PASO 4: Push ==========
# Publicar ambas tags
docker push javirebull/ml-service:v3.0.0
docker push javirebull/ml-service:latest

# Esperar a que termine (puede tomar 2-5 minutos)

# ========== PASO 5: Verificar ==========
# Test: Descargar imagen fresca
docker rmi javirebull/ml-service:v3.0.0  # Elimina local
docker pull javirebull/ml-service:v3.0.0  # Descarga desde Hub

# Ejecutar test
docker run -p 8000:8000 javirebull/ml-service:v3.0.0 &
sleep 3
curl http://localhost:8000/api/v1/health
```

### 7. Notas sobre Autenticación y Usuario

⚠️ **IMPORTANTE - Usuario Correcto**:

| Campo | Valor |
|-------|-------|
| **Docker Hub Username** | `javirebull` |
| **Repository** | `javirebull/ml-service` |
| **Full Image** | `javirebull/ml-service:v3.0.0` |
| **Access** | Público (cualquiera puede descargar) |

**Verificar credenciales**:
```bash
# Ver usuario actualmente loguado
cat ~/.docker/config.json | grep -A2 "docker.io"

# Si cambio de usuario, hacer logout
docker logout

# Luego login con el usuario correcto
docker login --username javirebull
```

**Si tienes errores de autenticación**:
```bash
# Error: "denied: requested access to the resource is denied"
# Solución:
docker logout
docker login --username javirebull  # Con espacios, no flags
# Ingresa token/contraseña cuando se pida
```

### Versioning Strategy

Recomendación de tags para futuras versiones:

```bash
# Versión patch (bug fixes)
docker tag mlops-team24:latest javirebull/ml-service:v3.0.1

# Versión minor (nuevas features)
docker tag mlops-team24:latest javirebull/ml-service:v3.1.0

# Versión major (cambios significativos)
docker tag mlops-team24:latest javirebull/ml-service:v4.0.0

# Siempre actualizar latest
docker tag mlops-team24:latest javirebull/ml-service:latest

# Push todas
docker push javirebull/ml-service:v3.0.1
docker push javirebull/ml-service:v3.1.0
docker push javirebull/ml-service:v4.0.0
docker push javirebull/ml-service:latest
```

---

## 🔄 Reproducibility & Seeds

### Seeds Configurados

**Archivo**: `acoustic_ml/__init__.py` y `acoustic_ml/config.py`

```python
import numpy as np
from sklearn.utils import check_random_state

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# En sklearn pipelines
RandomForestClassifier(random_state=RANDOM_SEED)
train_test_split(X, y, random_state=RANDOM_SEED)
```

**Todos los modelos usan**:
- `random_seed=42`
- `numpy seed=42`
- `sklearn seed=42`
- `pytorch seed=42` (si aplica)

### Requirements Fijado

**Archivo**: `requirements-prod.txt` (pip freeze)

```
scikit-learn==1.3.2
pandas==2.1.3
numpy==1.24.3
mlflow==2.10.0
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.4.2
```

**Generar nuevo freeze**:
```bash
pip freeze > requirements-prod.txt
```

### DVC Data Versioning

**Versionado**:
- ✅ `data/processed/` → `data.dvc`
- ✅ `models/optimized/` → `models/optimized.dvc`
- ✅ Tracked en S3: `mlops24-haowei-bucket`

**Pull datos antes de ejecutar**:
```bash
dvc pull
```

### Reproducir Pipeline Completo

```bash
# Opción 1: Pull datos + Drift validation
dvc pull && python -m drift.run_drift

# Opción 2: Full train + predict pipeline
dvc pull && python acoustic_ml/modeling/train.py

# Opción 3: Verificar sincronización
make verify-sync && pytest tests/ -q
```

### Validar Reproducibilidad

```bash
# 1. Ejecutar pipeline en máquina A
dvc pull
python acoustic_ml/modeling/train.py
# → Genera modelo con accuracy 80.17%

# 2. Ejecutar pipeline en máquina B (mismo código)
dvc pull
python acoustic_ml/modeling/train.py
# → DEBE generar exactamente el mismo modelo con accuracy 80.17%

# 3. Verificar hashes
md5sum models/optimized/production_model.pkl
# Deben coincidir entre máquinas
```

### Checklist Reproducibilidad

- ✅ Seeds configurados (numpy, sklearn, random)
- ✅ Requirements fijado con pip freeze
- ✅ DVC data versioning activo
- ✅ Docker containerización
- ✅ 33 tests pasando
- ✅ Git history limpio (conventional commits)
- ✅ Pipeline determinístico end-to-end

---

## ✅ Phase 3 Requirements Checklist

**Todos los requisitos de Fase 3 implementados y validados**:

| Requisito | Implementación | Status |
|-----------|---|--------|
| **1. Pruebas Unitarias/Integración** | 33 tests (pytest) en `tests/` - Unitarios, Integración, API, Full Pipeline | ✅ COMPLETO |
| **2. FastAPI Serving** | 5 endpoints en `app/` - /health, /predict, /train, /models, / | ✅ COMPLETO |
| **3. Reproducibilidad** | Seeds, requirements-prod.txt, DVC, Docker - dvc pull && python -m drift.run_drift | ✅ COMPLETO |
| **4. Docker Containerización** | docker-compose.yml con FastAPI + MLflow - docker compose up | ✅ COMPLETO |
| **5. Data Drift Detection** | Evidently + statistical monitoring - python -m drift.run_drift | ✅ COMPLETO |

### Verificación Rápida

```bash
# 1. Tests
pytest tests/ -q  # ✅ 33 passed

# 2. API
uvicorn app.main:app --reload
curl http://localhost:8000/api/v1/health  # ✅ healthy

# 3. Reproducibilidad
dvc pull && python -m drift.run_drift  # ✅ consistent results

# 4. Docker
docker compose up -d  # ✅ all services running

# 5. Drift
python -m drift.run_drift --test-mode  # ✅ drift_report.json generated
```

---

## 🗂️ Project Structure

**Estructura completa orientada a Fase 3**:

```
MLOps_Team24/
│
├── 📄 Configuración (Raíz)
│   ├── README.md                    ← Este archivo
│   ├── Makefile                     ← Comandos: make test, make train, etc
│   ├── requirements-prod.txt        ← Dependencies fijadas (pip freeze)
│   ├── requirements-dev.txt         ← Dev dependencies (pytest, etc)
│   ├── pyproject.toml               ← Proyecto Python config
│   ├── params.yaml                  ← Parámetros DVC pipeline
│   ├── dvc.yaml                     ← Pipeline stages
│   ├── docker-compose.yml           ← FastAPI + MLflow + MinIO stack
│   ├── Dockerfile                   ← Container image
│   ├── config.env.example           ← Template variables (AWS, MLflow)
│   ├── .gitignore                   ← config.env, .env, datos, modelos
│   └── .dvc/                        ← DVC configuración
│
├── 📦 acoustic_ml/                  ← Módulo Python principal
│   ├── __init__.py
│   ├── config.py                    ← Global config (RANDOM_SEED=42)
│   ├── dataset.py                   ← DatasetManager (Singleton)
│   ├── features.py                  ← Feature engineering
│   ├── plots.py                     ← Visualizaciones
│   └── modeling/
│       ├── train.py                 ← Training logic
│       ├── predict.py               ← Inference
│       ├── evaluate.py              ← Metrics
│       ├── pipeline.py              ← MLOps pipeline
│       └── sklearn_pipeline.py      ← Production pipeline
│
├── 🌐 app/                          ← FastAPI Application
│   ├── main.py                      ← Entry point (uvicorn)
│   ├── api/
│   │   ├── main.py                  ← APIRouter endpoints
│   │   ├── endpoints.py             ← Endpoint functions
│   │   └── schemas.py               ← Pydantic models
│   ├── core/
│   │   ├── config.py                ← API config
│   │   └── logger.py                ← Logging setup
│   └── services/
│       └── model_service.py         ← Model predictions
│
├── 🔍 drift/                        ← Drift Detection System
│   ├── __init__.py
│   ├── run_drift.py                 ← Main drift detection
│   ├── drift_detector.py            ← Evidently analysis
│   └── comparators.py               ← Feature comparators
│
├── 📊 data/                         ← Datos (versionados DVC)
│   ├── raw/                         ← Datos originales
│   ├── interim/                     ← Transformaciones intermedias
│   ├── processed/                   ← Datos finales
│   │   ├── turkish_music_emotion_v2_cleaned_full.csv (400+ filas)
│   │   ├── X_train.csv, X_test.csv
│   │   └── y_train.csv, y_test.csv
│   ├── data.dvc                     ← DVC tracking
│   └── .gitignore                   ← Ignorar archivos grandes
│
├── 💾 models/                       ← Modelos (versionados)
│   ├── optimized/
│   │   ├── production_model.pkl     ← Modelo actual (80.17%)
│   │   └── production_model_metadata.json
│   └── optimized.dvc                ← DVC tracking
│
├── 📈 mlflow_artifacts/             ← MLflow experiments
│   ├── exp_01_Random_Forest_Current_Best/
│   ├── experiments_summary.csv
│   └── experiments_report.txt
│
├── 📓 notebooks/                    ← Jupyter notebooks
│   ├── 1.0-team-eda-turkish-music.ipynb
│   ├── 2.0-team-preprocessing.ipynb
│   ├── 3.0-team-modeling-evaluation.ipynb
│   └── archive/
│
├── 📈 reports/                      ← Análisis y reportes
│   ├── figures/                     ← Visualizaciones
│   │   ├── confusion_matrices_top3.png
│   │   ├── final_confusion_matrix.png
│   │   └── *.png
│   ├── drift/                       ← Drift reports
│   │   └── drift_report.json        ← Salida drift detection
│   └── metrics.json
│
├── 🧪 tests/                        ← Test Suite (33 tests)
│   ├── test_dataset_equivalence.py  ← Dataset tests
│   ├── test_sklearn_pipeline.py     ← Pipeline tests
│   ├── test_full_integration.py     ← Integration tests
│   ├── test_api_endpoints.py        ← API tests (TestClient)
│   ├── validate_cookiecutter.py     ← Structure validation
│   ├── validate_dataset.py
│   ├── validate_features.py
│   └── validate_plots.py
│
├── 📚 scripts/                      ← Scripts automatizados
│   ├── training/
│   │   ├── train_baseline.py
│   │   └── run_mlflow_experiments.py
│   ├── analysis/
│   │   ├── analyze_outliers.py
│   │   └── compare_scalers.py
│   ├── validation/
│   │   └── verify_sync.py
│   └── data/
│       └── generate_drift_data.py   ← Synthetic drift generation
│
├── 📊 monitoring/                   ← Monitoring & Dashboards
│   ├── dashboard/
│   │   └── streamlit_dashboard.py   ← Cookiecutter validation dashboard
│   └── README.md
│
└── 📚 references/                   ← Documentación externa
    ├── Diccionario_Variables_Musica_Turca.xlsx
    ├── Fase1_Equipo24.pdf
    ├── Fase2_Equipo24.pdf
    └── Team24_Machine_Learning_Canvas.pdf
```

**Cookiecutter Data Science Compliance**: ✅ 95.2%

**Referencia**: [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---
## 🎵 Streamlit App - Production Demo

### 🌐 Aplicación Web Desplegada

Hemos desarrollado una **aplicación web interactiva** para demostrar las capacidades del sistema de reconocimiento de emociones musicales en producción.

**🔗 URL de Acceso**: **[tu-url-de-streamlit].streamlit.app**

**📱 Compatibilidad**: Desktop, Tablet, Mobile

---

### ✨ Características Principales

#### 🎼 1. Análisis de Música en Tiempo Real

- **Predicción instantánea** de emociones en canciones turcas
- **4 emociones detectadas**: Angry 😡, Happy 😊, Relax 😌, Sad 😢
- **Confianza de predicción**: Probabilidades por clase
- **Modelo**: Random Forest (76.9% accuracy)

#### 📁 2. Upload de Archivos

- **Formatos soportados**: `.mp3`, `.wav`, `.ogg`
- **Procesamiento automático**: Extracción de features acústicas
- **Análisis inmediato**: Resultados en segundos
- **Límite de tamaño**: 200MB por archivo

#### 📊 3. Visualizaciones Interactivas

**Waveform (Forma de Onda)**:
- Visualización temporal de la señal de audio
- Amplitud vs. tiempo
- Identificación de patrones rítmicos

**Spectrogram (Espectrograma)**:
- Representación tiempo-frecuencia
- Intensidad de frecuencias a lo largo del tiempo
- Identificación de características tonales

**Feature Importance**:
- Top 20 características más relevantes
- Impacto de cada feature en la predicción
- Análisis de MFCC, spectral features, temporal features

#### 🎯 4. Predicción con Audios de Muestra

- **Biblioteca de ejemplos**: Canciones turcas pre-cargadas
- **Cada emoción representada**: 1-2 ejemplos por clase
- **Testing rápido**: Probar el modelo sin subir archivos
- **Comparación**: Ver diferentes emociones musicales

#### 🔄 5. Batch Analysis

- **Análisis múltiple**: Subir y procesar varias canciones
- **Resultados agregados**: Estadísticas del conjunto
- **Exportar CSV**: Descargar predicciones completas
- **Comparación entre canciones**: Análisis comparativo

#### 🎚️ 6. Selector de Modelos (Local)

Si ejecutas la app localmente, puedes cambiar entre modelos:
- Random Forest (default) - 76.9% accuracy
- Gradient Boosting - 77.8% accuracy
- XGBoost - experimental

---

### 🚀 Cómo Usar la App

#### Opción 1: App en la Nube (Recomendado)

1. **Acceder**: Ir a [tu-url-de-streamlit].streamlit.app
2. **Elegir modo**:
   - 📁 **Upload**: Subir tu propio audio
   - 🎵 **Samples**: Usar audios de ejemplo
3. **Analizar**: La app procesará automáticamente
4. **Ver resultados**:
   - Emoción predicha con confianza
   - Visualizaciones (waveform, spectrogram)
   - Feature importance
5. **Experimentar**: Probar con diferentes canciones

#### Opción 2: Ejecutar Localmente

```bash
# 1. Navegar al directorio de la app
cd streamlit_app/  # o donde esté tu app de Streamlit

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar app
streamlit run app.py

# 4. Abrir en navegador
# Automáticamente abre en: http://localhost:8501
```

---

### 🛠️ Tecnologías Utilizadas

**Backend**:
- **Streamlit**: Framework de la aplicación
- **scikit-learn**: Modelo de ML (Random Forest)
- **librosa**: Procesamiento de audio y feature extraction
- **pandas/numpy**: Manipulación de datos

**Visualización**:
- **matplotlib**: Gráficas estáticas
- **plotly**: Visualizaciones interactivas
- **seaborn**: Styling de gráficas

**Deployment**:
- **Streamlit Cloud**: Hosting gratuito
- **GitHub Integration**: Deploy automático desde main branch
- **Secrets Management**: Configuración segura

---

### 📁 Estructura de la App

```
streamlit_app/
├── app.py                      <- Main application file
├── requirements.txt            <- Dependencies
├── .streamlit/
│   └── config.toml             <- Streamlit configuration
├── models/
│   └── baseline_model.pkl      <- Pre-trained model
├── sample_audios/              <- Sample Turkish music files
│   ├── angry_example.mp3
│   ├── happy_example.mp3
│   ├── relax_example.mp3
│   └── sad_example.mp3
└── utils/
    ├── audio_processor.py      <- Audio feature extraction
    ├── model_loader.py         <- Model loading utilities
    └── visualizations.py       <- Plot generation functions
```

---

### 🔧 Configuración y Personalización

#### Variables de Entorno

Si ejecutas localmente, puedes configurar:

```toml
# .streamlit/secrets.toml
[model]
default_model = "random_forest"
confidence_threshold = 0.5

[audio]
max_file_size = 200  # MB
allowed_formats = [".mp3", ".wav", ".ogg"]
sample_rate = 22050

[features]
n_mfcc = 13
n_fft = 2048
hop_length = 512
```

#### Personalizar Tema

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"
```

---

### 📊 Ejemplos de Uso

#### Ejemplo 1: Análisis de Audio Subido

```
1. Usuario sube: "turkish_song.mp3"
2. App extrae 50+ features acústicas
3. Modelo predice: "Happy" (confianza: 87.3%)
4. Visualizaciones generadas:
   - Waveform: Muestra patrones rítmicos alegres
   - Spectrogram: Frecuencias altas prominentes
   - Features: MFCC_3 y Spectral_Centroid destacados
```

#### Ejemplo 2: Comparación de Emociones

```
Usuario selecciona 4 samples (uno por emoción):
- Angry:  Predicción correcta (92.1%)
- Happy:  Predicción correcta (87.3%)
- Relax:  Predicción correcta (81.5%)
- Sad:    Predicción correcta (79.8%)

Resultado: 100% accuracy en samples
```

#### Ejemplo 3: Batch Analysis

```
Usuario sube 10 canciones:
- 7 predicciones correctas
- 3 con confusión Relax ↔ Sad
- Accuracy batch: 70%
- Confianza promedio: 78.4%
```

---

### 🎯 Casos de Uso

#### 🎵 Para Músicos y Productores

- **Validar la emoción** que transmite una composición
- **Comparar versiones** de la misma canción
- **Analizar el "mood"** de un álbum completo

#### 🔍 Para Investigadores

- **Estudiar características** de música emocional turca
- **Comparar con otros datasets** musicales
- **Validar modelos** de emoción musical

#### 📚 Para Educación

- **Demostrar ML aplicado** en análisis de audio
- **Enseñar feature engineering** en música
- **Mostrar pipeline MLOps** completo

#### 🎧 Para Oyentes

- **Descubrir canciones** con emociones específicas
- **Entender por qué** una canción suena "triste" o "alegre"
- **Explorar música turca** por emoción

---

### 🐛 Troubleshooting

#### Error: "Model not found"
```bash
# Verificar que el modelo existe
ls models/baseline_model.pkl

# Re-descargar desde DVC
dvc pull models/baseline.dvc
```

#### Error: "Audio file too large"
```python
# Comprimir audio antes de subir
from pydub import AudioSegment
audio = AudioSegment.from_mp3("large_file.mp3")
audio.export("compressed.mp3", format="mp3", bitrate="128k")
```

#### Error: "Feature extraction failed"
```python
# Verificar formato de audio
import librosa
y, sr = librosa.load("audio.mp3", sr=22050)
print(f"Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
```

---

### 🚀 Roadmap de la App

**Fase Actual (Phase 2)** ✅:
- ✅ Predicción básica con modelo Random Forest
- ✅ Upload de archivos de audio
- ✅ Visualizaciones (waveform, spectrogram)
- ✅ Análisis de feature importance
- ✅ Deploy en Streamlit Cloud

**Próximas Mejoras (Phase 3)**:
- 🔄 Multi-model comparison en tiempo real
- 🔄 A/B testing entre modelos
- 🔄 Export de reportes PDF
- 🔄 Integración con API REST
- 🔄 User authentication
- 🔄 Historial de predicciones

**Futuro (Phase 4+)**:
- 💡 Recomendaciones de canciones similares
- 💡 Análisis de playlists completas
- 💡 Integración con Spotify API
- 💡 Mobile app (React Native)
- 💡 Real-time audio recording y análisis

---

### 📸 Screenshots

> **Nota**: Agregar screenshots reales de la app cuando esté desplegada:

```markdown
![Home Page](docs/images/app_home.png)
*Página principal con opciones de análisis*

![Prediction Results](docs/images/app_prediction.png)
*Resultados de predicción con visualizaciones*

![Feature Importance](docs/images/app_features.png)
*Análisis de características más relevantes*
```

---

### 🔗 Links Relacionados

- **App en Producción**: [tu-url-de-streamlit].streamlit.app
- **Dashboard Cookiecutter**: [https://mlopsteam24-cookiecutter.streamlit.app](https://mlopsteam24-cookiecutter.streamlit.app)
- **Repositorio GitHub**: [tu-repo-url]
- **MLflow UI**: http://localhost:5001 (local)
- **Documentación de Streamlit**: https://docs.streamlit.io

---

## 📊 Monitoring y Validación

### Dashboard Streamlit

**URL**: [https://mlopsteam24-cookiecutter.streamlit.app](https://mlopsteam24-cookiecutter.streamlit.app)

**Características**:
- ✅ Validación estructura Cookiecutter (95.2%)
- ✅ Verificación de directorios críticos
- ✅ Validación de archivos configuración
- ✅ Estado de sincronización DVC
- ✅ Métricas de cumplimiento

**Local**:
```bash
cd monitoring/dashboard
streamlit run streamlit_dashboard.py
```

### Verificación de Sincronización

**Script**: `scripts/validation/verify_sync.py`

Verifica:
1. ✅ DVC status (sin cambios pendientes)
2. ✅ Git status (working tree clean)
3. ✅ S3 sync (archivos en sync)
4. ✅ Environment consistency

```bash
make verify-sync
# o
python scripts/validation/verify_sync.py
```

**Output Esperado**:
```
✅ DVC Status: Clean
✅ Git Status: Clean
✅ S3 Sync: OK
✅ Environment: Consistent
```


## 🔄 Workflows y Contribución

### Workflow Estándar

#### 1. Antes de Comenzar

```bash
# Activar entorno
conda activate acoustic_ml

# Verificar sincronización
make verify-sync

# Actualizar datos
dvc pull
git pull
```

#### 2. Crear Branch

```bash
git checkout -b feat/nueva-funcionalidad
```

#### 3. Hacer Cambios

**Si modificas código**:
```bash
# Editar archivos
vim acoustic_ml/features.py

# Ejecutar tests
python tests/validate_features.py

# Los cambios están disponibles inmediatamente (instalación -e)
```

**Si modificas datos**:
```bash
# DVC tracking
dvc add data
git add data.dvc data/.gitignore
dvc push
```

**Si instalas paquetes**:
```bash
pip install nuevo-paquete
make freeze
git add requirements.txt
```

#### 4. Commit Changes

```bash
git add .
git commit -m "feat: descripción clara"
```

Seguir [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` nueva funcionalidad
- `fix:` corrección de bug
- `docs:` documentación
- `refactor:` refactorización
- `test:` tests
- `chore:` mantenimiento

#### 5. Push Changes

```bash
git push origin feat/nueva-funcionalidad
dvc push  # Si modificaste datos
```

#### 6. Pull Request

Crear PR a `main` con descripción clara.

### Buenas Prácticas

#### ✅ DO

- ✅ Ejecutar `make verify-sync` antes de comenzar
- ✅ Usar `DatasetManager` para gestionar datos
- ✅ Usar `create_sklearn_pipeline()` para producción
- ✅ Ejecutar tests antes de commit
- ✅ Documentar experimentos en MLflow
- ✅ Mantener notebooks limpios (sin outputs)
- ✅ Usar `RobustScaler` para outliers
- ✅ Escribir docstrings completos
- ✅ Seguir Conventional Commits
- ✅ Hacer `dvc push` después de modificar datos

#### ❌ DON'T

- ❌ Modificar datos sin DVC tracking
- ❌ Commitear archivos temporales
- ❌ Usar código legacy sin revisar
- ❌ Hacer commits sin tests
- ❌ Push sin `dvc push` (si hay datos nuevos)
- ❌ Commitear notebooks con outputs
- ❌ Modificar `requirements.txt` manualmente
- ❌ Ignorar warnings de validación

### Code Review Checklist

Antes de aprobar PR:
- [ ] Tests pasan
- [ ] Documentación actualizada
- [ ] No hay archivos temporales
- [ ] DVC en sync (si aplica)
- [ ] Código sigue estándares del proyecto
- [ ] Commit messages son claros

---

## 👥 Equipo de Desarrollo

<div align="center">

<table style="width:100%; border:none;">
  <tr>
    <td align="center" style="border:none; padding:20px 10px;">
      <img src="https://iili.io/Kw90kmB.png" alt="David Cruz Beltrán" width="160" style="border-radius: 50%; border: 5px solid #667eea; box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);"/>
      <h3>David Cruz Beltrán</h3>
      <img src="https://img.shields.io/badge/ID-A01360416-667eea?style=for-the-badge" alt="Matrícula"/>
      <p><strong>🔧 Software Engineer</strong><br/>
      <em>Pipeline Architecture & Code Quality</em></p>
    </td>
    <td align="center" style="border:none; padding:20px 10px;">
      <img src="https://iili.io/KuvsGKx.png" alt="Javier Augusto Rebull Saucedo" width="160" style="border-radius: 50%; border: 5px solid #764ba2; box-shadow: 0 8px 16px rgba(118, 75, 162, 0.4);"/>
      <h3>Javier Augusto Rebull Saucedo</h3>
      <img src="https://img.shields.io/badge/ID-A01795838-764ba2?style=for-the-badge" alt="Matrícula"/>
      <p><strong>⚙️ SRE / Data Engineer</strong><br/>
      <em>DevOps, Infrastructure & Data Versioning</em></p>
    </td>
    <td align="center" style="border:none; padding:20px 10px;">
      <img src="https://iili.io/Kw91d74.png" alt="Sandra Luz Cervantes Espinoza" width="160" style="border-radius: 50%; border: 5px solid #f093fb; box-shadow: 0 8px 16px rgba(240, 147, 251, 0.4);"/>
      <h3>Sandra Luz Cervantes Espinoza</h3>
      <img src="https://img.shields.io/badge/ID-A01796937-f093fb?style=for-the-badge" alt="Matrícula"/>
      <p><strong>🤖 ML Engineer / Data Scientist</strong><br/>
      <em>Model Development & Experimentation</em></p>
    </td>
  </tr>
</table>

</div>

---

## 📚 Recursos Adicionales

### Documentación

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

### Referencias del Proyecto

- `references/Diccionario_Variables_Musica_Turca.xlsx`: Diccionario de variables
- `references/Fase 1_Equipo24.pdf`: Entrega Fase 1
- `references/Fase 2_Equipo24.pdf`: Entrega Fase 2
- `references/Team24_Machine Learning Canvas v1.0.pdf`: ML Canvas

---

<div align="center">

**⭐ Si este proyecto te resulta útil, considera darle una estrella ⭐**

---

**Desarrollado con ❤️ por MLOps Team 24**

🏗️ **Arquitectura Profesional** | 🧪 **Testing Comprehensivo** | 🎯 **Production-Ready**

📊 **95.2% Cookiecutter Compliance** | ☁️ **Cloud-Native** | 🔄 **Fully Reproducible**

---

*Última actualización: Noviembre 2024 - Phase 3 Production Deployment*

**Estructura basada en**: [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

</div>

---

## 📊 Data Drift Detection Dashboard

**Dashboard en Vivo:**
🔗 [Turkish Drift MLOps - Streamlit Cloud](https://turskishdriftmlops.streamlit.app)

### 🎯 Descripción

Dashboard interactivo para monitoreo de Data Drift en el modelo de reconocimiento de emociones en música turca. Implementado con Streamlit + Plotly con paleta de colores profesional.

### 📱 Secciones del Dashboard

#### 1️⃣ **Resumen Ejecutivo** (Executive Summary)
- **Métrica Baseline:** Valor de referencia de predicción promedio
- **Mayor Impacto:** Escenario con mayor % de drift detectado
- **Escenarios Críticos:** Conteo de escenarios que requieren atención

**Visualizaciones:**
- Gráfico de cambio de media de predicciones por escenario
- Distribución de clases de emoción (Angry 😠, Happy 😊, Relax 😌, Sad 😢)
- Tabla de recomendaciones por escenario
- Alertas contextuales (INFO, WARNING, CRITICAL)

#### 2️⃣ **Análisis Detallado** (Detailed Analysis)
Selecciona un escenario individual para inspeccionar:
- **Métricas por escenario:** Total muestras, media de predicción, clases únicas, timestamp
- **Distribución de emociones:** Gráfico de barras con porcentajes
- **Histograma de predicciones:** Frecuencia de cada clase predicha

#### 3️⃣ **Comparación de Escenarios** (Scenario Comparison)
Comparación lado a lado de todos los escenarios:
- Baseline (sin drift)
- Mean Shift (+30% en todas las medias)
- Variance Change (×1.5 varianza)
- Combined Drift (Mean +20% + Var ×1.3 + outliers)

**Tabla resumen** con media de predicción y total de muestras.

#### 4️⃣ **Metadatos** (Metadata)
- **Información del entrenamiento:** Muestras, features, fecha generación, ruta datos
- **Escenarios de drift:** Descripción, impacto, causa de cada escenario
- **Visualización JSON crudo:** Para debugging y validación

### 🎨 Paleta de Colores
Diseño profesional con colores teal/verde vividos para máxima legibilidad en light y dark mode:
- Primary: `#17A2A2` (Teal vivido)
- Background: `#F0FFFE` (Blanco suave)
- Surface: `#B3E5E1` (Tarjetas verde pastel)

### 🔧 Stack Tecnológico
- **Framework:** Streamlit 1.32.0
- **Visualización:** Plotly 5.18.0
- **Data:** Pandas 2.1.3, NumPy 1.24.4
- **Deployment:** Streamlit Cloud

### 📂 Estructura
```
drift/
├── drift_streamlit_dashboard.py    # Código principal
├── requirements.txt                # Dependencias
├── drift_baseline.json
├── drift_mean_shift.json
├── drift_variance_change.json
├── drift_combined_drift.json
├── drift_executive_summary.json
└── drift_scenarios_metadata.json
```

### ⚡ Interpretación de Resultados

| Impacto | Rango | Acción |
|---------|-------|--------|
| LOW | < 1% | Monitor |
| MEDIUM | 1-3% | Investigate |
| HIGH | 3-5% | Investigate |
| CRITICAL | > 5% | Retrain Model |

### 🚀 Ejecutar Localmente
```bash
cd drift/
bash run_dashboard.sh
```

Abrirá en `http://localhost:8501`

### 📊 Hallazgos Clave
- ✅ **Mean Shift Robusto (0% impacto):** El modelo es resiliente a desplazamientos sistemáticos
- ⚠️ **Variance Change Vulnerable (3.37% impacto):** Necesita atención a cambios de ruido
- 🔴 **Combined Drift Crítico (4.33% impacto):** Requiere re-entrenamiento

---

