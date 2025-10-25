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
[![Tests](https://img.shields.io/badge/tests-passing-success?logo=pytest&logoColor=white)](#-testing-y-validación)
[![Code Quality](https://img.shields.io/badge/code-production--ready-brightgreen?logo=python&logoColor=white)](#-arquitectura-del-código)
[![Accuracy](https://img.shields.io/badge/accuracy-80.17%25-success?logo=tensorflow&logoColor=white)](#-modelo-y-resultados)
[![Repo Status](https://img.shields.io/badge/repo-phase%202%20complete-success?logo=git&logoColor=white)](#-información-académica)

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
- [API y Deployment](#-api-y-deployment)
- [Monitoring y Validación](#-monitoring-y-validación)
- [Testing](#-testing-y-validación)
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

#### Código y Arquitectura
- 🏗️ **Módulo Python profesional** (`acoustic_ml`)
- 🎯 **Pipeline sklearn end-to-end** listo para producción
- 🧪 **Testing comprehensivo** con validación automatizada
- 🛡️ **Manejo robusto de outliers** y datos
- 📦 **API REST** con FastAPI (en desarrollo)

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
- **Fase Actual:** Fase 2 - Completada ✅

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

**Dashboard de Validación**: [https://mlopsteam24-cookiecutter2.streamlit.app](https://mlopsteam24-cookiecutter.streamlit.app)

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

### Uso Básico del Módulo

#### Cargar Datos

```python
from acoustic_ml.dataset import load_dataset

# Cargar dataset principal
df = load_dataset('v2_cleaned_full')

# O con splits
X_train, X_test, y_train, y_test = load_dataset('v2_cleaned_full', return_splits=True)
```

#### Feature Engineering

```python
from acoustic_ml.features import FeaturePipeline

# Crear pipeline
pipeline = FeaturePipeline()

# Transformar datos
X_transformed = pipeline.fit_transform(X_train, y_train)
X_test_transformed = pipeline.transform(X_test)
```

#### Entrenar Modelo

```python
from acoustic_ml.modeling.sklearn_pipeline import create_sklearn_pipeline

# Crear pipeline completo
model_pipeline = create_sklearn_pipeline(model_type='random_forest')

# Entrenar
model_pipeline.fit(X_train, y_train)

# Predecir
predictions = model_pipeline.predict(X_test)

# Evaluar
accuracy = model_pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

#### Visualizaciones

```python
from acoustic_ml.plots import plot_confusion_matrix, plot_feature_importance

# Confusion matrix
plot_confusion_matrix(y_test, predictions, save_path='reports/figures/cm.png')

# Feature importance
plot_feature_importance(model_pipeline, feature_names, top_n=20)
```

### Scripts Rápidos

#### Entrenar Modelo Baseline

```bash
python scripts/training/train_baseline.py
```

#### Ejecutar Experimentos MLflow

```bash
python scripts/training/run_mlflow_experiments.py
```

#### Análisis de Outliers

```bash
python scripts/analysis/analyze_outliers.py
```

#### Validación Completa

```bash
python tests/test_full_integration.py
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

## 🌐 API y Deployment

### FastAPI Application

**Estructura**:
```
app/
├── main.py              <- Entry point
├── api/
│   ├── main.py          <- API router
│   ├── endpoints.py     <- Endpoints
│   └── schemas.py       <- Pydantic models
├── core/
│   ├── config.py        <- Configuración
│   └── logger.py        <- Logging
└── services/
    └── model_service.py <- Modelo service
```

**Endpoints Planeados**:
```
POST /predict          - Predicción single
POST /predict/batch    - Predicción batch
GET  /model/info       - Info del modelo
GET  /health           - Health check
```

**Status**: En desarrollo (Fase 3)

### Docker Deployment

**Archivo**: `docker-compose.yml`

**Services**:
- `mlflow`: MLflow tracking server (port 5001)
- `minio` (planeado): S3-compatible storage

```bash
# Iniciar services
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener
docker-compose down
```

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

---

## 🧪 Testing y Validación

### Test Suite

**Directorio**: `tests/`

```bash
# Ejecutar todos los tests
make test

# Tests específicos
python tests/test_dataset_equivalence.py
python tests/test_sklearn_pipeline.py
python tests/test_full_integration.py
```

### Tests Disponibles

| Test | Propósito |
|------|-----------|
| `test_dataset_equivalence.py` | Validar equivalencia entre datasets |
| `test_ml_pipeline.py` | Test pipeline MLOps completo |
| `test_sklearn_pipeline.py` | Test pipeline sklearn |
| `test_full_integration.py` | Integration tests end-to-end |
| `validate_cookiecutter.py` | Validar estructura Cookiecutter |
| `validate_dataset.py` | Validar módulo dataset |
| `validate_features.py` | Validar módulo features |
| `validate_plots.py` | Validar módulo plots |

### Validación de Módulos

```bash
# Dataset
python tests/validate_dataset.py

# Features
python tests/validate_features.py

# Plots
python tests/validate_plots.py
```

### CI/CD (Planeado)

- GitHub Actions para tests automáticos
- Pre-commit hooks para linting
- Automated Cookiecutter validation

---

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

*Última actualización: Octubre 2024 - Fase 2 Completada*

**Estructura basada en**: [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

</div>
