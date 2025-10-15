# 🛠️ Guía de Configuración del Proyecto MLOps

> 📄 **Versión:** 1.0
> 📅 **Última actualización:** 15 de octubre de 2025
> 👤 **Autor:** Equipo MLOps

---

## 📘 Descripción

Esta guía proporciona los pasos necesarios para **configurar el entorno de desarrollo y producción** del proyecto `mlops-fastapi-service`. Incluye instalación de dependencias, configuración de entornos, y preparación para ejecución local y despliegue.

---

## 🔹 1. Requisitos Previos

Antes de empezar, asegúrate de tener instalados:

* **Python 3.10 o superior**
* **pip** actualizado
* **Git**
* **Docker** (opcional, para contenedorización)
* **MLflow** (opcional, para tracking de experimentos)
* **conda** o **venv** (recomendado para entornos virtuales)

Comprobación rápida:

```bash
python --version
pip --version
git --version
docker --version
```

---

## 🔹 2. Clonar el Repositorio

```bash
git clone https://github.com/jrebull/MLOps_Team24
cd mlops-fastapi
```

---

## 🔹 3. Configuración de Entorno Virtual

### Usando `venv`:

```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

### Usando `conda`:

```bash
conda create -n mlops_env python=3.10
conda activate mlops_env
```

---

## 🔹 4. Instalación de Dependencias

Instalar las dependencias principales:

```bash
pip install -r requirements.txt
```

Para dependencias de desarrollo (pruebas y linters):

```bash
pip install -r requirements_dev.txt
```

> 💡 Alternativa: Si usas `poetry`, ejecutar:

```bash
poetry install
```

---

## 🔹 5. Configuración de Variables de Entorno

Copia el archivo de ejemplo y modifica según tu entorno:

```bash
cp .env.example .env
```

Variables típicas:

```
DATABASE_URL=sqlite:///data/db.sqlite3
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=tu_clave_secreta
```

---

## 🔹 6. Inicializar MLflow (Opcional)

Para el seguimiento de experimentos:

```bash
mlflow ui
```

Accede a la interfaz web en: [http://localhost:5000](http://localhost:5000)

---

## 🔹 7. Pruebas Iniciales

Ejecuta los tests unitarios para confirmar que todo funciona correctamente:

```bash
pytest tests/ -v
```

---

## 🔹 8. Ejecutar la API FastAPI

Levanta el servidor localmente:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Accede a la documentación automática en: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔹 9. Contenerización con Docker (Opcional)

### Construir la imagen:

```bash
docker build -t mlops-fastapi .
```

### Ejecutar el contenedor:

```bash
docker run -p 8000:8000 mlops-fastapi
```

---

## 🔹 10. Notas Finales

* Siempre activa tu entorno virtual antes de ejecutar scripts o tests.
* Mantén actualizadas las dependencias con:

```bash
pip install --upgrade -r requirements.txt
```

* Para entornos de producción, configura las variables de entorno de manera segura y considera el uso de **Docker** o **Kubernetes**.

> ✅ Con estos pasos, tu entorno estará listo para desarrollo, pruebas y despliegue del proyecto MLOps FastAPI.



### 📤 Agregar nuevos datos

Si tienes un nuevo dataset:

```bash
# 1. Coloca tu archivo en data/processed/
cp ~/Downloads/nuevo_dataset.csv data/processed/

# 2. Actualiza el tracking de DVC
dvc add data

# 3. Sube a S3
dvc push
# o: make push

# 4. Commitea los metadatos a Git (NO los CSV)
git add data.dvc data/.gitignore
git commit -m "feat: add nuevo_dataset.csv"
git push
```

### 🔄 Actualizar un dataset existente

Si modificaste un archivo de datos:

```bash
# 1. Edita tu archivo
vim data/processed/turkish_music_emotion_v2_cleaned_full.csv

# 2. Actualiza DVC (detecta el cambio automáticamente)
dvc add data

# 3. Sube la nueva versión a S3
dvc push

# 4. Commitea el cambio de metadatos
git add data.dvc
git commit -m "feat: update v2_cleaned_full with improved imputation"
git push
```

### ⏮️ Volver a una versión anterior

```bash
# 1. Encuentra el commit donde estaba la versión que quieres
git log --oneline data.dvc

# 2. Vuelve a ese commit
git checkout <commit_hash> data.dvc

# 3. Descarga esa versión desde S3
dvc checkout

# 4. Si quieres quedarte con esta versión:
git add data.dvc
git commit -m "revert: rollback to previous dataset version"
git push
```

### 🔍 Verificar estado de los datos

```bash
# Ver si tus datos están sincronizados con S3
dvc status

# Ver configuración de remotes
dvc remote list

# Ver qué archivos trackea DVC
cat data.dvc
```

### 🌐 Ver datos en AWS Console

Accede visualmente a tus datos:

1. Ve a: **https://s3.console.aws.amazon.com/s3/buckets/mlops24-haowei-bucket**
2. Navega a: `files/` → `md5/`
3. Verás carpetas con tus datasets (almacenados por hash MD5)

### 🚨 Problemas comunes

**Problema:** `dvc pull` falla con error de AWS
```bash
# Solución: Verifica tus credenciales
aws s3 ls s3://mlops24-haowei-bucket/
# Si falla, reconfigura:
aws configure
```

**Problema:** "Cache is missing" o archivos no se descargan
```bash
# Solución: Fuerza la descarga
dvc pull -f
```


## 💻 Uso

### 🛠️ Usando el Makefile

Este repo incluye un `Makefile` con comandos cortos para las tareas comunes.

#### Comandos disponibles

```bash
# 1) Configurar entorno y dependencias
make setup

# 2) Abrir Jupyter Lab
make jupyter

# 3) Levantar MLflow en http://127.0.0.1:5001
make mlflow

# 4) Reproducir pipeline (solo si hubo cambios)
make reproduce

# 5) Forzar etapa de entrenamiento (nuevo run en MLflow)
make train

# 6) Ver métricas actuales y diferencias
make metrics
make diff

# 7) Sincronizar artefactos con el remoto DVC (S3)
make pull
make push

# 8) Limpiar el entorno local
make clean
make clean-caches

# 9) Exportar dependencias actuales
make freeze

# 10) Verificar sincronización antes de trabajar
make verify-sync

# 11) Muestra si hay datos desactualizados
make status
```

### 🐍 Usando el Módulo acoustic_ml

El proyecto está organizado como un módulo Python instalable. Ejemplos de uso:

#### Cargar datos

```python
from acoustic_ml.dataset import load_processed_data

# Cargar versión recomendada para ML (⭐ RECOMENDADO)
df = load_processed_data("turkish_music_emotion_v2_cleaned_full.csv")
print(f"✅ Dataset óptimo cargado: {df.shape}")

# Cargar otras versiones para comparación
df_original = load_processed_data("turkish_music_emotion_v1_original.csv")
df_aligned = load_processed_data("turkish_music_emotion_v2_cleaned_aligned.csv")

# Comparar dimensiones
print(f"\n📊 Comparación de versiones:")
print(f"v1_original:        {df_original.shape[0]} filas")
print(f"v2_cleaned_aligned: {df_aligned.shape[0]} filas")
print(f"v2_cleaned_full:    {df.shape[0]} filas (+{df.shape[0] - df_aligned.shape[0]} adicionales)")
```

#### Feature Engineering

```python
from acoustic_ml.features import create_features, select_features

# Crear features adicionales
df_with_features = create_features(df)

# Seleccionar features específicas
features = ['tempo', 'energy', 'valence']
df_selected = select_features(df_with_features, features)
```

#### Entrenar modelos

```python
from acoustic_ml.modeling.train import train_model
import mlflow

# Entrenar modelo con versión recomendada
# (registra automáticamente en MLflow)
with mlflow.start_run():
    # Documentar versión de dataset
    mlflow.set_tag("dataset_version", "v2_cleaned_full")
    mlflow.set_tag("dataset_rows", len(X_train))
    
    # Entrenar
    model = train_model(X_train, y_train)
```

#### Hacer predicciones

```python
from acoustic_ml.modeling.predict import load_model, predict

# Cargar modelo entrenado
model = load_model("baseline_model.pkl")

# Predecir
predictions = predict(model, X_test)
```

### Trabajar con Notebooks

**Jupyter Lab:**
```bash
jupyter-lab
# o usando make:
make jupyter
```

**Importar módulo en notebooks:**
```python
from acoustic_ml.dataset import load_processed_data
from acoustic_ml.config import PROCESSED_DATA_DIR

# ⭐ Usar versión recomendada
DATASET_VERSION = "turkish_music_emotion_v2_cleaned_full.csv"
df = load_processed_data(DATASET_VERSION)

print(f"📊 Dataset: {DATASET_VERSION}")
print(f"📏 Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"📍 Ubicación: {PROCESSED_DATA_DIR}")
```

### Tracking de Experimentos

Inicia el servidor MLflow:

```bash
mlflow ui --port 5001
# o usando make:
make mlflow
```

Accede a la interfaz en: **http://127.0.0.1:5001**

### Pipeline DVC

**Ejecutar el pipeline completo:**
```bash
dvc repro
# o usando make:
make reproduce
```

**Ver métricas actuales:**
```bash
dvc metrics show
# o usando make:
make metrics
```

**Comparar métricas entre commits:**
```bash
dvc metrics diff
# o usando make:
make diff
```

---



## 📓 Buenas Prácticas con Notebooks

Instala hooks para limpiar outputs y tener diffs legibles:

```bash
make nb-hooks
```

**Beneficios:**
- `nbstripout` limpia salidas/celdas ejecutadas al commitear
- `nbdime` muestra diffs de `.ipynb` de forma amigable

**Convención de nombres para notebooks:**
```
<número>.<versión>-<iniciales>-<descripción-corta>.ipynb

Ejemplos:
- 1.0-jrs-initial-data-exploration.ipynb
- 2.0-hw-feature-engineering.ipynb
- 3.1-sc-model-evaluation.ipynb
```

**Template recomendado para notebooks:**
```python
# === CONFIGURACIÓN INICIAL ===
import pandas as pd
from acoustic_ml.dataset import load_processed_data
from acoustic_ml.config import PROCESSED_DATA_DIR

# ⭐ Definir versión de dataset a usar
DATASET_VERSION = "turkish_music_emotion_v2_cleaned_full.csv"

print(f"📊 Notebook: [Nombre del notebook]")
print(f"📅 Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
print(f"📦 Dataset: {DATASET_VERSION}")

# Cargar datos
df = load_processed_data(DATASET_VERSION)
print(f"✅ Datos cargados: {df.shape}")
```
