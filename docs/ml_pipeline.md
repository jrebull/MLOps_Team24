# 🧠 Documentación del Pipeline de Machine Learning

## 📘 Descripción General

Este documento explica el **pipeline de Machine Learning (ML)** desarrollado para este proyecto de **MLOps**, detallando sus etapas principales: **carga de datos, preprocesamiento, entrenamiento, evaluación y seguimiento de experimentos** mediante herramientas como MLflow.

El objetivo es garantizar un flujo automatizado, reproducible y escalable para el desarrollo y despliegue de modelos de ML en entornos productivos.

---

## 📂 Arquitectura del Pipeline

La arquitectura del pipeline está organizada en módulos independientes, facilitando su mantenimiento y evolución:

```
mlops/
│
├── data_loader.py         # Carga de datos (CSV, bases de datos, APIs)
├── preprocessor.py        # Limpieza y transformación de datos
├── model_trainer.py       # Entrenamiento y validación de modelos
├── evaluator.py           # Evaluación y métricas de rendimiento
├── tracking.py            # Registro y seguimiento de experimentos (MLflow)
└── utils/                 # Funciones auxiliares
```

---

## 📥 Carga de Datos (`DataLoader`)

El módulo **DataLoader** se encarga de la ingesta de datos desde diversas fuentes y de la validación inicial.

### Funcionalidades principales:

* Admite múltiples formatos: `.csv`, `.parquet`, SQL, API REST.
* Valida el esquema de columnas y tipos de datos.
* Maneja errores de lectura y archivos faltantes.

### Ejemplo de uso:

```python
from ml_pipeline import DataLoader
loader = DataLoader(path='data/interim/turkish_music_emotion_cleaned.csv')
df = loader.load_csv()
```

---

#### 📦 Versión 1: Original (v1_original)

```
📍 Ubicación: data/processed/turkish_music_emotion_v1_original.csv
📏 Dimensiones: 400 filas × 21 columnas
🎯 Uso: Baseline y comparaciones históricas
🔖 Estado: Referencia oficial (sin modificaciones)
```

**Características:**
- Dataset crudo sin modificaciones post-descarga
- Incluye todas las inconsistencias originales del dataset fuente
- Punto de referencia oficial para todas las comparaciones
- Útil para reproducir análisis iniciales y validar mejoras

**Mejoras respecto a versión 0:**
- ✅ Versionado formal con DVC
- ✅ Documentación estructurada
- ✅ Punto de referencia estable

**Cuándo usar:**
- ✅ Baseline para comparar todas las versiones posteriores
- ✅ Validación de procesos de limpieza aplicados
- ✅ Documentación de transformaciones históricas
- ✅ Reproducción de experimentos iniciales del proyecto
- ❌ NO recomendado para entrenar nuevos modelos

---

#### 🔄 Versión 2: Limpia Alineada (v2_cleaned_aligned)

```
📍 Ubicación: data/processed/turkish_music_emotion_v2_cleaned_aligned.csv
📏 Dimensiones: 400 filas × 21 columnas
🎯 Uso: Comparaciones directas fila por fila con v1
🔖 Estado: Producción (análisis comparativos)
```

**Características:**
- Dataset limpio manteniendo exactamente las 400 filas originales
- Mismo orden y estructura que v1_original para facilitar diffs
- Valores faltantes imputados con estrategias estadísticas
- Outliers corregidos sin eliminar filas
- Perfecta alineación 1:1 con v1 para análisis de impacto

**Mejoras respecto a v1:**
- ✅ Limpieza sistemática de valores faltantes
- ✅ Corrección de outliers estadísticos
- ✅ Normalización de features numéricas
- ✅ Validación de consistencia de datos
- ✅ Preservación de estructura original (400 filas)

**Cuándo usar:**
- ✅ Análisis de impacto de limpieza (antes/después)
- ✅ Validación de transformaciones específicas fila por fila
- ✅ Reportes que comparan resultados con/sin limpieza
- ✅ Auditoría de cambios aplicados al dataset
- ⚠️ Puede usarse para entrenar modelos, pero v2_cleaned_full es superior

---

#### ⭐ Versión 3: Limpia Completa (v2_cleaned_full) **[RECOMENDADO]**

```
📍 Ubicación: data/processed/turkish_music_emotion_v2_cleaned_full.csv
📏 Dimensiones: 408 filas × 21 columnas
🎯 Uso: Entrenamiento de modelos de producción
🔖 Estado: Producción (versión oficial para ML)
```

**Características:**
- Dataset limpio más completo del proyecto (+8 filas adicionales)
- Máxima calidad y cantidad de datos para Machine Learning
- Duplicados inteligentemente consolidados sin pérdida de información
- Outliers corregidos manteniendo variabilidad natural
- Features normalizadas y validadas para ML
- Estrategias avanzadas de imputación de valores faltantes

**Mejoras respecto a v2_aligned:**
- ✅ +8 filas adicionales recuperadas mediante análisis avanzado
- ✅ Consolidación inteligente de duplicados (preservando información única)
- ✅ Imputación avanzada de valores faltantes (KNN, iterativa)
- ✅ Detección y corrección robusta de outliers multivariados
- ✅ Validación cruzada de consistencia en todas las features
- ✅ Máxima representatividad del espacio de características

**Cuándo usar:**
- ✅ **Entrenamiento de todos los modelos nuevos** (PRIMERA OPCIÓN)
- ✅ Experimentación y búsqueda de hiperparámetros
- ✅ Evaluación de performance de modelos
- ✅ Pipeline de producción y despliegue
- ✅ Benchmarking y competiciones internas
- ✅ Validación final de modelos antes de producción

---

### 📋 Comparación Rápida de Versiones

| Versión | Archivo | Filas | Uso Principal | Estado | Recomendación |
|---------|---------|-------|---------------|--------|---------------|
| **v0** | `turkish_music_emotion_cleaned.csv` | Variable | Histórico (notebook inicial) | 📚 Archivo | ❌ No usar |
| **v1** | `v1_original.csv` | 400 | Baseline sin modificar | 📖 Referencia | Solo comparaciones |
| **v2a** | `v2_cleaned_aligned.csv` | 400 | Comparación directa con v1 | 🔄 Análisis | Análisis de impacto |
| **v3** | `v2_cleaned_full.csv` | 408 | **Entrenamiento ML** | ⭐ Producción | **✅ USAR ESTO** |

---


### 🔧 Cómo usar cada versión en código

#### Ejemplo 1: Cargar versión recomendada

```python
from acoustic_ml.dataset import load_processed_data

# Versión recomendada para ML
df_full = load_processed_data("turkish_music_emotion_v2_cleaned_full.csv")
print(f"✅ Dataset óptimo cargado: {df_full.shape[0]} filas")
```

#### Ejemplo 2: Análisis comparativo entre versiones

```python
from acoustic_ml.dataset import load_processed_data
import pandas as pd

# Cargar las 3 versiones principales
df_v1 = load_processed_data("turkish_music_emotion_v1_original.csv")
df_v2a = load_processed_data("turkish_music_emotion_v2_cleaned_aligned.csv")
df_v3 = load_processed_data("turkish_music_emotion_v2_cleaned_full.csv")

# Comparar dimensiones
print(f"v1_original:        {df_v1.shape[0]} filas")
print(f"v2_cleaned_aligned: {df_v2a.shape[0]} filas")
print(f"v2_cleaned_full:    {df_v3.shape[0]} filas (+{df_v3.shape[0] - df_v2a.shape[0]} filas)")

# Comparar calidad de datos
print("\n📊 Valores faltantes por versión:")
print(f"v1: {df_v1.isnull().sum().sum()} valores faltantes")
print(f"v2a: {df_v2a.isnull().sum().sum()} valores faltantes")
print(f"v3: {df_v3.isnull().sum().sum()} valores faltantes")
```

#### Ejemplo 3: Uso en notebooks

```python
import pandas as pd
from acoustic_ml.config import PROCESSED_DATA_DIR

# Método 1: Usando el módulo
from acoustic_ml.dataset import load_processed_data
df = load_processed_data("turkish_music_emotion_v2_cleaned_full.csv")

# Método 2: Carga directa con pandas
df = pd.read_csv(PROCESSED_DATA_DIR / "turkish_music_emotion_v2_cleaned_full.csv")

print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"📍 Ubicación: {PROCESSED_DATA_DIR}")
```

#### Ejemplo 4: Validación de versión correcta

```python
from acoustic_ml.dataset import load_processed_data

def validate_dataset_version(df, expected_rows=408):
    """Valida que estés usando la versión correcta del dataset"""
    if df.shape[0] == expected_rows:
        print(f"✅ Usando v2_cleaned_full ({expected_rows} filas) - CORRECTO")
        return True
    else:
        print(f"⚠️  Advertencia: {df.shape[0]} filas (esperadas: {expected_rows})")
        print("💡 Considera usar 'turkish_music_emotion_v2_cleaned_full.csv'")
        return False

# Uso
df = load_processed_data("turkish_music_emotion_v2_cleaned_full.csv")
validate_dataset_version(df)
```

---

### 🎯 Casos de Uso por Versión

#### Cuando usar `turkish_music_emotion_cleaned.csv`:
- 🔍 Auditoría histórica del primer proceso de limpieza
- 📖 Documentación de evolución del proyecto
- ❌ **NUNCA para entrenamiento de modelos**

#### Cuando usar `v1_original.csv`:
- 📊 Establecer baseline de performance
- 📈 Medir impacto de limpieza en métricas
- 📝 Documentar transformaciones aplicadas
- ⚖️ Comparar con estado original del dataset

#### Cuando usar `v2_cleaned_aligned.csv`:
- 🔬 Análisis fila por fila de cambios aplicados
- 📊 Estudios de impacto de limpieza específica
- 🔍 Validar que la limpieza preserva estructura
- 📉 Comparación directa antes/después (400 filas constantes)

#### Cuando usar `v2_cleaned_full.csv` ⭐:
- 🤖 **Entrenar TODOS los modelos nuevos**
- 🔧 Ajuste de hiperparámetros
- 📊 Evaluación de performance
- 🚀 Despliegue en producción
- 🏆 Competiciones y benchmarks
- ✅ Cualquier tarea de Machine Learning

---

## 🧹 Preprocesamiento (`Preprocessor`)

El **Preprocessor** transforma los datos brutos en un formato adecuado para el modelado.

### Etapas principales:

1. **Eliminación de duplicados** – Evita la redundancia en los datos.
2. **Eliminación de columnas constantes** – Quita variables sin información relevante.
3. **Codificación de etiquetas** – Transforma variables categóricas a formato numérico.
4. **Normalización de datos sesgados** – Aplica transformaciones logarítmicas o Box-Cox.
5. **Detección de outliers** – Filtra valores extremos mediante Z-score o IQR.

### Ejemplo:

```python
from ml_pipeline import Preprocessor
pre = Preprocessor(df)
pre.drop_duplicates()
pre.encode_labels()
pre.normalize_skewed(threshold=0.5)
clean_df = pre.get_clean_df()
```

---

## ⚙️ Entrenamiento de Modelos (`ModelTrainer`)

El módulo **ModelTrainer** maneja la separación de datos, el escalado y el entrenamiento de distintos modelos de ML.

### Funcionalidades:

* División de datos en entrenamiento y prueba (train/test split).
* Escalado mediante `StandardScaler` o `MinMaxScaler`.
* Modelos disponibles: **Regresión Logística**, **KNN**, **Random Forest**, **XGBoost**.
* Soporte para **validación cruzada (cross-validation)**.

### Ejemplo:

```python
trainer = ModelTrainer(clean_df)
trainer.split_and_scale(test_size=0.2)
trainer.train_logistic()
```

---

## 📊 Evaluación de Modelos (`Evaluator`)

El **Evaluator** calcula las métricas clave del desempeño del modelo y genera visualizaciones de resultados.

### Métricas incluidas:

* Accuracy (exactitud)
* Precision, Recall, F1-score
* ROC-AUC
* Matriz de confusión

### Ejemplo:

```python
from ml_pipeline import Evaluator
eval = Evaluator(model, X_test, y_test)
eval.compute_metrics()
eval.plot_confusion_matrix()
```

---

## 📈 Seguimiento de Experimentos (`Tracking`)

El módulo **Tracking** utiliza **MLflow** para registrar parámetros, métricas y versiones de modelos.

### Características:

* Guarda los resultados de cada ejecución de entrenamiento.
* Permite comparar experimentos con distintas configuraciones.
* Compatible con almacenamiento local o remoto (S3, Databricks, GCP, etc.).

### Ejemplo:

```python
from ml_pipeline import Tracking
track = Tracking(experiment_name='ml_pipeline_v1')
track.log_params(params)
track.log_metrics(metrics)
track.log_model(model)
```

---

## 🧪 Pruebas Unitarias

El proyecto utiliza **pytest** para validar el correcto funcionamiento de cada componente del pipeline.

```bash
pytest tests/test_ml_pipeline.py -v
```

Las pruebas cubren la carga de datos, la limpieza, la transformación, el entrenamiento y la evaluación de modelos.

---

## 🚀 Despliegue e Integración

Una vez validado el modelo, puede:

* Serializarse con `joblib`, `pickle` o almacenarse en **MLflow Models**.
* Servirse mediante **FastAPI**, **AWS Lambda** o **Docker**.

### Ejemplo de endpoint en FastAPI:

```python
@app.post('/predict')
def predict(data: InputSchema):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)
    return {"prediccion": int(prediction[0])}
```

---

## 🔁 Automatización CI/CD

Integrado con herramientas como:

* **GitHub Actions** o **GitLab CI**: ejecución automática de pruebas y despliegues.
* **Docker**: contenedorización del entorno.
* **MLflow Model Registry**: control de versiones y promoción de modelos.

---

## 📚 Referencias

* [Documentación de MLflow](https://mlflow.org/docs/latest/index.html)
* [Guía de scikit-learn](https://scikit-learn.org/stable/user_guide.html)
* [Documentación de FastAPI](https://fastapi.tiangolo.com/)
* [Buenas Prácticas en MLOps (Google Cloud)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
