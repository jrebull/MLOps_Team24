# 🧠 API de Entrenamiento y Predicción de Modelos  
**Archivo:** `api_endpoints.md`  
**Módulo:** `app.api.router`

---

## 📘 Descripción General

Este módulo define los **endpoints principales de la API** para el ciclo de vida de un modelo de *Machine Learning*.

Utiliza **FastAPI** como framework para exponer servicios REST que permiten:

- ✅ Verificar el estado del servicio (`/health`)  
- 🧩 Entrenar un modelo de manera asíncrona (`/train`)  
- 🔮 Realizar predicciones (`/predict`)

El servicio de modelo (`ModelService`) encapsula la lógica de entrenamiento y predicción, mientras que los esquemas (`schemas`) definen los contratos de entrada y salida (basados en *Pydantic*).

---

## 🧩 Dependencias Importadas

```python
from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from app.api import schemas
from app.services.model_service import ModelService
from app.core.logger import logger
```

**Descripción de dependencias:**

| Módulo | Descripción |
|---------|--------------|
| **APIRouter** | Permite definir y agrupar endpoints modularmente en FastAPI. |
| **BackgroundTasks** | Ejecuta procesos en segundo plano, como el entrenamiento de un modelo. |
| **HTTPException** | Controla y devuelve errores HTTP personalizados. |
| **Depends** | Facilita la inyección de dependencias (por ejemplo, el `ModelService`). |
| **schemas** | Define los esquemas de datos (`TrainRequest`, `PredictRequest`, `PredictResponse`). |
| **ModelService** | Servicio que contiene la lógica de Machine Learning. |
| **logger** | Sistema centralizado de logging para registrar eventos y errores. |

---

## ⚙️ Inicialización del Router

```python
router = APIRouter()
```

**Descripción:**  
Crea una instancia del enrutador que agrupará los endpoints de esta sección.  
Posteriormente se registrará en el archivo principal `main.py`.

---

## 🧱 Dependencia: `get_model_service`

```python
def get_model_service():
    return ModelService()
```

**Propósito:**  
Función que devuelve una instancia del servicio de modelo.  
Se utiliza con `Depends()` para inyectar la dependencia en los endpoints.

**Patrón aplicado:**  
➡️ *Dependency Injection Pattern* — Mejora la mantenibilidad y testabilidad del código.

---
**Execute api**  
```python
uvicorn main:app --reload
```

**Swagger api**  
```python
http://127.0.0.1:8000/docs
```
---

## 🩺 Endpoint: `GET /health` — Verificación de estado

```python
@router.get("/health")
async def health():
    return {"status": "ok"}
```

**Propósito:**  
Permite verificar que la API está activa y respondiendo.

**Respuesta esperada:**
```json
{
  "status": "ok"
}
```

**Códigos de estado posibles:**
- `200 OK` — Servicio operativo.

---

## 🧠 Endpoint: `POST /train` — Entrenamiento del Modelo

```python
@router.post("/train")
async def train(
    req: schemas.TrainRequest,
    background_tasks: BackgroundTasks,
    svc: ModelService = Depends(get_model_service)
):
    def _train():
        try:
            path = svc.train(req.data_path, req.model_name)
            logger.info("Training finished, saved: %s", path)
        except Exception as e:
            logger.exception("Training failed")

    background_tasks.add_task(_train)
    return {"status": "started", "model_name": req.model_name}
```

### 🧾 Descripción
Este endpoint inicia el **entrenamiento de un modelo** en segundo plano utilizando `BackgroundTasks`.  
El proceso no bloquea la respuesta HTTP, permitiendo que la API siga disponible mientras se entrena el modelo.

### 📥 Parámetros de entrada

| Campo | Tipo | Descripción |
|--------|------|-------------|
| `data_path` | `str` | Ruta al dataset utilizado para el entrenamiento. |
| `model_name` | `str` | Nombre del modelo a entrenar o sobrescribir. |

### 📤 Respuesta

```json
{
  "status": "started",
  "model_name": "nombre_del_modelo"
}
```

### ⚠️ Posibles Errores
- Errores internos durante el entrenamiento se registran con `logger.exception`.
- No se detiene la ejecución del servidor, ya que la tarea es asíncrona.

### 🧩 Ejemplo de uso (cURL)
```bash
curl -X POST "http://localhost:8000/train"      -H "Content-Type: application/json"      -d '{"data_path": "data/train.csv", "model_name": "random_forest_v1"}'
```

---

## 🔮 Endpoint: `POST /predict` — Predicción

```python
@router.post("/predict", response_model=schemas.PredictResponse)
async def predict(
    req: schemas.PredictRequest,
    svc: ModelService = Depends(get_model_service)
):
    try:
        result = svc.predict(req.features)
        return schemas.PredictResponse(
            prediction=result["prediction"],
            probabilities=result["probabilities"]
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
```

### 🧾 Descripción
Este endpoint recibe un conjunto de **características** y devuelve la **predicción del modelo entrenado** junto con las probabilidades asociadas.

### 📥 Parámetros de entrada

| Campo | Tipo | Descripción |
|--------|------|-------------|
| `features` | `dict` | Diccionario con los valores de entrada requeridos por el modelo. |

### 📤 Respuesta esperada

```json
{
  "prediction": "label_predicha",
  "probabilities": {"label_1": 0.75, "label_2": 0.25}
}
```

### ⚠️ Posibles errores

| Código | Causa | Descripción |
|--------|--------|-------------|
| `500` | Error interno | Fallo en la ejecución del modelo o formato inválido de entrada. |

### 🧩 Ejemplo de uso (cURL)
```bash
curl -X POST "http://localhost:8000/predict"      -H "Content-Type: application/json"      -d '{"features": {"age": 45, "income": 78000, "city": "NY"}}'
```

---

## 📦 Schemas esperados (estructura simplificada)

### `TrainRequest`
```python
class TrainRequest(BaseModel):
    data_path: str
    model_name: str
```

### `PredictRequest`
```python
class PredictRequest(BaseModel):
    features: dict
```

### `PredictResponse`
```python
class PredictResponse(BaseModel):
    prediction: Any
    probabilities: dict
```

---

## 🧰 Ejemplo de flujo completo

1. Llamar a `/train` con la ruta de datos → inicia el entrenamiento.  
2. Esperar a que el modelo esté disponible.  
3. Llamar a `/predict` con las características deseadas → obtiene la predicción.  
4. Usar `/health` para monitorear la disponibilidad del servicio.

---

## ⚙️ Buenas prácticas MLOps aplicadas

- **Separación de responsabilidades:**  
  Los endpoints solo orquestan; la lógica de ML se encuentra en `ModelService`.

- **Tareas asíncronas:**  
  Entrenamiento manejado con `BackgroundTasks` para evitar bloqueos.

- **Logging estructurado:**  
  Registra eventos de entrenamiento y errores con `logger`.

- **Inyección de dependencias:**  
  Facilita testeo unitario y mantenimiento.

- **Validación Pydantic:**  
  Garantiza consistencia en las entradas/salidas de la API.

---

## 📄 Autoría y versión

**Autor:** Equipo 24 MLOps  
**Última actualización:** Octubre 2025  
**Framework:** FastAPI v0.115+  
