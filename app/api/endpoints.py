"""
API Endpoints - Turkish Music Emotion Recognition
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Dict
from datetime import datetime
import logging

from app.api import schemas
from app.services.model_service import ModelService
from app.core.logger import logger

# ========================
# ROUTER
# ========================
router = APIRouter()

# ========================
# DEPENDENCY INJECTION
# ========================
def get_model_service() -> ModelService:
    """Dependency para inyectar ModelService"""
    return ModelService()


# ========================
# ENDPOINTS
# ========================

@router.get(
    "/health",
    response_model=schemas.HealthResponse,
    summary="Health Check",
    tags=["System"],
    responses={
        200: {
            "description": "Servicio operacional",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "model_name": "turkish-music-emotion-rf",
                        "model_version": "production"
                    }
                }
            }
        }
    }
)
async def health(svc: ModelService = Depends(get_model_service)) -> schemas.HealthResponse:
    """
    Verificar que el servicio esté operacional.
    
    Returns:
        - status: "ok" si todo funciona
        - model_name: Nombre del modelo en uso
        - model_version: Versión del modelo
    """
    try:
        svc.load()
        return schemas.HealthResponse(
            status="ok",
            model_name="turkish-music-emotion-rf",
            model_version="production"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable"
        )


@router.post(
    "/predict",
    response_model=schemas.PredictResponse,
    summary="Predecir emoción en música turca",
    tags=["Predictions"],
    responses={
        200: {
            "description": "Predicción exitosa",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": 0,
                        "emotion": "Happy",
                        "confidence": 0.87,
                        "probabilities": [0.87, 0.05, 0.04, 0.04],
                        "probabilities_mapped": {
                            "Happy": 0.87,
                            "Sad": 0.05,
                            "Angry": 0.04,
                            "Relax": 0.04
                        },
                        "timestamp": "2025-11-14T10:30:45.123456"
                    }
                }
            }
        },
        422: {
            "description": "Validación fallida - Features inválidas",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "value_error",
                                "loc": ["body", "features"],
                                "msg": "Deben ser 48 características, recibidas: 10"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Error interno del servidor",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Error en el modelo de predicción"
                    }
                }
            }
        }
    }
)
async def predict(
    req: schemas.PredictRequest,
    svc: ModelService = Depends(get_model_service)
) -> schemas.PredictResponse:
    """
    Realiza predicción de emoción basada en características acústicas.
    
    **Parámetros de entrada:**
    - `features`: Array de 48 características acústicas:
      - MFCC coefficients (13)
      - Energy features (4)
      - Spectral features (8)
      - Temporal features (10)
      - Statistical features (13)
    
    **Respuesta:**
    - `prediction`: Índice de clase (0-3)
    - `emotion`: Etiqueta de emoción (Happy/Sad/Angry/Relax)
    - `confidence`: Confianza de predicción (0-1)
    - `probabilities`: Array de probabilidades por clase
    - `timestamp`: Hora de la predicción (UTC)
    
    **Ejemplo de uso (cURL):**
```bash
    curl -X POST "http://localhost:8000/api/v1/predict" \\
      -H "Content-Type: application/json" \\
      -d '{"features": [0.5, 0.3, ..., 0.7]}'
```
    
    **Ejemplo (Python):**
```python
    import requests
    
    url = "http://localhost:8000/api/v1/predict"
    payload = {
        "features": [0.5] * 48  # 48 características
    }
    response = requests.post(url, json=payload)
    print(response.json())
```
    """
    try:
        # Realizar predicción
        result = svc.predict(req.features)
        
        # Mapear resultado al schema de respuesta
        confidence = max(result["probabilities"]) if result["probabilities"] else 0.0
        
        return schemas.PredictResponse(
            prediction=result["prediction"],
            emotion=result["emotion"],
            confidence=confidence,
            probabilities=result["probabilities"],
            probabilities_mapped=result["probabilities_mapped"],
            timestamp=result["timestamp"]
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en predicción: {str(e)}"
        )


@router.get(
    "/model-info",
    summary="Información del modelo",
    tags=["Model"],
    responses={
        200: {
            "description": "Metadata del modelo cargado",
            "content": {
                "application/json": {
                    "example": {
                        "model_name": "turkish-music-emotion-rf",
                        "run_id": "3e7ecefffa2343d59a23e6d31e0ab705",
                        "artifact_path": "model/rf_raw_84pct.pkl",
                        "model_type": "RandomForestClassifier",
                        "emotion_classes": {
                            "0": "Happy",
                            "1": "Sad",
                            "2": "Angry",
                            "3": "Relax"
                        },
                        "expected_input_dim": 48,
                        "predictions_served": 42
                    }
                }
            }
        }
    }
)
async def get_model_info(
    svc: ModelService = Depends(get_model_service)
) -> Dict[str, Any]:
    """
    Retorna metadatos e información del modelo en producción.
    
    **Información incluida:**
    - Nombre y versión del modelo
    - ID del run en MLflow
    - Ruta del artifact
    - Tipo de modelo
    - Clases de emoción soportadas
    - Dimensión esperada de entrada
    - Número de predicciones servidas
    """
    try:
        return svc.get_model_info()
    except Exception as e:
        logger.exception(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error obteniendo información del modelo"
        )


@router.get(
    "/batch-predict",
    summary="Predicción por lotes (múltiples muestras)",
    tags=["Predictions"],
    deprecated=True,
    responses={
        501: {
            "description": "Endpoint no implementado aún"
        }
    }
)
async def batch_predict():
    """
    **PRÓXIMAMENTE:** Predecir múltiples muestras en una sola solicitud.
    
    Útil para análisis de múltiples canciones en batch.
    """
    raise HTTPException(
        status_code=501,
        detail="Batch prediction coming in Phase 3.3"
    )


@router.get(
    "/",
    summary="Información de la API",
    tags=["Info"],
    responses={
        200: {
            "description": "Información de la API",
            "content": {
                "application/json": {
                    "example": {
                        "name": "Turkish Music Emotion Recognition API",
                        "version": "1.0.0",
                        "description": "API para clasificar emociones en música turca",
                        "endpoints": {
                            "/health": "Health check del servicio",
                            "/predict": "Predicción de emoción",
                            "/model-info": "Información del modelo",
                            "/docs": "Documentación interactiva (Swagger UI)",
                            "/redoc": "Documentación ReDoc"
                        }
                    }
                }
            }
        }
    }
)
async def api_info() -> Dict[str, Any]:
    """
    Información general de la API.
    
    Retorna links a documentación, endpoints, etc.
    """
    return {
        "name": "Turkish Music Emotion Recognition API",
        "version": "1.0.0",
        "description": "API para clasificación de emociones en música turca (Happy, Sad, Angry, Relax)",
        "phase": "Phase 3 - Production",
        "model": "Random Forest (84.29% accuracy)",
        "endpoints": {
            "/health": "Verificar que el servicio esté operacional",
            "/predict": "Predecir emoción de música",
            "/model-info": "Información del modelo",
            "/docs": "Documentación Swagger UI",
            "/redoc": "Documentación ReDoc",
            "/openapi.json": "Schema OpenAPI"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
