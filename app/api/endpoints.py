from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from app.api import schemas
from app.services.model_service import ModelService
from app.core.logger import logger

# Inicializa un router de la API para organizar los endpoints.
router = APIRouter()

#DO NOT RUN UNTIL THE FINAL VERSION IS RELEASED.
def get_model_service():
    """
    Función de dependencia de FastAPI que inicializa y devuelve una instancia
    del ModelService.
    """
    return ModelService()

@router.get("/health")
async def health():
    """
    Endpoint de verificación de salud de la API.
    
    Returns:
        dict: Estado simple de que el servicio está activo.
    """
    return {"status": "ok"}

@router.post("/train")
async def train(
    req: schemas.TrainRequest, 
    background_tasks: BackgroundTasks, 
    svc: ModelService = Depends(get_model_service)
):
    """
    Endpoint para iniciar el entrenamiento del modelo.
    
    El entrenamiento se ejecuta como una tarea en segundo plano (BackgroundTasks) 
    para no bloquear la respuesta HTTP.
    
    Args:
        req (schemas.TrainRequest): Petición con la ruta del dataset y el nombre del modelo.
        background_tasks (BackgroundTasks): Objeto inyectado por FastAPI para manejar tareas asíncronas.
        svc (ModelService): Instancia del servicio de modelo inyectada vía Depends.
        
    Returns:
        dict: Confirmación de que el proceso de entrenamiento ha iniciado.
    """
    def _train():
        """Función interna que contiene la lógica de entrenamiento a ejecutar en segundo plano."""
        try:
            # Llama al método de entrenamiento del servicio de modelo.
            path = svc.train(req.data_path, req.model_name)
            logger.info("Training finished, saved: %s", path)
        except Exception as e:
            # Registra cualquier error que ocurra durante el entrenamiento.
            logger.exception("Training failed")

    # Agrega la función de entrenamiento a las tareas de fondo.
    background_tasks.add_task(_train)
    return {"status": "started", "model_name": req.model_name}

@router.post("/predict", response_model=schemas.PredictResponse)
async def predict(
    req: schemas.PredictRequest, 
    svc: ModelService = Depends(get_model_service)
):
    """
    Endpoint para realizar predicciones utilizando el modelo cargado.
    
    Args:
        req (schemas.PredictRequest): Petición con la lista de características para predecir.
        svc (ModelService): Instancia del servicio de modelo inyectada vía Depends.
        
    Raises:
        HTTPException: Si la predicción falla, se retorna un error 500.
        
    Returns:
        schemas.PredictResponse: La predicción (clase) y las probabilidades (opcional).
    """
    try:
        # Llama al método de predicción del servicio de modelo.
        result = svc.predict(req.features)
        # Retorna la respuesta validada por el esquema PredictResponse.
        return schemas.PredictResponse(prediction=result["prediction"], probabilities=result["probabilities"])
    except Exception as e:
        logger.exception("Prediction failed")
        # Levanta un error HTTP 500 con el detalle de la excepción.
        raise HTTPException(status_code=500, detail=str(e))
