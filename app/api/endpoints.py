from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Any

from app.api import schemas
from app.services.model_service import ModelService
from app.core.logger import logger

router = APIRouter()

def get_model_service() -> ModelService:
    return ModelService()

@router.get("/health")
async def health() -> JSONResponse:
    """
    Endpoint de health check.
    """
    return JSONResponse(content={"status": "ok"})

@router.post("/train")
async def train(
    req: schemas.TrainRequest,
    background_tasks: BackgroundTasks,
    svc: ModelService = Depends(get_model_service)
) -> JSONResponse:
    """
    Endpoint para iniciar entrenamiento en background.
    """

    def _train() -> None:
        try:
            path = svc.train(req.data_path, req.model_name)
            logger.info("Training finished, model saved at: %s", path)
        except Exception as e:
            logger.exception("Training failed: %s", e)

    background_tasks.add_task(_train)
    logger.info("Training started for model: %s", req.model_name)
    return JSONResponse(content={"status": "started", "model_name": req.model_name})


@router.post("/predict", response_model=schemas.PredictResponse)
async def predict(
    req: schemas.PredictRequest,
    svc: ModelService = Depends(get_model_service)
) -> Any:
    """
    Endpoint para realizar predicciones con un modelo entrenado.
    """
    try:
        result = svc.predict(req.features)

        # Validación de estructura del resultado
        if "prediction" not in result or "probabilities" not in result:
            logger.error("Predict result missing keys: %s", result)
            raise HTTPException(status_code=500, detail="Invalid prediction result structure")

        return schemas.PredictResponse(
            prediction=result["prediction"],
            probabilities=result["probabilities"]
        )
    except HTTPException:
        # Re-lanza HTTPExceptions sin modificar
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models(svc: ModelService = Depends(get_model_service)) -> JSONResponse:
    """
    Retorna una lista de modelos entrenados disponibles.
    """
    try:
        models = svc.list_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.exception("Listing models failed")
        raise HTTPException(status_code=500, detail=str(e))
