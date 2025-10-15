from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends
from app.api import schemas
from app.services.model_service import ModelService
from app.core.logger import logger

router = APIRouter()

#DO NOT RUN UNTIL THE FINAL VERSION IS RELEASED.
def get_model_service():
    return ModelService()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/train")
async def train(req: schemas.TrainRequest, background_tasks: BackgroundTasks, svc: ModelService = Depends(get_model_service)):
    def _train():
        try:
            path = svc.train(req.data_path, req.model_name)
            logger.info("Training finished, saved: %s", path)
        except Exception as e:
            logger.exception("Training failed")
    background_tasks.add_task(_train)
    return {"status": "started", "model_name": req.model_name}

@router.post("/predict", response_model=schemas.PredictResponse)
async def predict(req: schemas.PredictRequest, svc: ModelService = Depends(get_model_service)):
    try:
        result = svc.predict(req.features)
        return schemas.PredictResponse(prediction=result["prediction"], probabilities=result["probabilities"])
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
