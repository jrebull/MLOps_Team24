from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger("main") 

app = FastAPI(title=settings.APP_NAME)
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting app %s", settings.APP_NAME)

@app.get("/")
async def root():
    return {"app": settings.APP_NAME}
