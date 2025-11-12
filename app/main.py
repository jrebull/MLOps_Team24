from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting app %s", settings.APP_NAME)
    yield
    logger.info("Shutting down app %s", settings.APP_NAME)

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"app": settings.APP_NAME}
