from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import logger
from app.core.config import settings

# DO NOT RUN UNTIL THE FINAL VERSION IS RELEASED.

app = FastAPI(title=settings.APP_NAME)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting app: %s", settings.APP_NAME)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down app: %s", settings.APP_NAME)