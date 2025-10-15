from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import logger
from app.core.config import settings

#DO NOT RUN UNTIL THE FINAL VERSION IS RELEASED.

app = FastAPI(title=settings.APP_NAME)
app.include_router(api_router, prefix="/api/v1")

@app
async def lifespan(app: FastAPI):
    logger.info(f"Starting app {settings.APP_NAME}")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

@app.get("/")
async def root():
    return {"app": settings.APP_NAME}
