from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import logger
from app.core.config import settings

#DO NOT RUN UNTIL THE FINAL VERSION IS RELEASED.

# Inicializa la aplicación FastAPI utilizando el nombre de la aplicación definido en Settings.
app = FastAPI(title=settings.APP_NAME)

# Incluye el router que contiene todos los endpoints de la API.
# Las rutas serán accesibles bajo el prefijo "/api/v1".
app.include_router(api_router, prefix="/api/v1")

@app
async def lifespan(app: FastAPI):
    """
    Función de gestión del ciclo de vida de la aplicación.
    Se ejecuta al iniciar la aplicación (startup) y al cerrarla (shutdown).
    """
    # Lógica de inicio de la aplicación (antes de que empiece a servir peticiones)
    logger.info(f"Starting app {settings.APP_NAME}")

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

# Define el endpoint raíz ("/") de la aplicación.
@app.get("/")
async def root():
    """Endpoint de salud (Health check) o de bienvenida."""
    return {"app": settings.APP_NAME}
