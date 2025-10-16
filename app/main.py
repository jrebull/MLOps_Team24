from fastapi import FastAPI
from app.api.endpoints import router as api_router
from app.core.logger import logger
from app.core.config import settings

# 1. Creación de la instancia de la aplicación
# Inicializa la aplicación FastAPI. El título se toma de la configuración.
app = FastAPI(title=settings.APP_NAME)

# 2. Inclusión del Router API
# Monta todas las rutas definidas en el módulo endpoints.
# Todos los endpoints (e.g., /predict, /train) serán accesibles bajo /api/v1/...
app.include_router(api_router, prefix="/api/v1")

# 3. Gestión del Ciclo de Vida (Lifespan)
# Define una función que se ejecuta en eventos de inicio y/o cierre.
# Al usar lifespan("startup"), la función se ejecuta solo al iniciar.
@app.lifespan("startup")
async def startup_event():
    """
    Función que se ejecuta cuando la aplicación se inicia.
    Aquí se realiza cualquier inicialización necesaria:
    - Conexión a bases de datos.
    - Carga de modelos grandes en memoria.
    - Comprobaciones de recursos.
    """
    # Registra un mensaje para confirmar que la aplicación ha iniciado correctamente.
    logger.info("Starting app %s", settings.APP_NAME)

# 4. Endpoint Raíz (Health Check / Bienvenida)
@app.get("/")
async def root():
    """
    Endpoint principal/raíz.
    Se utiliza típicamente como un chequeo de salud básico o como una página de bienvenida.
    """
    return {"app": settings.APP_NAME}
