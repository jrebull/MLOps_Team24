"""
FastAPI Application - Turkish Music Emotion Recognition API
Phase 3: Production Model Serving
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api.endpoints import router as api_router
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger("main")

# ========================
# LIFESPAN EVENTS
# ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eventos de ciclo de vida de la aplicación.
    
    - startup: Cargar modelo y verificar conexiones
    - shutdown: Limpiar recursos
    """
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Starting Turkish Music Emotion Recognition API")
    logger.info(f"   App: {settings.APP_NAME}")
    logger.info(f"   Version: 1.0.0")
    logger.info(f"   Environment: {settings.ENV}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("🛑 Shutting down API")
    logger.info("=" * 60)


# ========================
# FASTAPI APP
# ========================
app = FastAPI(
    title="Turkish Music Emotion Recognition API",
    description="""
    🎵 **API para Clasificación de Emociones en Música Turca**
    
    **Fase 3 - Production Model Serving**
    
    Este API expone un modelo Random Forest (84.29% accuracy) para clasificar
    emociones en características acústicas de música turca.
    
    ## 📊 Emociones Soportadas
    - **Happy** (Feliz) - Índice 0
    - **Sad** (Triste) - Índice 1
    - **Angry** (Enojado) - Índice 2
    - **Relax** (Relajante) - Índice 3
    
    ## 🎯 Características de Entrada
    - **48 características acústicas** por muestra:
      - 13 MFCC coefficients
      - 4 Energy features
      - 8 Spectral features
      - 10 Temporal features
      - 13 Statistical features
    
    ## 📍 Endpoints Principales
    - `POST /api/v1/predict` - Predecir emoción
    - `GET /api/v1/health` - Health check
    - `GET /api/v1/model-info` - Información del modelo
    
    ## 🔧 MLflow Integration
    - Modelo: `mlruns/512019673449096809/3e7ecefffa2343d59a23e6d31e0ab705/artifacts/model/`
    - Accuracy: 84.29%
    - Run ID: `3e7ecefffa2343d59a23e6d31e0ab705`
    
    ## 📚 Documentación
    - Swagger UI: `/docs`
    - ReDoc: `/redoc`
    - OpenAPI Schema: `/openapi.json`
    
    **Team:** MLOps Team 24 - Tecnológico de Monterrey
    """,
    version="1.0.0",
    contact={
        "name": "MLOps Team 24",
        "url": "https://github.com/haowei-team/MLOps_Team24",
        "email": "team24@itesm.mx"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ========================
# MIDDLEWARE
# ========================
# CORS - Permitir requests desde múltiples orígenes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# ROUTERS
# ========================
app.include_router(api_router, prefix="/api/v1", tags=["Turkish Music Emotion"])


# ========================
# CUSTOM OPENAPI SCHEMA
# ========================
def custom_openapi():
    """Personalizar schema OpenAPI con información adicional"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Turkish Music Emotion Recognition API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
        contact=app.contact,
        license_info=app.license_info,
    )
    
    # Agregar información de servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8001", "description": "Local Development"},
        {"url": "https://api.example.com", "description": "Production (cuando esté disponible)"}
    ]
    
    # Agregar tags con descripción
    openapi_schema["tags"] = [
        {
            "name": "Predictions",
            "description": "Endpoints para realizar predicciones de emociones"
        },
        {
            "name": "System",
            "description": "Endpoints de sistema (health checks, info)"
        },
        {
            "name": "Model",
            "description": "Información del modelo y metadatos"
        },
        {
            "name": "Info",
            "description": "Información general de la API"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# ========================
# ROOT ENDPOINT
# ========================
@app.get("/", tags=["Info"])
async def root():
    """Endpoint raíz - información general"""
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "status": "ok",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
