from pydantic import BaseSettings

class Settings(BaseSettings):
    """
    Clase de configuración de la aplicación.

    Utiliza Pydantic para cargar y validar los valores de configuración.
    Los valores se leen de las variables de entorno o del archivo .env,
    utilizando los valores por defecto si no se encuentran.
    """
    APP_NAME: str = "mlops-fastapi"
    """Nombre de la aplicación, usado para logging o branding."""

    ENV: str = "dev"
    """Entorno de ejecución (e.g., 'dev', 'staging', 'prod')."""

    MODEL_PATH: str = "app/models/model.joblib"
    """Ruta local o absoluta al archivo del modelo serializado (joblib)."""

    # Se usa 'Optional[str]' si no se usa Python 3.10+ para 'str | None'
    MLFLOW_TRACKING_URI: str | None = None
    """URI del servidor de MLflow para el seguimiento de experimentos. None si es local."""

    LOG_LEVEL: str = "INFO"
    """Nivel de registro para el sistema de logging (e.g., 'INFO', 'DEBUG', 'WARNING')."""

    # Comentario para indicar dónde se deben agregar más configuraciones.
    # add DB, S3 creds, API keys...
    class Config:
        """Configuración interna de Pydantic."""
        env_file = ".env"
        """Especifica que Pydantic debe cargar variables desde el archivo .env."""

# Inicializa la configuración: carga los valores y los valida.
settings = Settings()
