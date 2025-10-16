import logging
import sys
from logging.config import dictConfig
from .config import settings

def setup_logging():
    """
    Configura el sistema de logging de la aplicación utilizando
    la configuración de diccionario de Python (dictConfig).

    La configuración se basa en la variable de entorno/configuración LOG_LEVEL.
    """
    # Determina el nivel de logging (DEBUG, INFO, WARNING, etc.) a partir de settings.
    # Usa logging.INFO como valor por defecto si la configuración no es válida.
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Configuración del logging mediante un diccionario (dictConfig).
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False, # Permite que otros loggers preexistentes continúen funcionando.
        "formatters": {
            # Define el formato de salida de los mensajes del log.
            "default": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
        },
        "handlers": {
            # Define un 'handler' para enviar los logs a la salida estándar (consola).
            "default": {
                "class": "logging.StreamHandler", 
                "stream": sys.stdout, # Dirige la salida a stdout (la consola).
                "formatter": "default" # Usa el formato definido arriba.
            }
        },
        "root": {
            # Configuración del logger principal (root logger).
            "level": level, # Establece el nivel de registro definido (e.g., INFO).
            "handlers": ["default"] # Asigna el handler 'default' al logger principal.
            },
    })

# Ejecuta la función para aplicar la configuración al inicio de la aplicación.
setup_logging()

# Crea una instancia de logger para este módulo, que hereda la configuración 'root'.
logger = logging.getLogger(__name__)
