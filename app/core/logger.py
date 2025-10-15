import logging
import sys
from logging.config import dictConfig
from .config import settings

def setup_logging():
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}
        },
        "handlers": {
            "default": {"class": "logging.StreamHandler", "stream": sys.stdout, "formatter": "default"}
        },
        "root": {"level": level, "handlers": ["default"]},
    })

setup_logging()
logger = logging.getLogger(__name__)
