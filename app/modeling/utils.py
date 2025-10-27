import logging
from logging import Logger
from pathlib import Path
from app.core.config import settings

class Utils:
    _logger: Logger = None

    @classmethod
    def get_logger(cls, name: str = __name__) -> Logger:
        if cls._logger is None:
            cls._logger = logging.getLogger(name)
            cls._logger.setLevel(logging.INFO)

            # Evitar duplicados si ya tiene handlers
            if not cls._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
        return cls._logger
