import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("mlops")
logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler("logs/app.log", maxBytes=5_000_000, backupCount=3)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
