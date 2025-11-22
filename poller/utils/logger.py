import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name, level="INFO", file_path=None):
    logger = logging.getLogger(name)
    if isinstance(level, str):
        level_value = getattr(logging, level.upper(), logging.INFO)
    else:
        level_value = level
    logger.setLevel(level_value)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)
        if file_path:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            handler = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=3)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger