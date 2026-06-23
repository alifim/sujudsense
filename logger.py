import os
import logging

def setup_logger(name: str = "SujudSense") -> logging.Logger:
    """
    Configures and returns a standardized logger instance.
    Log level is controlled via the LOG_LEVEL environment variable.
    """
    # Resolve log level dynamically (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, env_level, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times in hot-reloads
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Singleton instance for application-wide import
logger = setup_logger()