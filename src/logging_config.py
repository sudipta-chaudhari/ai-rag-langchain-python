"""
Logging Configuration Module for the RAG Pipeline.

This module provides centralized logging setup with support for both console
and file-based logging. It uses rotating file handlers to prevent log files
from consuming excessive disk space.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(logger_name: str = "rag_pipeline", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for the RAG pipeline.

    Console output is controlled centrally via Config.console_logging_enabled.
    File logging is always enabled.

    Args:
        logger_name (str): Name of the logger. Defaults to "rag_pipeline".
                          Use __name__ when calling from modules for proper hierarchy.
        log_level (int): Logging level for console output.

    Returns:
        logging.Logger: Configured logger instance ready for use.

    Example:
        >>> from logging_config import setup_logging
        >>> logger = setup_logging(__name__)
        >>> logger.info("Application started")
    """
    from config import Config  # Import here to avoid circular imports
    
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured to avoid duplicate handlers
    if not logger.handlers:
        logger.setLevel(log_level)

        # Create formatter for consistent log message format
        # Format: timestamp - logger_name - level - function_name:line_number - message
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler (stdout) - controlled by Config.console_logging_enabled
        config = Config()
        if config.console_logging_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler with rotation - maintains detailed logs without filling disk
        logs_folder = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(logs_folder, exist_ok=True)
        log_file = os.path.join(logs_folder, f"{logger_name}.log")

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5  # Keep 5 backup files (50 MB total)
        )
        file_handler.setLevel(logging.DEBUG)  # File captures everything
        file_handler.setFormatter(formatter)

        # Attach handlers to logger
        logger.addHandler(file_handler)

        logger.debug(f"Logger '{logger_name}' initialized successfully (console_logging_enabled={config.console_logging_enabled})")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    Use this when you need to get a logger that was previously configured
    by setup_logging() without reconfiguring it.

    Args:
        name (str): The name of the logger to retrieve.

    Returns:
        logging.Logger: The requested logger instance.

    Example:
        >>> from logging_config import get_logger
        >>> logger = get_logger("rag_pipeline.ingestion")
    """
    return logging.getLogger(name)
