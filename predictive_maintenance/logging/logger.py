"""
logger.py
---------
Reusable logging module for the entire Python project.

Usage Example:
    from logger import get_logger

    log = get_logger(__name__)
    log.info("This is an info message")
"""

import logging  # Python's built-in logging library
from logging.handlers import (
    RotatingFileHandler,
)  # For rotating log files (prevents huge single log)
import os  # For creating directories and file paths
from datetime import datetime  # For timestamping log files if needed

# ============================
# 1. Logging Configuration Constants
# ============================

LOG_DIR = "logs"  # Folder where log files will be stored
LOG_FILE_NAME = (
    f"{datetime.now().strftime('%d_%m_%Y_%H_%M')}.log"  # Name of the main log file
)
LOG_LEVEL = logging.DEBUG  # Global log level: capture all messages from DEBUG and up
MAX_LOG_SIZE = 5_000_000  # 5 MB max size before rotating the log file
BACKUP_COUNT = 3  # Keep up to 3 old rotated log files (e.g., app.log.1, app.log.2)

# ============================
# 2. Ensure Log Directory Exists
# ============================

# Create the logs folder if it doesn't exist already
os.makedirs(LOG_DIR, exist_ok=True)

# Build the full file path for the log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# ============================
# 3. Create Formatter
# ============================

# The formatter controls how each log message looks.
# Example output:
# 2025-10-07 15:22:11,345 - my_module - INFO - This is a message
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# ============================
# 4. Create File Handler (Rotating)
# ============================

# This handler writes log messages to a file on disk.
# RotatingFileHandler automatically "rolls over" the log file
# once it reaches MAX_LOG_SIZE, and keeps BACKUP_COUNT older files.
file_handler = RotatingFileHandler(
    LOG_FILE_PATH,  # Where to store the log file
    maxBytes=MAX_LOG_SIZE,  # Max size before rotation
    backupCount=BACKUP_COUNT,  # Number of backup files to keep
    encoding="utf-8",  # Handle Unicode properly in log messages
)

# Set the minimum log level for the file handler.
# Only INFO and above will be written to the file.
file_handler.setLevel(logging.INFO)

# Apply the formatter to control how the file logs look.
file_handler.setFormatter(formatter)

# ============================
# 5. Create Console Handler
# ============================

# This handler outputs logs to the terminal (stdout)
# Useful for seeing logs in real time during development.
console_handler = logging.StreamHandler()

# Set the minimum log level for console output.
# DEBUG and above messages will appear in the terminal.
console_handler.setLevel(logging.DEBUG)

# Apply the same formatter to console output for consistency.
console_handler.setFormatter(formatter)

# ============================
# 6. Logger Factory Function
# ============================


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Args:
        name (str): Typically pass __name__ (the current module name)
                    so that each logger is clearly identified in log output.

    Returns:
        logging.Logger: A logger instance with file & console handlers attached.
    """
    # Create or retrieve a logger with the given name.
    logger = logging.getLogger(name)

    # Set the global log level for this logger.
    # This determines which messages are processed at all.
    logger.setLevel(LOG_LEVEL)

    # Avoid adding multiple handlers to the same logger
    # This can happen if get_logger() is called multiple times in the same module.
    if not logger.hasHandlers():
        # Attach both file and console handlers to this logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Return the fully configured logger to the caller
    return logger
