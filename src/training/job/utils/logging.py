"""
Shared logging configuration for local submission and remote execution.
"""

import logging
import sys


# ----------------------------------------------------------------------------------------------------------------------
# Logger configuration

LOG_FORMAT = (
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)
HANDLER_NAME = "training.stdout"


def setup_logger() -> logging.Logger:
    """
    Configure and return the shared project logger.

    Reuse the project handler on repeated calls while preserving existing handlers.

    :return: Project logger configured to write formatted messages to standard
    output.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Set up the project logger

    # Use one named logger for messages from every pipeline module
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # ------------------------------------------------------------------------------------------------------------------
    # Install the project stream handler once

    # Add our handler once, preserving any handlers installed by callers
    # Name the installed handler so repeated setup calls remain idempotent
    if not any(handler.name == HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name(HANDLER_NAME)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
