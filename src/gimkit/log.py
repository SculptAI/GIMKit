import logging

from logging.config import dictConfig
from sys import stdout


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "no_datetime": {
            "format": "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": logging.DEBUG,
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "no_datetime",
        },
    },
    "root": {
        "level": logging.WARNING,
        "handlers": ["console"],
    },
}


def get_logger(name: str | None = None) -> logging.Logger:
    """returns the project logger, scoped to a child name if provided
    Args:
        name: will define a child logger
    """
    dictConfig(LOGGING_CONFIG)

    parent_logger = logging.getLogger("")
    if name:
        return parent_logger.getChild(name)
    return parent_logger
