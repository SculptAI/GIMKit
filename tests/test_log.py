import logging

from gimkit.log import get_logger


def test_get_logger_with_name():
    logger = get_logger("test_child")
    assert logger.name == "test_child"
    assert isinstance(logger, logging.Logger)


def test_get_logger_without_name():
    logger = get_logger()
    assert logger.name == "root"
    assert isinstance(logger, logging.Logger)
