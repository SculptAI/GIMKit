import logging

from gimkit.log import get_logger


def test_get_logger_with_name():
    """Tests that get_logger returns a child logger when a name is provided."""
    logger = get_logger("test_child")
    assert logger.name == "test_child"
    assert isinstance(logger, logging.Logger)


def test_get_logger_without_name():
    """Tests that get_logger returns the root logger when no name is provided."""
    logger = get_logger()
    assert logger.name == "root"
    assert isinstance(logger, logging.Logger)
