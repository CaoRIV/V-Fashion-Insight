import logging

from v_fashion_insight.common.logging import configure_logging, get_logger


def test_get_logger_does_not_configure_root_logger() -> None:
    root_logger = logging.getLogger()
    handlers_before = tuple(root_logger.handlers)

    logger = get_logger("test")

    assert logger.name == "v_fashion_insight.test"
    assert tuple(root_logger.handlers) == handlers_before


def test_configure_logging_is_idempotent_and_scoped() -> None:
    root_logger = logging.getLogger()
    handlers_before = tuple(root_logger.handlers)
    logger_name = "v_fashion_insight.test_configured"

    logger = configure_logging(logging.DEBUG, logger_name)
    configured_again = configure_logging(logging.INFO, logger_name)

    assert configured_again is logger
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert len(logger.handlers) == 1
    assert tuple(root_logger.handlers) == handlers_before

    logger.handlers.clear()
