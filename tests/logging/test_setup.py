from ml_project_template.logging import setup_logging # type: ignore
from _pytest.logging import LogCaptureFixture
import logging

def test_setup_logging(caplog: LogCaptureFixture):
    setup_logging(
    level = logging.DEBUG,
    json_logs = True,
    output_file = "tests/logging/test_logs.json",
)
    logger = logging.getLogger("test_logger")

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    assert len(caplog.messages) == 5