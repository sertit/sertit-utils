""" Script testing the log_utils """
import os
import logging
import tempfile
from sertit import logs
from sertit.logs import LOGGING_FORMAT

LOGGER = logging.getLogger("Test_logger")


def test_log():
    """ Testing log functions """
    # -- INIT LOGGER --
    log_lvl = logging.WARNING
    LOGGER.handlers.append(logging.StreamHandler())  # Handler to remove
    logs.init_logger(LOGGER, log_lvl=log_lvl, log_format=LOGGING_FORMAT)
    logs.reset_logging()
    assert LOGGER.handlers == []
    assert LOGGER.filters == []
    assert LOGGER.level == 0

    # Re init logger
    logs.init_logger(LOGGER, log_lvl=log_lvl, log_format=LOGGING_FORMAT)

    # Test init
    assert LOGGER.level == log_lvl
    assert LOGGER.handlers[0].formatter._fmt == LOGGING_FORMAT

    # -- CREATE LOGGER --
    file_log_lvl = logging.DEBUG
    stream_log_lvl = logging.DEBUG
    with tempfile.TemporaryDirectory() as tmp_dir:
        logs.create_logger(LOGGER,
                           file_log_level=file_log_lvl,
                           stream_log_level=stream_log_lvl,
                           output_folder=tmp_dir,
                           name="test_log.txt",
                           other_logger_names=["test"])

        # Test create
        assert len(LOGGER.handlers) == 2  # File and stream
        for handler in LOGGER.handlers:
            if isinstance(handler, logging.FileHandler):
                assert handler.level == file_log_lvl
                log_file = handler.baseFilename
                assert log_file is not None and os.path.isfile(log_file)
            elif isinstance(handler, logging.StreamHandler):
                assert handler.level == stream_log_lvl
            else:
                raise TypeError(f"Invalid handler type: {handler.__class__}, "
                                f"the logger should only have Stream and File handlers")

        # Cleanup
        logs.shutdown_logger(LOGGER)  # If this fails, we will not be able to cleanup the tmp dir
