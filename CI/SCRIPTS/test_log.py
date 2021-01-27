""" Script testing the log_utils """
import os
import logging
import tempfile
from sertit_utils.core import log_utils
from sertit_utils.core.log_utils import LOGGING_FORMAT

LOGGER = logging.getLogger("Test_logger")


def test_log():
    """ Testing log functions """
    # -- INIT LOGGER --
    log_lvl = logging.WARNING
    log_utils.init_logger(LOGGER, log_lvl=log_lvl, log_format=LOGGING_FORMAT)
    log_utils.reset_logging()
    assert LOGGER.handlers == []
    assert LOGGER.filters == []
    assert LOGGER.level == 0

    # Re init logger
    log_utils.init_logger(LOGGER, log_lvl=log_lvl, log_format=LOGGING_FORMAT)

    # Test init
    assert LOGGER.level == log_lvl
    assert LOGGER.handlers[0].formatter._fmt == LOGGING_FORMAT

    # -- CREATE LOGGER --
    file_log_lvl = logging.DEBUG
    stream_log_lvl = logging.DEBUG
    tmp_dir = tempfile.TemporaryDirectory()
    log_utils.create_logger(LOGGER,
                            file_log_level=file_log_lvl,
                            stream_log_level=stream_log_lvl,
                            output_folder=tmp_dir.name,
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
            raise TypeError("Invalid handler type: {}, "
                            "the logger should only have Stream and File handlers".format(handler.__class__))

    # Cleanup
    log_utils.shutdown_logger(LOGGER)  # If this fails, we will not be able to cleanup the tmp dir
    tmp_dir.cleanup()
