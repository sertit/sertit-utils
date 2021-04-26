# -*- coding: utf-8 -*-
# Copyright 2021, SERTIT-ICube - France, https://sertit.unistra.fr/
# This file is part of sertit-utils project
#     https://github.com/sertit/sertit-utils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Script testing the logs """
import logging
import os
import sys
import tempfile

import colorlog as clog

from sertit import logs
from sertit.logs import LOGGING_FORMAT

LOGGER = logging.getLogger("Test_logger")


def test_log():
    """Testing log functions"""
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
    logger_test = logging.getLogger("test")
    file_log_lvl = logging.DEBUG
    stream_log_lvl = logging.DEBUG
    with tempfile.TemporaryDirectory() as tmp_dir:
        logs.create_logger(
            LOGGER,
            file_log_level=file_log_lvl,
            stream_log_level=stream_log_lvl,
            output_folder=tmp_dir,
            name="test_log.txt",
            other_loggers_names=["test"],
        )

        LOGGER.info("Hey you!")

        # Test create
        assert len(LOGGER.handlers) == 2  # File and stream
        assert len(logger_test.handlers) == 2  # File and stream
        for handler in LOGGER.handlers + logger_test.handlers:
            if isinstance(handler, logging.FileHandler):
                assert handler.level == file_log_lvl
                log_file = handler.baseFilename
                assert log_file is not None and os.path.isfile(log_file)
            elif isinstance(handler, logging.StreamHandler):
                assert handler.level == stream_log_lvl
            else:
                raise TypeError(
                    f"Invalid handler type: {handler.__class__}, "
                    f"the logger should only have Stream and File handlers"
                )

        logs.reset_logging()
        logs.create_logger(
            LOGGER, stream_log_level=stream_log_lvl, other_loggers_names="test"
        )

        LOGGER.info("Hi there!")

        # Test create
        assert len(LOGGER.handlers) == 1  # Only stream
        assert len(logger_test.handlers) == 1  # Only stream
        colored_fmt_cls = clog.ColoredFormatter
        for handler in LOGGER.handlers + logger_test.handlers:
            if isinstance(handler, logging.StreamHandler):
                assert handler.level == stream_log_lvl
                assert isinstance(handler.formatter, colored_fmt_cls)
            else:
                raise TypeError(
                    f"Invalid handler type: {handler.__class__}, "
                    f"the logger should only have Stream handler"
                )

        # Without color
        import colorlog

        colorlog_sys = sys.modules["colorlog"]
        del colorlog
        sys.modules["colorlog"] = None
        logs.create_logger(LOGGER, stream_log_level=stream_log_lvl)

        LOGGER.info("Urk!")

        for handler in LOGGER.handlers:
            assert isinstance(handler.formatter, logging.Formatter)

        # Just in case
        sys.modules["colorlog"] = colorlog_sys

        # Cleanup
        logs.shutdown_logger(
            LOGGER
        )  # If this fails, we will not be able to cleanup the tmp dir
