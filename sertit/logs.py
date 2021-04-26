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
""" Logging tools """
import logging
import logging.config
import os
from datetime import datetime
from typing import Union

LOGGING_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
SU_NAME = "sertit"


def init_logger(
    curr_logger: logging.Logger,
    log_lvl: int = logging.DEBUG,
    log_format: str = LOGGING_FORMAT,
) -> None:
    """
    Initialize a very basic logger to trace the first lines in the stream.

    To be done before everything (like parsing log_file etc...)

    ```python
    >>> logger = logging.getLogger("logger_test")
    >>> init_logger(logger, logging.INFO, '%(asctime)s - [%(levelname)s] - %(message)s')
    >>> logger.info("MESSAGE")
    2021-03-02 16:57:35 - [INFO] - MESSAGE
    ```

    Args:
        curr_logger (logging.Logger): Logger to be initialize
        log_lvl (int): Logging level to be set
        log_format (str): Logger format to be set
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "fmt": {
                    "format": log_format,
                }
            },
            "handlers": {
                "stream": {
                    "level": logging.getLevelName(log_lvl),
                    "class": "logging.StreamHandler",
                    "formatter": "fmt",
                },
            },
            "loggers": {
                curr_logger.name: {
                    "handlers": ["stream"],
                    "propagate": False,
                    "level": logging.getLevelName(log_lvl),
                }
            },
        }
    )


# pylint: disable=R0913
# Too many arguments (8/5) (too-many-arguments)
def create_logger(
    logger: logging.Logger,
    file_log_level: int = logging.DEBUG,
    stream_log_level: int = logging.INFO,
    output_folder: str = None,
    name: str = None,
    other_loggers_names: Union[str, list] = None,
    other_loggers_file_log_level: int = None,
    other_loggers_stream_log_level: int = None,
) -> None:
    """
    Create file and stream logger at the wanted level for the given logger.

    - If you have `colorlog` installed, it will produce colored logs.
    - If you do not give any output and name, it won't create any file logger

    It will also manage the log level of other specified logger that you give.

    ```python
    >>> logger = logging.getLogger("logger_test")
    >>> create_logger(logger, logging.DEBUG, logging.INFO, "path\\to\\log", "log.txt")
    >>> logger.info("MESSAGE")
    2021-03-02 16:57:35 - [INFO] - MESSAGE

    >>> # "logger_test" will also log DEBUG messages
    >>> # to the "path\\to\\log\\log.txt" file with the same format
    ```

    Args:
        logger (logging.Logger): Logger to create
        file_log_level (int): File log level
        stream_log_level (int): Stream log level
        output_folder (str): Output folder. Won't create File logger if not specified
        name (str): Name of the log file, prefixed with the date and suffixed with _log. Can be None.
        other_loggers_names (Union[str, list]): Other existing logger to manage (setting the right format and log level)
        other_loggers_file_log_level (int): File log level for other loggers
        other_loggers_stream_log_level (int): Stream log level for other loggers
    """
    if not isinstance(other_loggers_names, list):
        other_loggers_names = [other_loggers_names]

    # Manage other log levels
    if other_loggers_file_log_level is None:
        other_loggers_file_log_level = file_log_level
    if other_loggers_stream_log_level is None:
        other_loggers_stream_log_level = stream_log_level

    # Formatters
    basic_fmter = {
        "format": "%(asctime)s - [%(levelname)s] - %(message)s",
    }
    try:
        # 'colorlog.ColoredFormatter' imported but unused
        from colorlog import ColoredFormatter  # noqa: F401

        color_fmter = {
            "()": "colorlog.ColoredFormatter",
            "format": "%(asctime)s - [%(log_color)s%(levelname)s%(reset)s] - %(message_log_color)s%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "reset": True,
            "log_colors": {
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "cyan",
                "ERROR": "red",
                "CRITICAL": "fg_bold_red,bg_white",
            },
            "secondary_log_colors": {
                "message": {
                    "DEBUG": "white",
                    "INFO": "green",
                    "WARNING": "cyan",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                }
            },
            "style": "%",
        }
    except ModuleNotFoundError:
        logger.debug("Impossible to import colorlog, will log without colors.")
        color_fmter = basic_fmter

    # Initiate the logging configuration dictionary
    logging_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"color_fmter": color_fmter, "basic_fmter": basic_fmter},
        "handlers": {
            "stream_main": {
                "level": logging.getLevelName(stream_log_level),
                "class": "logging.StreamHandler",
                "formatter": "color_fmter",
            },
            "stream_other": {
                "level": logging.getLevelName(other_loggers_stream_log_level),
                "class": "logging.StreamHandler",
                "formatter": "color_fmter",
            },
        },
    }

    # Get logger file path
    if output_folder:
        date = datetime.today().replace(microsecond=0).strftime("%y%m%d_%H%M%S")
        log_file_name = f"{date}{f'_{name}' if name else ''}_log.txt"
        log_path = os.path.join(output_folder, log_file_name)

        logging_dict["handlers"].update(
            {
                "file_main": {
                    "level": logging.getLevelName(file_log_level),
                    "class": "logging.FileHandler",
                    "filename": log_path,
                    "formatter": "basic_fmter",
                },
                "file_other": {
                    "level": logging.getLevelName(other_loggers_file_log_level),
                    "class": "logging.FileHandler",
                    "filename": log_path,
                    "formatter": "basic_fmter",
                },
            }
        )
        handlers_main = ["stream_main", "file_main"]
        handlers_other = ["stream_other", "file_other"]
    else:
        handlers_main = ["stream_main"]
        handlers_other = ["stream_other"]

    # Add logger to the config dict
    logging_dict.update(
        {
            "loggers": {
                logger.name: {
                    "handlers": handlers_main,
                    "propagate": False,
                    "level": "DEBUG",
                }
            }
        }
    )

    # Manage other loggers
    if other_loggers_names:
        for log_name in other_loggers_names:
            logging_dict["loggers"].update(
                {
                    log_name: {
                        "handlers": handlers_other,
                        "propagate": False,
                        "level": "DEBUG",
                    }
                }
            )

    logging.config.dictConfig(logging_dict)


def shutdown_logger(logger: logging.Logger) -> None:
    """
    Shutdown logger (if you need to delete the log file for example)

    ```python
    >>> logger = logging.getLogger("logger_test")
    >>> shutdown_logger(logger)
    >>> # "logger_test" won't log anything after another init
    ```

    Args:
        logger (logging.Logger): Logger to shutdown
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.flush()
        handler.close()


def reset_logging() -> None:
    """
    Reset root logger

    .. WARNING::
        MAY BE OVERKILL**

    ```python
    >>> reset_logging()
    Reset root logger
    ```

    """
    manager = logging.root.manager
    manager.disabled = logging.NOTSET
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                # Copied from `logging.shutdown`.
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)
