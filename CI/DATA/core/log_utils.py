""" Log sertit_utils """
import os
import logging
from datetime import datetime
from colorlog import ColoredFormatter

LOGGING_FORMAT = '%(asctime)s - [%(levelname)s] - %(message)s'
LOGGER = logging.getLogger('sertit_utils')


def init_logger(curr_logger, log_lvl=logging.DEBUG, log_format=LOGGING_FORMAT):
    """
    Initialize a very basic logger to trace the first lines in the stream.

    To be done before everything (like parsing log_file etc...)

    Args:
        curr_logger (logging.Logger): Logger to be initialize
        log_lvl (int): Logging level to be set
        log_format (str): Logger format to be set
    """
    curr_logger.setLevel(log_lvl)
    formatter = logging.Formatter(log_format)

    # Set stream handler
    if curr_logger.handlers:
        curr_logger.handlers[0].setLevel(log_lvl)
        curr_logger.handlers[0].setFormatter(formatter)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(log_lvl)
        stream_handler.setFormatter(formatter)
        curr_logger.addHandler(stream_handler)


def create_logger(file_log_level: int,
                  stream_log_level: int,
                  output_folder: str,
                  name: str,
                  other_logger_names: list) -> None:
    """
    Create file and stream logger with colored logs.

    Args:
        file_log_level (int): File log level
        stream_log_level (int): Stream log level
        output_folder (str): Output folder
        name (str): Name of the log
        other_logger_names (list): Other existing logger to manage (setting the right format and log level)
    """
    # Logger data
    date = datetime.today().replace(microsecond=0).strftime("%y%m%d_%H%M%S")
    log_file_name = "{}_{}_log.txt".format(date, name)

    # Remove already created console handler
    if LOGGER.handlers:
        for handler in LOGGER.handlers:
            LOGGER.removeHandler(handler)

    # Add stream handler
    color_fmt = ColoredFormatter(
        "%(asctime)s - [%(log_color)s%(levelname)s%(reset)s] - %(message_log_color)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'cyan',
            'ERROR': 'red',
            'CRITICAL': 'fg_bold_red,bg_white',
        },
        secondary_log_colors={
            'message': {
                'DEBUG': 'white',
                'INFO': 'green',
                'WARNING': 'cyan',
                'ERROR': 'red',
                'CRITICAL': 'bold_red'
            }
        },
        style='%'
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_log_level)
    stream_handler.setFormatter(color_fmt)
    LOGGER.addHandler(stream_handler)

    # Get logger file output path
    log_path = os.path.join(output_folder, log_file_name)

    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    LOGGER.addHandler(file_handler)

    # Manage other loggers
    for log_name in other_logger_names:
        other_logger = logging.getLogger(log_name)

        # Remove all handlers (brand new logger)
        other_logger.handlers = []

        # Set the right format anf log level
        other_logger.setLevel(logging.INFO)
        other_logger.addHandler(stream_handler)
        other_logger.addHandler(file_handler)