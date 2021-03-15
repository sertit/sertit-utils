""" Script testing the type_utils """
import argparse
import logging
from datetime import datetime, date

import pytest

from sertit import strings


def test_conversion():
    # Str to bool
    true_str = (True, "yes", "true", "t", "y", "1")
    false_str = (False, "no", "false", "f", "n", "0")
    for true, false in zip(true_str, false_str):
        assert strings.str_to_bool(true)
        assert not strings.str_to_bool(false)

    # Str to logging verbosity
    debug_str = ('debug', 'd', 10)
    info_str = ('info', 'i', 20)
    warn_str = ('warning', 'w', 'warn', 30)
    err_str = ('error', 'e', 'err', 40)
    for debug, info, warn, err in zip(debug_str, info_str, warn_str, err_str):
        assert strings.str_to_verbosity(debug) == logging.DEBUG
        assert strings.str_to_verbosity(info) == logging.INFO
        assert strings.str_to_verbosity(warn) == logging.WARNING
        assert strings.str_to_verbosity(err) == logging.ERROR

    with pytest.raises(argparse.ArgumentTypeError):
        strings.str_to_verbosity("oopsie")

    # Str to list of dates
    list_of_str_dates = "20200909105055, 2019-08-06;19560702121212\t2020-08-09"
    list_of_datetimes = [datetime(2020, 9, 9, 10, 50, 55), datetime(2019, 8, 6),
                         datetime(1956, 7, 2, 12, 12, 12), datetime(2020, 8, 9)]
    list_of_dates = [datetime(2020, 9, 9, 10, 50, 55), date(2019, 8, 6),
                     datetime(1956, 7, 2, 12, 12, 12), date(2020, 8, 9)]
    assert (strings.str_to_list_of_dates(list_of_str_dates,
                                         date_format="%Y%m%d%H%M%S",
                                         additional_separator="\t") == list_of_datetimes)
    assert (strings.str_to_list_of_dates(list_of_dates) == list_of_datetimes)
    assert (strings.str_to_list_of_dates(list_of_datetimes) == list_of_datetimes)

    # Str to list
    list_str_up = ["A", "B"]
    list_str_low = ["a", "b"]
    assert strings.str_to_list(list_str_up) == list_str_up
    assert strings.str_to_list(list_str_low, case="upper") == list_str_up
    assert strings.str_to_list(list_str_up, case="lower") == list_str_low


def test_str():
    """ Test string function """
    tstr = "ThisIsATest"
    assert strings.snake_to_camel_case(strings.camel_to_snake_case(tstr)) == tstr
    assert strings.to_cmd_string(tstr) == f'"{tstr}"'
