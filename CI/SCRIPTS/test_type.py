""" Script testing the type_utils """
import logging
from enum import unique

import pytest
from datetime import datetime

from sertit_utils.core import type_utils
from sertit_utils.core.type_utils import ListEnum


def test_conversion():
    # Str to bool
    true_str = (True, "yes", "true", "t", "y", "1")
    false_str = (False, "no", "false", "f", "n", "0")
    for true, false in zip(true_str, false_str):
        assert type_utils.str_to_bool(true)
        assert not type_utils.str_to_bool(false)

    # Str to logging verbosity
    debug_str = ('debug', 'd', 10)
    info_str = ('info', 'i', 20)
    warn_str = ('warning', 'w', 'warn', 30)
    err_str = ('error', 'e', 'err', 40)
    for debug, info, warn, err in zip(debug_str, info_str, warn_str, err_str):
        assert type_utils.str_to_verbosity(debug) == logging.DEBUG
        assert type_utils.str_to_verbosity(info) == logging.INFO
        assert type_utils.str_to_verbosity(warn) == logging.WARNING
        assert type_utils.str_to_verbosity(err) == logging.ERROR

    # Str to list of dates
    lof_dates = "20200909105055, 2019-08-06;19560702121212\t2020-08-09"
    assert (type_utils.str_to_list_of_dates(lof_dates, date_format="%Y%m%d%H%M%S", additional_separator="\t") ==
            [datetime(2020, 9, 9, 10, 50, 55), datetime(2019, 8, 6),
             datetime(1956, 7, 2, 12, 12, 12), datetime(2020, 8, 9)])


def test_list_dict():
    """ Test dict functions """
    test_list = ["A", "T", "R", "", 3, None]
    test_dict = {"A": "T", "R": 3}
    test_list = type_utils.remove_empty_values(test_list)
    assert test_list == ["A", "T", "R", 3]

    # List to dict
    assert type_utils.list_to_dict(test_list) == test_dict

    # Nested set
    res_dict = {"A": "T",
                "R": 3,
                "B": {
                    "C": {
                        "D": "value"
                    }
                }}
    type_utils.nested_set(test_dict, ["B", "C", "D"], "value")
    assert test_dict == res_dict

    # Mandatory keys
    type_utils.check_mandatory_keys(test_dict, ["A", "B"])  # True
    with pytest.raises(ValueError):
        type_utils.check_mandatory_keys(test_dict, ["C"])  # False

    # Find by key
    assert type_utils.find_by_key(test_dict, "D") == "value"


def test_str():
    """ Test string function """
    tstr = "ThisIsATest"
    assert type_utils.snake_to_camel_case(type_utils.camel_to_snake_case(tstr)) == tstr
    assert type_utils.to_cmd_string(tstr) == f'"{tstr}"'

def test_enum():
    """ Test ListEnum """
    @unique
    class Polarization(ListEnum):
        """ SAR Polarizations """
        hh = "HH"
        vv = "VV"
        vh = "VH"
        hv = "HV"

    assert Polarization.list_values() == ["HH", "VV", "VH", "HV"]
    assert Polarization.list_names() == ["hh", "vv", "vh", "hv"]
    assert Polarization.from_value("HH") == Polarization.hh
    assert Polarization.from_value(Polarization.hh) == Polarization.hh