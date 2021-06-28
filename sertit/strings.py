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
""" Tools concerning strings """

import argparse
import logging
import re
from datetime import date, datetime
from typing import Union

from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def str_to_bool(bool_str: str) -> bool:
    """
    Convert a string to a bool.

    Accepted values (compared in lower case):

    - `True` <=> `yes`, `true`, `t`, `1`
    - `False` <=> `no`, `false`, `f`, `0`

    ```python
    >>> str_to_bool("yes") == True  # Works with "yes", "true", "t", "y", "1" (accepted with any letter case)
    True

    >>> str_to_bool("no") == False  # Works with "no", "false", "f", "n", "0" (accepted with any letter case)
    True
    ```

    Args:
        bool_str: Bool as a string

    Returns:
        bool: Boolean value
    """

    if isinstance(bool_str, bool):
        return bool_str

    true_str = ("yes", "true", "t", "y", "1")
    false_str = ("no", "false", "f", "n", "0")

    if bool_str.lower() in true_str:
        bool_val = True
    elif bool_str.lower() in false_str:
        bool_val = False
    else:
        raise ValueError(
            f"Invalid true or false value, "
            f"should be {true_str} if True or {false_str} if False, not {bool_str}"
        )
    return bool_val


def str_to_verbosity(verbosity_str: str) -> int:
    """
    Return a logging level from a string (compared in lower case).

    - `DEBUG`   <=> {`debug`, `d`, `10`}
    - `INFO`    <=> {`info`, `i`, `20`}
    - `WARNING` <=> {`warning`, `w`, `warn`}
    - `ERROR`   <=> {`error`, `e`, `err`}

    ```python
    >>> str_to_bool("d") == logging.DEBUG  # Works with 'debug', 'd', 10 (accepted with any letter case)
    True

    >>> str_to_bool("i") == logging.INFO  # Works with 'info', 'i', 20 (accepted with any letter case)
    True

    >>> str_to_bool("w") == logging.WARNING  # Works with 'warning', 'w', 'warn', 30 (accepted with any letter case)
    True

    >>> str_to_bool("e") == logging.ERROR  # Works with 'error', 'e', 'err', 40 (accepted with any letter case)
    True
    ```

    Args:
        verbosity_str (str): String to be converted

    Returns:
        logging level: Logging level (INFO, DEBUG, WARNING, ERROR)
    """
    debug_str = ("debug", "d", 10)
    info_str = ("info", "i", 20)
    warn_str = ("warning", "w", "warn", 30)
    err_str = ("error", "e", "err", 40)

    if isinstance(verbosity_str, str):
        verbosity_str = verbosity_str.lower()

    if verbosity_str in info_str:
        verbosity = logging.INFO
    elif verbosity_str in debug_str:
        verbosity = logging.DEBUG
    elif verbosity_str in warn_str:
        verbosity = logging.WARNING
    elif verbosity_str in err_str:
        verbosity = logging.ERROR
    else:
        raise argparse.ArgumentTypeError(
            f"Incorrect logging level value: {verbosity_str}, "
            f"should be {info_str}, {debug_str}, {warn_str} or {err_str}"
        )

    return verbosity


def str_to_list(
    list_str: Union[str, list], additional_separator: str = "", case: str = None
) -> list:
    """
    Convert str to list with `,`, `;`, ` ` separators.

    ```python
    >>> str_to_list("A, B; C D")
    ["A", "B", "C", "D"]
    ```

    Args:
        list_str (Union[str, list]): List as a string
        additional_separator (str): Additional separators. Base ones are `,`, `;`, ` `.
        case (str): {none, 'lower', 'upper'}
    Returns:
        list: A list from split string
    """
    if isinstance(list_str, str):
        # Concatenate separators
        separators = ",|;| "
        if additional_separator:
            separators += "|" + additional_separator

        # Split
        listed_str = re.split(separators, list_str)
    elif isinstance(list_str, list):
        listed_str = list_str
    else:
        raise ValueError(
            f"List should be given as a string or a list of string: {list_str}"
        )

    out_list = []
    for item in listed_str:
        # Check if there are null items
        if item:
            if case == "lower":
                item_case = item.lower()
            elif case == "upper":
                item_case = item.upper()
            else:
                item_case = item

            out_list.append(item_case)

    return out_list


def str_to_date(
    date_str: Union[str, datetime], date_format: str = DATE_FORMAT
) -> datetime:
    """
    Convert string to a `datetime.datetime`.

    Also accepted date formats:

    - "now": datetime.today()
    - Usual JSON date format: '%Y-%m-%d'
    - Already formatted datetimes and dates

    ```python
    # Default date format (isoformat)
    >>> str_to_date("2020-05-05T08:05:15")
    datetime(2020, 5, 5, 8, 5, 15)

    # This usual JSON format is also accepted
    >>> str_to_date("2019-08-06")
    datetime(2019, 8, 6)

    # User date's format
    >>> str_to_date("20200909105055", date_format="%Y%m%d%H%M%S")
    datetime(2020, 9, 9, 10, 50, 55)
    ```

    Args:
        date_str (str): Date as a string
        date_format (str): Format of the date (as ingested by strptime)

    Returns:
        datetime.datetime: A date as a python datetime object
    """
    if isinstance(date_str, datetime):
        dtm = date_str
    elif isinstance(date_str, date):
        dtm = datetime.fromisoformat(date_str.isoformat())
    else:
        try:
            if date_str.lower() == "now":
                # Now with correct format (no microseconds if not specified and so on)
                dtm = datetime.strptime(
                    datetime.today().strftime(date_format), date_format
                )
            else:
                dtm = datetime.strptime(date_str, date_format)
        except ValueError:
            # Just try with the usual JSON format
            json_date_format = "%Y-%m-%d"
            try:
                dtm = datetime.strptime(date_str, json_date_format)
            except ValueError as ex:
                raise ValueError(
                    f"Invalid date format: {date_str}; should be {date_format} "
                    f"or {json_date_format}"
                ) from ex
    return dtm


def str_to_list_of_dates(
    date_str: Union[list, str],
    date_format: str = DATE_FORMAT,
    additional_separator: str = "",
) -> list:
    """
    Convert a string containing a list of dates to a list of `datetime.datetime`.

    Also accepted date formats:

    - "now": datetime.today()
    - Usual JSON date format: '%Y-%m-%d'
    - Already formatted datetimes and dates

    ```python
    >>> # Default date format (isoformat)
    >>> str_to_list_of_dates("20200909105055, 2019-08-06;19560702121212\t2020-08-09",
    >>>                      date_format="%Y%m%d%H%M%S",
    >>>                      additional_separator="\t")
    [datetime(2020, 9, 9, 10, 50, 55), datetime(2019, 8, 6), datetime(1956, 7, 2, 12, 12, 12), datetime(2020, 8, 9)]
    ```

    Args:
        date_str (Union[list, str]): Date as a string
        date_format (str): Format of the date (as ingested by strptime)
        additional_separator (str): Additional separator

    Returns:
        list: A list containing datetimes objects
    """
    # Split string to get a list of strings
    list_of_dates_str = str_to_list(date_str, additional_separator)

    # Convert strings to date
    list_of_dates = [str_to_date(dt, date_format) for dt in list_of_dates_str]

    return list_of_dates


def to_cmd_string(unquoted_str: str) -> str:
    """
    Add quotes around the string in order to make the command understand it's a string
    (useful with tricky symbols like & or white spaces):

    ```python
    >>> # This str wont work in the terminal without quotes (because of the &)
    >>> pb_str = r"D:\Minab_4-DA&VHR\Minab_4-DA&VHR.shp"
    >>> to_cmd_string(pb_str)
    "\"D:\Minab_4-DA&VHR\Minab_4-DA&VHR.shp\""
    ```

    Args:
        unquoted_str (str): String to update

    Returns:
        str: Quoted string
    """
    if not isinstance(unquoted_str, str):
        unquoted_str = str(unquoted_str)

    cmd_str = unquoted_str
    if not unquoted_str.startswith('"'):
        cmd_str = '"' + cmd_str
    if not unquoted_str.endswith('"'):
        cmd_str = cmd_str + '"'
    return cmd_str


def snake_to_camel_case(snake_str: str) -> str:
    """
    Convert a `snake_case` string to `CamelCase`.

    ```python
    >>> snake_to_camel_case("snake_case")
    "SnakeCase"
    ```

    Args:
        snake_str (str): String formatted in snake_case

    Returns:
        str: String formatted in CamelCase
    """
    return "".join((w.capitalize() for w in snake_str.split("_")))


def camel_to_snake_case(snake_str: str) -> str:
    """
    Convert a `CamelCase` string to `snake_case`.

    ```python
    >>> camel_to_snake_case("CamelCase")
    "camel_case"
    ```

    Args:
        snake_str (str): String formatted in CamelCase

    Returns:
        str: String formatted in snake_case
    """
    return "".join(["_" + c.lower() if c.isupper() else c for c in snake_str]).lstrip(
        "_"
    )
