""" Tools concerning type conversion, helpers for enums, dict and strings """
import argparse
import logging
from datetime import datetime, date
from enum import Enum, unique
import pprint
import re
from typing import Any
from sertit_utils.core.log_utils import SU_NAME

LOGGER = logging.getLogger(SU_NAME)
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


@unique
class ListEnum(Enum):
    """ List Enum (enum with function listing names and values) """

    @classmethod
    def list_values(cls) -> list:
        """
        Get the value list of this enum

        ```python
        @unique
        class TsxPolarization(ListEnum):
            SINGLE = "S"  # Single
            DUAL = "D"  # Dual
            QUAD = "Q"  # Quad
            TWIN = "T"  # Twin

        TsxPolarization.list_values()
        # >> ["S", "D", "Q", "T"]
        ```

        """
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls) -> list:
        """
        Get the name list of this enum:

        ```python
        @unique
        class TsxPolarization(ListEnum):
            SINGLE = "S"  # Single
            DUAL = "D"  # Dual
            QUAD = "Q"  # Quad
            TWIN = "T"  # Twin

        TsxPolarization.list_values()
        # >> ["SINGLE", "DUAL", "QUAD", "TWIN"]
        ```
        """
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_value(cls, val: Any) -> 'ListEnum':
        """
        Get the enum class from its value:

        ```python
        @unique
        class TsxPolarization(ListEnum):
            SINGLE = "S"  # Single
            DUAL = "D"  # Dual
            QUAD = "Q"  # Quad
            TWIN = "T"  # Twin

        TsxPolarization.from_value("Q")
        # >> <TsxPolarization.QUAD: 'Q'>
        ```

        Args:
            val (Any): Value of the Enum

        Returns:
            ListEnum: Enum with value
        """
        if isinstance(val, cls):
            val = val.value
        try:
            return next(enum for enum in cls if enum.value == val)
        except StopIteration as ex:
            raise ValueError(f"Non existing {val} in {cls.list_values()}") from ex


def str_to_bool(bool_str):
    """
    Convert a string to a bool.

    Accepted values (compared in lower case):

    - `True` <=> `yes`, `true`, `t`, `1`
    - `False` <=> `no`, `false`, `f`, `0`

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
        raise ValueError(f"Invalid true or false value, "
                         f"should be {true_str} if True or {false_str} if False, not {bool_str}")
    return bool_val


def str_to_verbosity(verbosity_str):
    """
    Return a logging level from a string (compared in lower case).

    - `DEBUG`   <=> {`debug`, `d`, `10`}
    - `INFO`    <=> {`info`, `i`, `20`}
    - `WARNING` <=> {`warning`, `w`, `warn`}
    - `ERROR`   <=> {`error`, `e`, `err`}

    Args:
        verbosity_str (str): String to be converted

    Returns:
        logging level: Logging level (INFO, DEBUG, WARNING, ERROR)
    """
    debug_str = ('debug', 'd', 10)
    info_str = ('info', 'i', 20)
    warn_str = ('warning', 'w', 'warn', 30)
    err_str = ('error', 'e', 'err', 40)

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
        raise argparse.ArgumentTypeError(f'Incorrect logging level value: {verbosity_str}, '
                                         f'should be {info_str}, {debug_str}, {warn_str} or {err_str}')

    return verbosity


def str_to_list(list_str, additional_separator='', letter_case=None, keyword_list=None):
    """
    Convert str to list with `,`, `;`, ` ` separators.

    Args:
        list_str (str): List as a string
        additional_separator (str): Additional separators. Base ones are `,`, `;`, ` `.
        letter_case (str): {none, 'lower', 'upper'}
        keyword_list (list): List of keywords that can appear
    Returns:
        list: A list from split string
    """
    if isinstance(list_str, str):
        # Concatenate separators
        separators = ',|;| '
        if additional_separator:
            separators += '|' + additional_separator

        # Split
        listed_str = re.split(separators, list_str)
    elif isinstance(list_str, list):
        listed_str = list_str
    else:
        raise ValueError(f"List should be given as a string or a list of string: {list_str}")

    out_list = []
    for item in listed_str:
        # Check if there are null items
        if item:
            if letter_case == 'lower':
                item_case = item.lower()
            elif letter_case == 'upper':
                item_case = item.upper()
            else:
                item_case = item

            if keyword_list is not None and item_case not in keyword_list:
                raise ValueError(f'Wrong keyword ({item_case}), should be chosen among {keyword_list}')

            out_list.append(item_case)

    return out_list


def str_to_date(date_str: str, date_format: str = DATE_FORMAT) -> datetime:
    """
    Convert string to a `datetime.datetime`:

    ```python
    # Default date fmt = "%Y-%m-%dT%H:%M:%S"
    date_str = "2020-05-05T08:05:15"
    str_to_date(date_str)
    # >> datetime(2020, 5, 5, 8, 5, 15)
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
                dtm = datetime.strptime(datetime.today().strftime(date_format), date_format)
            else:
                dtm = datetime.strptime(date_str, date_format)
        except ValueError:
            # Just try with the usual JSON format
            json_date_format = '%Y-%m-%d'
            try:
                dtm = datetime.strptime(date_str, json_date_format)
            except ValueError as ex:
                raise ValueError(f"Invalid date format: {date_str}; should be {date_format} "
                                 f"or {json_date_format}") from ex
    return dtm


def str_to_list_of_dates(date_str: str, date_format: str = DATE_FORMAT, additional_separator: str = '') -> list:
    """
    Convert a string containing a list of dates to a list of `datetime.datetime`.


    ```python
    # Default date fmt = "%Y-%m-%dT%H:%M:%S"
    date_str = "2020-05-05T08:05:15, 2020-05-05T08:05:15; 2020-05-05T08:05:15"
    str_to_date(date_str)
    # >> [datetime(2020, 5, 5, 8, 5, 15), datetime(2020, 5, 5, 8, 5, 15), datetime(2020, 5, 5, 8, 5, 15)]
    ```

    Args:
        date_str (str): Date as a string
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


def list_to_dict(dict_list: list) -> dict:
    """
    Return a dictionary from a list `[key, value, key_2, value_2...]`

    ```python
    list = ["A","T", "R", 3]
    list_to_dict(list)
    # >> {"A": "T", "R": 3}
    ```

    Args:
        dict_list (list[str]): Dictionary as a list

    Returns:
        dict: Dictionary
    """
    dictionary = {dict_list[i]: dict_list[i + 1] for i in range(0, len(dict_list), 2)}
    return dictionary


def nested_set(dic: dict, keys: list, value: Any):
    """
    Set value in nested directory:

    ```python
    dict = {"A": "T", "R": 3}
    nested_set(test_dict, keys=["B", "C", "D"], value="value")
    # >>
    # {
    # "A": "T",
    # "R": 3,
    # "B": {
    #      "C": {
    #           "D": "value"
    #           }
    #      }
    # }
    ```

    Args:
        dic (dict): Dictionary
        keys (list[str]): Keys as a list
        value: Value to be set
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def check_mandatory_keys(data_dict: dict, mandatory_keys: list) -> None:
    """
    Check all mandatory argument in a dictionary.
    Raise an exception if a mandatory argument is missing.

    **Note**: nested keys do not work here !

    ```python
    dict = {"A": "T", "R": 3}
    check_mandatory_keys(dict, ["A", "R"])  # Returns nothing, is OK
    check_mandatory_keys(dict, ["C"])  # Raises Value Error, is NOK
    ```

    Args:
        data_dict (dict): Data dictionary to be checked
        mandatory_keys (list[str]): List of mandatory keys
    """

    for mandatory_key in mandatory_keys:
        if mandatory_key not in data_dict:
            raise ValueError(f"Missing mandatory key '{mandatory_key}' among {pprint.pformat(data_dict)}")


def remove_empty_values(list_with_empty_values: list) -> list:
    """
    Remove empty values from list:

    ```python
    list = ["A", "T", "R", "", 3, None]
    list_to_dict(list)
    # >> ["A", "T", "R", 3]
    ```

    Args:
        list_with_empty_values (list): List with empty values
    Returns:
        list: Curated list
    """

    return list(filter(None, list_with_empty_values))


def find_by_key(data: dict, target: str) -> Any:
    """
    Find a value by key in a dictionary.

    ```python
    dict = {
            "A": "T",
            "R": 3,
            "B": {
                "C": {
                    "D": "value"
                     }
                 }
           }
    find_by_key(dict, "D")
    # >> "value"
    ```

    Args:
        data (dict): Dict to walk through
        target (str): target key

    Returns:
        Any: Value data[...][target]

    """
    val = None
    for key, value in data.items():
        if isinstance(value, dict):
            val = find_by_key(value, target)
            if val:
                break
        elif key == target:
            val = value
    return val


def to_cmd_string(unquoted_str: str) -> str:
    """
    Add quotes around the string in order to make the command understand it's a string
    (useful with tricky symbols like & or white spaces):

    ```python
    # This str wont work in the terminal without quotes (because of the &)
    pb_str = r"D:\Minab_4-DA&VHR\Minab_4-DA&VHR.shp"
    to_cmd_string(pb_str)
    # >> "\"D:\Minab_4-DA&VHR\Minab_4-DA&VHR.shp\""
    ```

    Args:
        unquoted_str (str): String to update

    Returns:
        str: Quoted string
    """
    cmd_str = unquoted_str
    if not unquoted_str.startswith('"'):
        cmd_str = '"' + cmd_str
    if not unquoted_str.endswith('"'):
        cmd_str = cmd_str + '"'
    return cmd_str


def snake_to_camel_case(snake_str: str) -> str:
    """
    Convert a `snake_case` string to `CamelCase`.

    Args:
        snake_str (str): String formatted in snake_case

    Returns:
        str: String formatted in CamelCase
    """
    return ''.join((w.capitalize() for w in snake_str.split('_')))


def camel_to_snake_case(snake_str: str) -> str:
    """
    Convert a `CamelCase` string to `snake_case`.

    Args:
        snake_str (str): String formatted in CamelCase

    Returns:
        str: String formatted in snake_case
    """
    return ''.join(['_' + c.lower() if c.isupper() else c for c in snake_str]).lstrip('_')
