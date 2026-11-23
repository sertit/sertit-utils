# Copyright 2026, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""Miscellaneous Tools"""

import logging
import os
import pprint
import re
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum, unique
from functools import wraps
from typing import Any

from packaging.version import Version

from sertit import AnyPath
from sertit.logs import SU_NAME
from sertit.types import AnyPathStrType

LOGGER = logging.getLogger(SU_NAME)


@unique
class ListEnum(Enum):
    """
    List Enum (enum with function listing names and values)

    Example:
        >>> @unique
        ... class TsxPolarization(ListEnum):
        ...     SINGLE = "S"  # Single
        ...     DUAL = "D"  # Dual
        ...     QUAD = "Q"  # Quad
        ...     TWIN = "T"  # Twin
    """

    @classmethod
    def list_values(cls) -> list:
        """
        Get the value list of this enum

        Example:
            >>> TsxPolarization.list_values()
            ["S", "D", "Q", "T"]

        """
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls) -> list:
        """
        Get the name list of this enum.

        Example:
            >>> TsxPolarization.list_names()
            ["SINGLE", "DUAL", "QUAD", "TWIN"]
        """
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_value(cls, val: Any) -> "ListEnum":
        """
        Get the enum class from its value:

        Args:
            val (Any): Value of the Enum

        Returns:
            ListEnum: Enum with value

        Example:
            >>> TsxPolarization.from_value("Q")
            <TsxPolarization.QUAD: 'Q'>
        """
        if isinstance(val, cls):
            val = val.value
        try:
            return next(enum for enum in cls if enum.value == val)
        except StopIteration as ex:
            raise ValueError(f"Non existing {val} in {cls.list_values()}") from ex

    @classmethod
    def convert_from(cls, to_convert: list | str) -> list:
        """
        Convert from a list or a string to an enum instance

        Args:
            to_convert (list | str): List or string to convert into an enum instance

        Returns:
            list: Converted list

        Example:
            >>> TsxPolarization.convert_from(["SINGLE", "S", TsxPolarization.QUAD])
            [<TsxPolarization.SINGLE: 'S'>, <TsxPolarization.SINGLE: 'S'>, <TsxPolarization.QUAD: 'Q'>]

        """
        if not isinstance(to_convert, list):
            to_convert = [to_convert]

        enums = []

        for tc in to_convert:
            if tc in cls.list_values():
                enums.append(cls.from_value(tc))
            elif tc in cls.list_names():
                enums.append(getattr(cls, tc))
            elif isinstance(tc, cls):
                enums.append(tc)
            else:
                raise TypeError(
                    f"Invalid name {tc}, "
                    f"should be chosen among {cls.list_values()} or {cls.list_names()}"
                )

        return enums


def unique(sequence: list):
    """
    Keep only the unique values of a sequence (any Iterable should work).

    Preserves the order of the sequence (except for sets, which are sorted).

    Args:
        sequence (Iterable): Sequence from which to keep only the unique values

    Returns:
        list: List containing only the unique values

    Example:
        >>> # With a list
        >>> unique([5, 4, 1, 2, 3, 1, 2])
        [5, 4, 1, 2, 3]

        >>> # With a tuple
        >>> unique((5, 4, 1, 2, 3, 1, 2))
        [5, 4, 1, 2, 3]

        >>> # With a set (the values are sorted, as a set would do)
        >>> unique({5, 4, 1, 2, 3})
        [1, 2, 3, 4, 5]

    """
    return list(dict.fromkeys(sequence))


def remove_empty_values(
    object_with_empty_values: list | dict, other_empty_values: list = None
) -> list | dict:
    """
    Remove the empty values from a list or a dictionary.

    Args:
        object_with_empty_values (list | dict): List or dictionary with empty values
        other_empty_values (list): Other values to consider as empty (in addition to :code:`None`)

    Returns:
        list | dict: List or dictionary without the empty values

    Example:
        >>> lst = ["A", "T", "R", "", 3, None]
        >>> remove_empty_values(lst)
        ["A", "T", "R", 3]

    """
    if not other_empty_values:
        other_empty_values = []

    if isinstance(object_with_empty_values, dict):
        return {
            k: v
            for k, v in object_with_empty_values.items()
            if v is not None and v not in other_empty_values
        }
    else:
        return [
            v
            for v in filter(None, object_with_empty_values)
            if v not in other_empty_values
        ]


def select_dict(dict_to_select: dict, keys_to_select: list) -> dict:
    """
    Select keys from a dictionary.

    Args:
        dict_to_select (dict): Dictionary to select from
        keys_to_select (list): List of keys to select in the dictionary

    Returns:
        dict: Dictionary with selected keys
    """
    return {k: v for k, v in dict_to_select.items() if k in keys_to_select}


def prune_dict(dict_to_prune: dict, keys_to_prune: list) -> dict:
    """
    Prune keys from a dictionary.

    Args:
        dict_to_prune (dict): Dictionary to prune from
        keys_to_prune (list): List of keys to prune in the dictionary

    Returns:
        dict: Dictionary with pruned keys
    """
    return {k: v for k, v in dict_to_prune.items() if k not in keys_to_prune}


# Error messages used by engines to reject a keyword argument by name:
#   TypeError  - "f() got an unexpected keyword argument 'name'"
#   ValueError - "unrecognized option 'NAME' for driver 'DRIVER'" (pyogrio)
_REJECTED_KWARG_RE = re.compile(
    r"got an unexpected keyword argument '(\w+)'"
    r"|unrecognized option '(\w+)' for driver"
)


def strip_kwargs(function: Callable) -> Callable:
    """
    Decorator: keyword arguments the engine does not accept are warned and dropped.

    When the engine rejects a keyword argument by name through its error message,
    e.g.:
    - :code:`TypeError: f() got an unexpected keyword argument 'name'`
    - :code:`ValueError: unrecognized option 'NAME' for driver 'DRIVER'` (pyogrio)

    the option is warned, dropped and the function is re-invoked without it.
    Engines that do not raise on unknown options (e.g. GDAL) cannot be handled
    this way: their unsupported options are reported by the engine itself.

    Any other error is propagated as is.

    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: Decorated function
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        for _ in range(len(kwargs) + 1):
            try:
                return function(*args, **kwargs)
            except (TypeError, ValueError) as exc:
                match = _REJECTED_KWARG_RE.search(str(exc))
                rejected = (match.group(1) or match.group(2)).lower() if match else None
                key = (
                    next((k for k in kwargs if k.lower() == rejected), None)
                    if rejected
                    else None
                )
                if key is None:
                    raise
                LOGGER.warning(
                    f"'{function.__name__}' does not accept '{key}'. It will be ignored."
                )
                del kwargs[key]

    return wrapper


def list_to_dict(dict_list: list) -> dict:
    """
    Return a dictionary from a list :code:`[key, value, key_2, value_2...]`

    Args:
        dict_list (list[str]): Dictionary as a list

    Returns:
        dict: Dictionary

    Example:
        >>> lst = ["A","T", "R", 3]
        >>> list_to_dict(lst)
        {"A": "T", "R": 3}
    """
    dictionary = {dict_list[i]: dict_list[i + 1] for i in range(0, len(dict_list), 2)}
    return dictionary


def nested_set(dic: dict, keys: list, value: Any) -> None:
    """
    Set a value in a nested dictionary.

    Args:
        dic (dict): Dictionary
        keys (list[str]): Keys as a list
        value (Any): Value to be set

    Example:
        >>> dct = {"A": "T", "R": 3}
        >>> nested_set(dct, keys=["B", "C", "D"], value="value")
        >>> dct
        {'A': 'T', 'R': 3, 'B': {'C': {'D': 'value'}}}
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def check_mandatory_keys(data_dict: dict, mandatory_keys: list) -> None:
    """
    Check all the mandatory arguments in a dictionary.
    Raise an exception if a mandatory argument is missing.

    **Note**: nested keys do not work here!

    Args:
        data_dict (dict): Data dictionary to be checked
        mandatory_keys (list[str]): List of mandatory keys

    Example:
        >>> dct = {"A": "T", "R": 3}
        >>> check_mandatory_keys(dct, ["A", "R"])  # Returns nothing, is OK
        >>> check_mandatory_keys(dct, ["C"])
        Traceback (most recent call last):
            ...
        ValueError: Missing mandatory key 'C' among {'A': 'T', 'R': 3}
    """

    for mandatory_key in mandatory_keys:
        if mandatory_key not in data_dict:
            raise ValueError(
                f"Missing mandatory key '{mandatory_key}' among {pprint.pformat(data_dict)}"
            )


def find_by_key(data: dict, target: str) -> Any:
    """
    Find a value by key in a dictionary.

    Args:
        data (dict): Dict to walk through
        target (str): target key

    Returns:
        Any: Value data[...][target]

    Example:
        >>> dct = {
        ...         "A": "T",
        ...         "R": 3,
        ...         "B": {
        ...             "C": {
        ...                 "D": "value"
        ...                  }
        ...              }
        ...        }
        >>> find_by_key(dct, "D")
        "value"

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


def run_cli(
    cmd: list | str,
    timeout: float = None,
    check_return_value: bool = True,
    in_background: bool = True,
    cwd="/",
) -> (int, str):
    """
    Run a command line.

    Args:
        cmd (str | list[str]): Command as a string or a list of strings
        timeout (float): Timeout
        check_return_value (bool): Check the return value of the executable
        in_background (bool): Run the subprocess in the background
        cwd (str): Working directory

    Returns:
        int, str: Return value and output

    Example:
        >>> # Run a command and get the output
        >>> retval, output = run_cli(["gdalinfo", "--version"], in_background=False)
        >>> retval
        0
    """
    if isinstance(cmd, list):
        cmd = [str(cmd_i) for cmd_i in cmd]
        cmd_line = " ".join(cmd)
    elif isinstance(cmd, str):
        cmd_line = cmd
    else:
        raise TypeError("The command line should be given as a str or a list")

    # Background
    LOGGER.debug(cmd_line)
    if in_background:
        stdout = None
        stderr = None
        close_fds = True
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        close_fds = False

    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    with subprocess.Popen(
        cmd_line,
        shell=True,
        stdout=stdout,
        stderr=stderr,
        cwd=cwd,
        start_new_session=True,
        close_fds=close_fds,
    ) as process:
        output = ""
        if not in_background:
            for line in process.stdout:  # pragma: no cover
                line = line.decode(
                    encoding=sys.stdout.encoding,
                    errors=(
                        "replace" if sys.version_info < (3, 5) else "backslashreplace"
                    ),
                ).rstrip()
                LOGGER.info(line)
                output += line

        # Get return value
        retval = process.wait(timeout)

        # Kill process
        process.kill()

    # Check return value
    if check_return_value and retval != 0:
        raise RuntimeError(f"Exe {cmd[0]} has failed.")  # pragma: no cover

    return retval, output


def get_function_name() -> str:
    """
    Get the name of the function where this one is launched.

    Returns:
        str: Function's name

    Example:
        >>> def huhuhu():
        ...     return get_function_name()
        ...
        >>> huhuhu()
        "huhuhu"
    """
    # pylint: disable=W0212
    return sys._getframe(1).f_code.co_name


def in_docker() -> bool:
    """
    Check if the session is running inside Docker.

    Returns:
        bool: True if inside a Docker container

    Example:
        >>> if in_docker():
        ...     print("OMG we are stuck in a Docker! Get me out of here!")
        ... else:
        ...     print("We are safe")
    """
    try:
        with open("/proc/1/cgroup") as ifh:
            in_dck = "docker" in ifh.read()
    except Exception:  # pragma: no cover
        in_dck = False

    return in_dck


@contextmanager
def chdir(newdir: AnyPathStrType) -> None:
    """
    Change the current directory, used as a context manager, i.e.:

    Args:
        newdir (str): New directory

    Example:
        >>> folder = r"C:/"
        >>> with chdir(folder):
        ...     print(os.getcwd())
        'C:/'
    """
    newdir = AnyPath(newdir)
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def compare(a, b, operation: str) -> bool:
    """
    Compare two objects using a specific operation.
    Using this function allows to ask the user the operation he wants (see compare_version for example)

    Args:
        a (Any): First object
        b (Any): Second object
        operation (str): Operator to use (:code:`>`, :code:`<`, :code:`>=`, :code:`<=`, :code:`==`)

    Returns:
        bool: True if the comparison between the two objects is respected

    Example:
        >>> compare(1, 2, ">=")
        False

    """
    import operator

    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
    }
    return ops[operation](a, b)


def compare_version(lib: str | Version, version_to_check: str, operator: str) -> bool:
    """
    Compare the version of a library to a reference version, given the operator.

    Args:
        lib (str | Version): Name of the library, its version as a string, or as a Version object
        version_to_check (str): Version of the library to be compared
        operator (str): Operator to use (:code:`>`, :code:`<`, :code:`>=`, :code:`<=`, :code:`==`)

    Returns:
        bool: True if the comparison between the version of the library and the reference version is respected

    Example:
        >>> compare_version("geopandas", "0.10.0", ">=")
        True
        >>> compare_version(sertit.__version__, "1.0.0", ">=")
        True

    """
    from importlib.metadata import PackageNotFoundError, version

    if isinstance(lib, Version):
        lib_version = lib
    elif isinstance(lib, str):
        try:
            lib_version = Version(version(lib))
        except PackageNotFoundError:
            lib_version = Version(lib)
    else:
        raise TypeError(
            "'lib' should either be the name of your library as a string or directly a 'Version' object."
        )

    return compare(lib_version, Version(version_to_check), operator)
