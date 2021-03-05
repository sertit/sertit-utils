""" Miscellaneous Tools """

import os
import sys
import subprocess
import logging
import pprint
from contextlib import contextmanager
from enum import Enum, unique
from typing import Any, Union
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


@unique
class ListEnum(Enum):
    """
    List Enum (enum with function listing names and values)

    ```python
    >>> @unique
    >>> class TsxPolarization(ListEnum):
    >>>     SINGLE = "S"  # Single
    >>>     DUAL = "D"  # Dual
    >>>     QUAD = "Q"  # Quad
    >>>     TWIN = "T"  # Twin
    ```
    """

    @classmethod
    def list_values(cls) -> list:
        """
        Get the value list of this enum

        ```python
        >>> TsxPolarization.list_values()
        ["S", "D", "Q", "T"]
        ```

        """
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_names(cls) -> list:
        """
        Get the name list of this enum:

        ```python
        >>> TsxPolarization.list_values()
        ["SINGLE", "DUAL", "QUAD", "TWIN"]
        ```
        """
        return list(map(lambda c: c.name, cls))

    @classmethod
    def from_value(cls, val: Any) -> 'ListEnum':
        """
        Get the enum class from its value:

        ```python
        >>> TsxPolarization.from_value("Q")
        <TsxPolarization.QUAD: 'Q'>
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


def remove_empty_values(list_with_empty_values: list) -> list:
    """
    Remove empty values from list:

    ```python
    >>> lst = ["A", "T", "R", "", 3, None]
    >>> list_to_dict(lst)
    ["A", "T", "R", 3]
    ```

    Args:
        list_with_empty_values (list): List with empty values
    Returns:
        list: Curated list
    """

    return list(filter(None, list_with_empty_values))


def list_to_dict(dict_list: list) -> dict:
    """
    Return a dictionary from a list `[key, value, key_2, value_2...]`

    ```python
    >>> lst = ["A","T", "R", 3]
    >>> list_to_dict(lst)
    {"A": "T", "R": 3}
    ```

    Args:
        dict_list (list[str]): Dictionary as a list

    Returns:
        dict: Dictionary
    """
    dictionary = {dict_list[i]: dict_list[i + 1] for i in range(0, len(dict_list), 2)}
    return dictionary


def nested_set(dic: dict, keys: list, value: Any) -> None:
    """
    Set value in nested directory:

    ```python
    >>> dct = {"A": "T", "R": 3}
    >>> nested_set(dct, keys=["B", "C", "D"], value="value")
    {
    "A": "T",
    "R": 3,
    "B": {
         "C": {
              "D": "value"
              }
         }
    }
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
    >>> dct = {"A": "T", "R": 3}
    >>> check_mandatory_keys(dct, ["A", "R"])  # Returns nothing, is OK
    >>> check_mandatory_keys(dct, ["C"])
    Traceback (most recent call last):
      File "<input>", line 1, in <module>
      File "<input>", line 167, in check_mandatory_keys
    ValueError: Missing mandatory key 'C' among {'A': 'T', 'R': 3}
    ```

    Args:
        data_dict (dict): Data dictionary to be checked
        mandatory_keys (list[str]): List of mandatory keys
    """

    for mandatory_key in mandatory_keys:
        if mandatory_key not in data_dict:
            raise ValueError(f"Missing mandatory key '{mandatory_key}' among {pprint.pformat(data_dict)}")


def find_by_key(data: dict, target: str) -> Any:
    """
    Find a value by key in a dictionary.

    ```python
    >>> dct = {
    >>>         "A": "T",
    >>>         "R": 3,
    >>>         "B": {
    >>>             "C": {
    >>>                 "D": "value"
    >>>                  }
    >>>              }
    >>>        }
    >>> find_by_key(dct, "D")
    "value"
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


def run_cli(cmd: Union[str, list],
            timeout: float = None,
            check_return_value: bool = True,
            in_background: bool = True,
            cwd='/') -> (int, str):
    """
    Run a command line.

    ```python
    >>> cmd_hillshade = ["gdaldem", "--config",
    >>>                  "NUM_THREADS", "1",
    >>>                  "hillshade", strings.to_cmd_string(dem_path),
    >>>                  "-compute_edges",
    >>>                  "-z", self.nof_threads,
    >>>                  "-az", azimuth,
    >>>                  "-alt", zenith,
    >>>                  "-of", "GTiff",
    >>>                  strings.to_cmd_string(hillshade_dem)]
    >>> # Run command
    >>> run_cli(cmd_hillshade)
    ```

    Args:
        cmd (str or list[str]): Command as a list
        timeout (float): Timeout
        check_return_value (bool): Check output value of the exe
        in_background (bool): Run the subprocess in background
        cwd (str): Working directory

    Returns:
        int, str: return value and output log
    """
    if isinstance(cmd, list):
        cmd = [str(cmd_i) for cmd_i in cmd]
        cmd_line = ' '.join(cmd)
    elif isinstance(cmd, str):
        cmd_line = cmd
    else:
        raise TypeError('The command line should be given as a str or a list')

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
    with subprocess.Popen(cmd_line,
                          shell=True,
                          stdout=stdout,
                          stderr=stderr,
                          cwd=cwd,
                          start_new_session=True,
                          close_fds=close_fds) as process:
        output = ''
        if not in_background:
            for line in process.stdout:
                line = line.decode(encoding=sys.stdout.encoding,
                                   errors='replace' if sys.version_info < (3, 5)
                                   else 'backslashreplace').rstrip()
                LOGGER.info(line)
                output += line

        # Get return value
        retval = process.wait(timeout)

        # Kill process
        process.kill()

    # Check return value
    if check_return_value and retval != 0:
        raise RuntimeError(f"Exe {cmd[0]} has failed.")

    return retval, output


def get_function_name() -> str:
    """
    Get the name of the function where this one is launched.

    ```python
    >>> def huhuhu():
    >>>     return get_function_name()
    >>> huhuhu()
    "huhuhu"
    ```

    Returns:
        str: Function's name
    """
    # pylint: disable=W0212
    return sys._getframe(1).f_code.co_name


def in_docker() -> bool:
    """
    Check if the session is running inside a docker

    ```python
    if in_docker():
        print("OMG we are stock in a Docker ! Get me out of here !")
    else:
        print("We are safe")
    ```

    Returns:
        bool: True if inside a docker
    """
    try:
        with open('/proc/1/cgroup', 'rt') as ifh:
            in_dck = 'docker' in ifh.read()
    # pylint: disable=W0703
    except Exception:
        in_dck = False

    return in_dck


@contextmanager
def chdir(newdir: str) -> None:
    """
    Change current directory, used as a context manager, ie:

    ```python
    >>> folder = r"C:\"
    >>> with chdir(folder):
    >>>     print(os.getcwd())
    'C:\\'
    ```

    Args:
        newdir (str): New directory
    """
    prevdir = os.getcwd()
    newdir = newdir.replace("\\", "/")
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
