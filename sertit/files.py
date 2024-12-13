# Copyright 2024, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""Tools for paths and files"""

import hashlib
import json
import logging
import os
import shutil
from datetime import date, datetime
from enum import Enum
from json import JSONDecoder, JSONEncoder
from pathlib import Path
from typing import Any, Union

import dill
import numpy as np

from sertit import AnyPath, logs, path, s3
from sertit.logs import SU_NAME
from sertit.strings import DATE_FORMAT
from sertit.types import AnyPathStrType, AnyPathType

LOGGER = logging.getLogger(SU_NAME)


def get_root_path() -> AnyPathType:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get the root path of the current disk:

    - On Linux this returns :code:`/`
    - On Windows this returns :code:`C:/` or whatever the current drive is

    Example:
        >>> get_root_path()
        "/" on Linux
        "C:/" on Windows (if you run this code from the C: drive)
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_root_path()


def listdir_abspath(directory: AnyPathStrType) -> list:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get absolute path of all files in the given directory.

    It is the same function than :code:`os.listdir` but returning absolute paths.

    Args:
        directory (AnyPathStrType): Relative or absolute path to the directory to be scanned

    Returns:
        str: Absolute path of all files in the given directory

    Example:
        >>> folder = "."
        >>> listdir_abspath(folder)
        ['D:/_SERTIT_UTILS/sertit-utils/sertit/files.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/logs.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/misc.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/network.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/rasters_rio.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/strings.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/vectors.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/version.py',
        'D:/_SERTIT_UTILS/sertit-utils/sertit/__init__.py']
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.listdir_abspath(directory)


def to_abspath(
    raw_path: AnyPathStrType,
    create: bool = True,
    raise_file_not_found: bool = True,
) -> AnyPathType:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Return the absolute path of the specified path and check if it exists

    If not:

    - If it is a file (aka has an extension), it raises an exception
    - If it is a folder, it creates it

    To be used with argparse to retrieve the absolute path of a file, like:

    Args:
        raw_path (AnyPathStrType): Path as a string (relative or absolute)
        create (bool): Create directory if not existing

    Returns:
        AnyPathType: Absolute path

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> # Add config file path key
        >>> parser.add_argument(
        >>>     "--config",
        >>>     help="Config file path (absolute or relative)",
        >>>     type=to_abspath
        >>> )
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.to_abspath(raw_path, create, raise_file_not_found)


def real_rel_path(raw_path: AnyPathStrType, start: AnyPathStrType) -> AnyPathType:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Gives the real relative path from a starting folder.
    (and not just adding :code:`../..` between the start and the target)

    Args:
        raw_path (AnyPathStrType): Path to make relative
        start (AnyPathStrType): Start, the path being relative from this folder.

    Returns:
        Relative path

    Example:
        >>> path = r'D:/_SERTIT_UTILS/sertit-utils/sertit'
        >>> start = os.path.join(".", "..", "..")
        >>> real_rel_path(path, start)
        'sertit-utils/sertit'
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.real_rel_path(raw_path, start)


def get_filename(file_path: AnyPathStrType, other_exts: Union[list, str] = None) -> str:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get file name (without extension) from file path, i.e.:

    Args:
        file_path (AnyPathStrType): Absolute or relative file path (the file doesn't need to exist)
        other_exts (Union[list, str]): Other double extensions to discard

    Returns:
        str: File name (without extension)

    Example:
        >>> file_path = 'D:/path/to/filename.zip'
        >>> get_file_name(file_path)
        'filename'
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_filename(file_path, other_exts)


def get_ext(file_path: AnyPathStrType) -> str:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get file extension from file path.

    .. WARNING::
        Extension is given WITHOUT THE FIRST POINT

    Args:
        file_path (AnyPathStrType): Absolute or relative file path (the file doesn't need to exist)

    Returns:
        str: File name (without extension)

    Example:
        >>> file_path = 'D:/path/to/filename.zip'
        >>> get_ext(file_path)
        'zip'
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_ext(file_path)


def remove(path: AnyPathStrType) -> None:
    """
    Deletes a file or a directory (recursively) using :code:`shutil.rmtree` or :code:`os.remove`.

    Args:
        path (AnyPathStrType): Path to be removed

    Example:
        >>> path_to_remove = 'D:/path/to/remove'  # Could also be a file
        >>> remove(path_to_remove)
        path_to_remove deleted
    """
    path = AnyPath(path)
    if not path.exists():
        LOGGER.debug("Non existing %s", path)

    elif path.is_dir():
        try:
            shutil.rmtree(path)
        except OSError:
            LOGGER.debug("Impossible to remove the directory %s", path, exc_info=True)

    elif path.is_file():
        try:
            path.unlink()
        except OSError:
            LOGGER.debug("Impossible to remove the file %s", path, exc_info=True)


def remove_by_pattern(
    directory: AnyPathStrType,
    name_with_wildcard: str = "*",
    extension: str = None,
) -> None:
    """
    Remove files corresponding to a pattern from a directory.

    Args:
        directory (AnyPathStrType): Directory where to find the files
        name_with_wildcard (str): Filename (wildcards accepted)
        extension (str): Extension wanted, optional. With or without point. (yaml or .yaml accepted)

    Example:
        >>> directory = 'D:/path/to/folder'
        >>> os.listdir(directory)
        ["huhu.exe", "blabla.geojson", "haha.txt", "blabla"]
        >>>
        >>> remove(directory, "blabla*")
        >>> os.listdir(directory)
        ["huhu.exe", "haha.txt"] # Removes also directories
        >>>
        >>> remove(directory, "*", extension="txt")
        >>> os.listdir(directory)
        ["huhu.exe"]
    """
    directory = AnyPath(directory)
    if extension and not extension.startswith("."):
        extension = "." + extension

    file_list = directory.glob(name_with_wildcard + extension)
    for file in file_list:
        remove(file)


def copy(src: AnyPathStrType, dst: AnyPathStrType) -> AnyPathType:
    """
    Copy a file or a directory (recursively) with :code:`copytree` or :code:`copy2`.

    Args:
        src (AnyPathStrType): Source Path
        dst (AnyPathStrType): Destination Path (file or folder)

    Returns:
        AnyPathType: New path

    Examples:
        >>> src = 'D:/path/to/copy'
        >>> dst = 'D:/path/to/output'
        >>> copy(src, dst)
        copydir 'D:/path/to/output/copy'

        >>> src = 'D:/path/to/copy.txt'
        >>> dst = 'D:/path/to/output/huhu.txt'
        >>> copyfile = copy(src, dst)
        'D:/path/to/output/huhu.txt' but with the content of copy.txt
    """
    src = AnyPath(src)

    if path.is_cloud_path(src):
        out = s3.download(src, dst)
    else:
        out = None
        try:
            if src.is_dir():
                out = AnyPath(shutil.copytree(src, dst))
            elif os.path.isfile(src):
                out = AnyPath(shutil.copy2(src, dst))
        except shutil.Error:
            LOGGER.debug("Error in copy!", exc_info=True)
            out = src
            # eg. source or destination doesn't exist
        except OSError as ex:
            raise OSError(f"Copy error: {ex.strerror}") from ex

    return out


def find_files(
    names: Union[list, str],
    root_paths: Union[list, AnyPathStrType],
    max_nof_files: int = -1,
    get_as_str: bool = False,
) -> Union[list, str]:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Returns matching files recursively from a list of root paths.

    Regex are allowed (using glob)

    Args:
        names (Union[list, str]): File names.
        root_paths (Union[list, str]): Root paths
        max_nof_files (int): Maximum number of files (set to -1 for unlimited)
        get_as_str (bool): if only one file is found, it can be retrieved as a string instead of a list

    Returns:
        list: File name

    Examples:
        >>> root_path = 'D:/root'
        >>> dir1_path = 'D:/root/dir1'
        >>> dir2_path = 'D:/root/dir2'
        >>>
        >>> os.listdir(dir1_path)
        ["haha.txt", "huhu.txt", "hoho.txt"]
        >>> os.listdir(dir2_path)
        ["huhu.txt", "hehe.txt"]
        >>>
        >>> find_files("huhu.txt", root_path)
        ['D:/root/dir1/huhu.txt', 'D:/root/dir2/huhu.txt']
        >>>
        >>> find_files("huhu.txt", root_path, max_nof_files=1)
        ['D:/root/dir1/huhu.txt']

        >>> find_files("huhu.txt", root_path, max_nof_files=1, get_as_str=True)
        found = 'D:/root/dir1/huhu.txt'
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.find_files(names, root_paths, max_nof_files, get_as_str)


# subclass JSONDecoder
class CustomDecoder(JSONDecoder):
    """Decoder for JSON with methods for datetimes"""

    # Override the default method
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, *args, object_hook=self.object_hook, **kwargs)

    def object_hook(self, obj: Any):
        """
        Overload of object_hook function that deals with :code:`datetime.datetime`

        Args:
            obj (dict): Dict containing objects to decode from JSON

        Returns:
            dict: Dict with decoded object
        """
        for key, val in obj.items():
            if isinstance(val, str):
                try:
                    # Datetime -> Encoder saves dates as isoformat: "%Y-%m-%dT%H:%M:%S" (DATE_FORMAT)
                    # Isoformat in Python 3.11 has been extended and has too many ways of reading datetimes
                    # We just want the one reading the output of 'save_json' (see the encoder)
                    # See controversy here: https://github.com/python/cpython/issues/107779
                    obj[key] = datetime.strptime(val, DATE_FORMAT)
                except ValueError:
                    try:
                        if "." in val:
                            # We have microseconds...
                            obj[key] = datetime.strptime(val, DATE_FORMAT + ".%f")
                        else:
                            # Date -> Encoder saves date as isoformat: %Y-%m-%d
                            obj[key] = datetime.strptime(val, "%Y-%m-%d").date()
                    except ValueError:
                        obj[key] = val
            else:
                obj[key] = val
        return obj


# subclass JSONEncoder
class CustomEncoder(JSONEncoder):
    """Encoder for JSON with methods for datetimes and np.int64"""

    # pylint: disable=W0221
    def default(self, obj):
        """Overload of the default method"""
        if isinstance(obj, (date, datetime)):
            out = obj.isoformat()
        elif isinstance(obj, (np.int64, np.int32)):
            out = int(obj)
        elif isinstance(obj, Enum):
            out = obj.value
        elif isinstance(obj, set | Path) or path.is_cloud_path(obj):
            out = str(obj)
        else:
            out = json.JSONEncoder.default(self, obj)

        return out


def read_json(json_file: AnyPathStrType, print_file: bool = True) -> dict:
    """
    Read a JSON file

    Args:
        json_file (AnyPathStrType): Path to JSON file
        print_file (bool):  Print the configuration file

    Returns:
        dict: JSON data

    Example:
        >>> json_path = 'D:/path/to/json.json'
        >>> read_json(json_path, print_file=False)
        {"A": 1, "B": 2}
    """

    with open(json_file) as file:
        data = json.load(file, cls=CustomDecoder)
        if print_file:
            LOGGER.debug(
                "Configuration file %s contains:\n%s",
                json_file,
                json.dumps(data, indent=3, cls=CustomEncoder),
            )
    return data


def save_json(json_dict: dict, output_json: AnyPathStrType, **kwargs) -> None:
    """
    .. versionchanged:: 1.32.0
       The order of the function has changed. Please set json_dict in first!

    Save a JSON file, with datetime, numpy types and Enum management.

    Args:
        json_dict (dict): Json dictionary
        output_json (AnyPathStrType): Output file
        **kwargs: Other arguments

    Example:
        >>> output_json = 'D:/path/to/json.json'
        >>> json_dict = {"A": np.int64(1), "B": datetime.today(), "C": SomeEnum.some_name}
        >>> save_json(output_json, json_dict)
    """
    if isinstance(output_json, dict):
        # Old order. Swap the variables.
        logs.deprecation_warning(
            "The order of the function has changed. Please set json_dict in first!"
        )
        tmp = output_json
        output_json = json_dict
        json_dict = tmp

    kwargs["indent"] = kwargs.get("indent", 3)
    kwargs["cls"] = kwargs.get("cls", CustomEncoder)

    with open(output_json, "w") as output_file:
        json.dump(json_dict, output_file, **kwargs)


def save_obj(obj: Any, path: AnyPathStrType, **kwargs) -> None:
    """
    Save an object as a pickle (can save any Python objects).

    Args:
        obj (Any): Any object serializable
        path (AnyPathStrType): Path where to write the pickle

    Example:
        >>> output_pkl = 'D:/path/to/pickle.pkl'
        >>> pkl_dict = {"A": np.ones([3, 3]),
                        "B": datetime.today(),
                        "C": SomeEnum.some_name}
        >>> save_json(output_pkl, pkl_dict)
    """
    with open(path, "wb+") as file:
        dill.dump(obj, file, **kwargs)


def load_obj(path: AnyPathStrType) -> Any:
    """
    Load a pickled object.

    Args:
        path (AnyPathStrType): Path of the pickle

    Returns:
        object (Any): Pickled object

    Example:
        >>> output_pkl = 'D:/path/to/pickle.pkl'
        >>> load_obj(output_pkl)
        {"A": np.ones([3, 3]), "B": datetime.today(), "C": SomeEnum.some_name}

    """
    with open(path, "rb") as file:
        return dill.load(file)


# too many arguments
# pylint: disable=R0913
def get_file_in_dir(
    directory: AnyPathStrType,
    pattern_str: str,
    extension: str = None,
    filename_only: bool = False,
    get_list: bool = False,
    exact_name: bool = False,
) -> Union[AnyPathType, list]:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get one or all matching files (pattern + extension) from inside a directory.

    Note that the pattern is a regex with glob's convention, i.e. :code:`*pattern*`.

    If :code:`exact_name` is :code:`False`, the searched pattern will be :code:`*{pattern}*.{extension}`,
    else :code:`{pattern}.{extension}`.

    Args:
        directory (str): Directory where to find the files
        pattern_str (str): Pattern wanted as a string, with glob's convention.
        extension (str): Extension wanted, optional. With or without point. (:code:`yaml` or :code:`.yaml` accepted)
        filename_only (bool): Get only the filename
        get_list (bool): Get the whole list of matching files
        exact_name (bool): Get the exact name (without adding :code:`*` before and after the given pattern)

    Returns:
        Union[AnyPathType, list]: File

    Example:
        >>> directory = 'D:/path/to/dir'
        >>> os.listdir(directory)
        ["haha.txt", "huhu1.txt", "huhu1.geojson", "hoho.txt"]
        >>>
        >>> get_file_in_dir(directory, "huhu")
        'D:/path/to/dir/huhu1.geojson'
        >>>
        >>> get_file_in_dir(directory, "huhu", extension="txt")
        'D:/path/to/dir/huhu1.txt'
        >>>
        >>> get_file_in_dir(directory, "huhu", get_list=True)
        ['D:/path/to/dir/huhu1.txt', 'D:/path/to/dir/huhu1.geojson']
        >>>
        >>> get_file_in_dir(directory, "huhu", filename_only=True, get_list=True)
        ['huhu1.txt', 'huhu1.geojson']
        >>>
        >>> get_file_in_dir(directory, "huhu", get_list=True, exact_name=True)
        []
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_file_in_dir(
        directory, pattern_str, extension, filename_only, get_list, exact_name
    )


# pylint: disable=E1121
def hash_file_content(file_content: str, len_param: int = 5) -> str:
    """
    Hash a file into a unique str.

    Args:
        file_content (str): File content
        len_param (int): Length parameter for the hash (length of the key will be 2x this number)

    Returns:
        str: Hashed file content

    Example:
        >>> read_json("path/to/json.json")
        {"A": 1, "B": 2}
        >>>
        >>> hash_file_content(str(file_content))
        "d3fad5bdf9"
    """
    hasher = hashlib.shake_256()
    hasher.update(str.encode(file_content))
    return hasher.hexdigest(len_param)


def is_writable(dir_path: AnyPathStrType):
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Determine whether the directory is writeable or not.

    Args:
        dir_path (AnyPathStrType): Directory path

    Returns:
        bool: True if the directory is writable
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.is_writable(dir_path)
