# Copyright 2025, SERTIT-ICube - France, https://sertit.unistra.fr/
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
from typing import Any

import dill
import numpy as np

from sertit import AnyPath, path, s3
from sertit.logs import SU_NAME
from sertit.strings import DATE_FORMAT
from sertit.types import AnyPathStrType, AnyPathType

LOGGER = logging.getLogger(SU_NAME)


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
