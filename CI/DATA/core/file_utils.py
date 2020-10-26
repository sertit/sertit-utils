""" Tools for paths and files """

import glob
import os
import logging
import pprint
import tarfile
import tempfile
import zipfile
import json
import shutil
import pickle
import hashlib
from tqdm import tqdm
import numpy as np
from datetime import date, datetime
from json.encoder import JSONEncoder
from typing import Union, Any

from sertit_utils.core import sys_utils

LOGGER = logging.getLogger('sertit_utils')


def root_path():
    """
    Get the root path of the current disk:
    - On Linux this returns `/`
    - On Windows this returns `C:\\` or whatever the current drive is
    """
    return os.path.abspath(os.sep)

def listdir_abspath(directory):
    """
    Get absolute path of all files in the given directory.
    It is the same function than `os.listdir` but returning absolute paths.

    Args:
        directory (str): Relative or absolute path to the directory to be scanned

    Returns:
        str: Absolute path of all files in the given directory
    """
    return [os.path.abspath(os.path.join(directory, file)) for file in os.listdir(directory)]


def to_abspath(path_str):
    """
    Return the absolute path of the specified path and check if it exists

    If not:

    - If it is a file (aka has an extension), it raises an exception
    - If it is a folder, it creates it

    To be used with argparse like:
    ```python:
    import argparse
    parser = argparse.ArgumentParser()

    # Add config file path key
    parser.add_argument("--config",
                        help="Config file path (absolute or relative)",
                        type=to_abspath,
                        required=True)
    ```

    Args:
        path_str (str): Path as a string (relative or absolute)

    Returns:
        str: Absolute path
    """
    abs_path = os.path.abspath(path_str)

    if not os.path.exists(abs_path):
        if os.path.splitext(abs_path)[1]:
            # If the path specifies a file (with extension), it raises an exception
            raise Exception("Non existing file: {}".format(abs_path))

        # If the path specifies a folder, it creates it
        os.makedirs(abs_path)

    return abs_path


def real_rel_path(path: str, start: str) -> str:
    """
    Gives the real relative path from a starting folder.
    (and not just adding `..\..` between the start and the target)

    Args:
        path (str): Path to make relative
        start (str): Start, the path being relative from this folder.

    Returns:
        Relative path
    """
    return os.path.join(os.path.relpath(os.path.dirname(path), start), os.path.basename(path))


def extract_file(file_path: str, output: str, overwrite: bool = False) -> str:
    """
    Extract an archived file (zip or others). Overwrites if specified.

    Args:
        file_path (str): Archive file path
        output (str): Output where to put the extracted file
        overwrite (bool): Overwrite found extracted files

    Returns:
        str: Extracted file path
    """
    # Get extracted name
    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            extracted_name = os.path.dirname(zip_ref.namelist()[0])
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tar"):
        # Tar files have no subdirectories, so create one
        extracted_name = get_file_name(file_path)
    else:
        raise TypeError("ExtractEO can only extract {}".format(file_path))

    # Get extracted directory
    extracted_dir = os.path.join(output, extracted_name)

    # Do not process if already existing directory
    if os.path.isdir(extracted_dir):
        if overwrite:
            LOGGER.debug("Already existing extracted %s. It will be overwritten as asked.", extracted_name)
            remove(extracted_dir)
        else:
            LOGGER.debug("Already existing extracted %s. It won't be overwritten.", extracted_name)
    else:
        LOGGER.debug("Extracting %s", extracted_name)
        # Inside docker, extracting files is really slow -> copy the archive in a tmp directory
        tmp = None
        if sys_utils.in_docker():
            tmp = tempfile.TemporaryDirectory()
            copy(file_path, tmp.name)
            file_path = os.path.join(tmp.name, os.path.basename(file_path))
            tmp_extract_output = tmp.name
            tmp_extracted_dir = os.path.join(tmp_extract_output, extracted_name)  # Recreate dir with tmp output
        else:
            tmp_extract_output = output
            tmp_extracted_dir = extracted_dir

        # Get extractor
        if file_path.endswith(".zip"):
            archive = zipfile.ZipFile(file_path, "r")
        else:
            archive = tarfile.open(file_path, "r")
            tmp_extract_output = tmp_extracted_dir  # Tar files do not contain a file tree

        # Extract product
        try:
            os.makedirs(tmp_extracted_dir, exist_ok=True)
            archive.extractall(path=tmp_extract_output)
        except tarfile.ReadError as ex:
            raise TypeError("Impossible to extract {}".format(file_path)) from ex

        # Copy back if we are running inside docker
        if tmp is not None:
            copy(tmp_extracted_dir, extracted_dir)
            tmp.cleanup()

    return extracted_dir


def extract_files(archives: list, output: str, overwrite: bool = False) -> list:
    """
    Extract all archived files. Overwrites if specified.

    Args:
        archives (list of str): List of archives to be extracted
        output (str): Output folder where extracted files will be written
        overwrite (bool): Overwrite found extracted files

    Returns:
        list: Extracted files (even pre-existing ones)
    """
    LOGGER.debug("Unzipping products in %s", output)
    progress_bar = tqdm(archives)
    extracts = []
    for archive in archives:
        progress_bar.set_description('Unzipping products {}'.format(os.path.basename(archive)))
        extracts.append(extract_file(archive, output, overwrite))

    return extracts


def get_file_name(file_path: str) -> str:
    """
    Get file name (without extension) from file path, ie:

    `docs\\html\\sertit_utils\\core\\file_utils.html` -> `file_utils`

    Args:
        file_path (str): Absolute or relative file path (the file doesn't need to exist)

    Returns:
        str: File name (without extension)
    """
    if file_path.endswith("/") or file_path.endswith("\\"):
        file_path = file_path[:-1]

    # We need to avoid splitext because of nested extensions such as .tar.gz
    basename = os.path.basename(file_path)
    return basename.split(".")[0]


def remove(path: str) -> None:
    """
    Delete a file or a directory (recursively) using `rmtree` or `remove`.

    Args:
        path (str): Path to be removed
    """
    if not os.path.exists(path):
        LOGGER.debug("Non existing %s", path)

    elif os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError as ex:
            LOGGER.debug("Impossible to remove the directory %s", path)
            LOGGER.debug(ex)

    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError as ex:
            LOGGER.debug("Impossible to remove the file %s", path)
            LOGGER.debug(ex)


def remove_by_pattern(directory: str, name_with_wildcard: str = "*", extension: str = None) -> None:
    """
    Remove files corresponding to a pattern from a directory.

    Args:
        directory (str): Directory where to find the files
        name_with_wildcard (str): Filename (wildcards accepted)
        extension (str): Extension wanted, optional. With or without point. (yaml or .yaml accepted)
    """

    if extension and not extension.startswith("."):
        extension = '.' + extension

    pattern_str = os.path.join(directory, name_with_wildcard + extension)
    file_list = glob.glob(pattern_str, recursive=False)
    for file in file_list:
        remove(file)


def copy(src: str, dst: str) -> str:
    """
    Copy a file or a directory (recursively) with `copytree` or `copy2`.

    Args:
        src (str): Source Path
        dst (str): Destination Path (file or folder)

    Returns:
        str: New path
    """
    out = None
    try:
        if os.path.isdir(src):
            out = shutil.copytree(src, dst)
        elif os.path.isfile(src):
            out = shutil.copy2(src, dst)
    except shutil.Error as ex:
        LOGGER.debug(ex)
        out = src
        # eg. source or destination doesn't exist
    except IOError as ex:
        raise IOError('Copy error: {}'.format(ex.strerror)) from ex

    return out


def find_files(names: Union[list, str],
               root_paths: Union[list, str],
               max_nof_files: int = -1,
               get_as_str: bool = False) -> Union[list, str]:
    """
    Returns matching files recursively from a list of root paths.

    Regex are not allowed, only exact matches are looking for (using `os.walk()`)

    Args:
        names ([str]): File names.
        root_paths ([str]): Root paths
        max_nof_files (int): Maximum number of files (set to -1 for unlimited)
        get_as_str (bool): if only one file is requested, it can be retrieved as a string instead of a list

    Returns:
        list: File name
    """
    paths = []

    # Transform to list
    if not isinstance(names, list):
        names = [names]

    if not isinstance(root_paths, list):
        root_paths = [root_paths]

    nof_found_file = 0

    try:
        for root_path in root_paths:
            for root, _, files in os.walk(root_path):
                for name in names:
                    if name in files:
                        paths.append(os.path.join(root, name))
                        nof_found_file += 1

                    if nof_found_file > 0 and nof_found_file == max_nof_files:
                        raise StopIteration

    except StopIteration:
        pass

    # Check if found
    if not paths:
        raise FileNotFoundError("Files {} not found in {}".format(names, root_paths))

    LOGGER.debug("Paths found in %s for filenames %s:\n%s", root_paths, names, pprint.pformat(paths))

    # Get str if needed
    if max_nof_files == 1 and get_as_str:
        paths = paths[0]

    return paths


def read_json(json_file, print_file=True):
    """
    Read a JSON file

    Args:
        json_file (str): Path to JSON file
        print_file (bool):  Print the configuration file

    Returns:
        dict: JSON data

    """
    with open(json_file) as file:
        data = json.load(file)
        if print_file:
            LOGGER.debug("Configuration file %s contains:\n%s", json_file, json.dumps(data, indent=3))
    return data


def save_json(output_json: str, json_dict: dict) -> None:
    """
    Save a JSON file, with datetime and numpy types management.

    Args:
        output_json (str): Output file
        json_dict (dict): Json dictionary
    """

    # subclass JSONEncoder
    class CustomEncoder(JSONEncoder):
        """ Encoder for JSON with methods for datetimes and np.int64 """

        # Parameters differ from overridden 'get_default_band_path' method (arguments-differ)
        # pylint: disable=W0221
        # Override the default method
        def default(self, obj):
            if isinstance(obj, (date, datetime)):
                return obj.isoformat()
            if isinstance(obj, np.int64):
                return int(obj)

            return obj

    with open(output_json, 'w') as output_config_file:
        json.dump(json_dict, output_config_file, indent=3, cls=CustomEncoder)


def save_obj(obj: Any, path: str) -> None:
    """
    Save an object as a pickle.

    Args:
        obj (Any): Any object serializable
        path (str): Path where to write the pickle

    Returns:

    """
    with open(path, 'wb+') as file:
        pickle.dump(obj, file)


def load_obj(path: str) -> Any:
    """
    Load a pickled object.

    Args:
        path (str): Path of the pickle
    Returns:
        object (Any): Pickled object

    """
    with open(path, 'rb') as file:
        return pickle.load(file)


# too many arguments
# pylint: disable=R0913
def get_file_in_dir(directory: str,
                    pattern_str: str,
                    extension: str = None,
                    filename_only: bool = False,
                    get_list: bool = False,
                    exact_name: bool = False) -> Union[str, list]:
    """
    Get one or all matching files (pattern + extension) from inside a directory.

    Note that the pattern is a regex with glob's convention, ie. `*pattern*`.

    If `exact_name` is `False`, the searched pattern will be `*{pattern}*.{extension}`, else `{pattern}.{extension}`.

    Args:
        directory (str): Directory where to find the files
        pattern_str (str): Pattern wanted as a string, with glob's convention.
        extension (str): Extension wanted, optional. With or without point. (`yaml` or `.yaml` accepted)
        filename_only (bool): Get only the filename
        get_list (bool): Get the whole list of matching files
        exact_name (bool): Get the exact name (without adding `*` before and after the given pattern)

    Returns:
        Union[str, list]: File
    """
    # Glob pattern
    if exact_name:
        glob_pattern = pattern_str
    else:
        glob_pattern = '*' + pattern_str + '*'
    if extension:
        if not extension.startswith("."):
            extension = '.' + extension
        glob_pattern += extension

    # Search for the pattern in the directory
    file_list = glob.glob(os.path.join(directory, glob_pattern))

    if len(file_list) == 0:
        raise FileNotFoundError("File with pattern {} not found in {}".format(glob_pattern, directory))

    # Return list, file path or file name
    if get_list:
        file = file_list
    else:
        if len(file_list) > 1:
            LOGGER.warning("More than one file corresponding to the pattern %s has been found here %s. "
                           "Only the first item will be returned.", glob_pattern, directory)
        file = file_list[0]
        if filename_only:
            file = os.path.basename(file)

    return file


# pylint: disable=E1121
def hash_file_content(file_content: str, len_param: int = 5) -> str:
    """
    Hash a file into a unique str.

    Args:
        file_content (str): File content
        len_param (int): Length parameter for the hash (length of the key will be 2x this number)

    Returns:
        str: Hashed file content
    """
    hasher = hashlib.shake_256()
    hasher.update(str.encode(file_content))
    return hasher.hexdigest(len_param)
