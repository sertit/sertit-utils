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
from enum import Enum
from json import JSONDecoder, JSONEncoder
from datetime import date, datetime
from typing import Union, Any
from tqdm import tqdm
import numpy as np

from sertit_utils.core import sys_utils
from sertit_utils.core.log_utils import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def get_root_path():
    """
    Get the root path of the current disk:

    - On Linux this returns `/`
    - On Windows this returns `C:\\\\` or whatever the current drive is
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

    ```python
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


def extract_file(file_path: str, output: str, overwrite: bool = False) -> Union[list, str]:
    """
    Extract an archived file (zip or others). Overwrites if specified.
    For zipfiles, in case of multiple folders archived, pay attention that what is returned is the first folder.

    Args:
        file_path (str): Archive file path
        output (str): Output where to put the extracted file
        overwrite (bool): Overwrite found extracted files

    Returns:
        Union[list, str]: Extracted file paths (as str if only one)
    """
    # Get extracted names
    if file_path.endswith(".zip"):
        # Manage the case with several directories inside one zipfile
        arch = zipfile.ZipFile(file_path, "r")
        extr_names = list({path.split("/")[0] for path in arch.namelist()})
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tar"):
        # Tar files have no subdirectories, so create one
        extr_names = [get_file_name(file_path)]
        arch = tarfile.open(file_path, "r")
    else:
        raise TypeError("ExtractEO can only extract {}".format(file_path))

    # Get extracted directory
    extr_dirs = [os.path.join(output, extr_name) for extr_name in extr_names]

    # Loop over basedirs from inside the archive
    for extr_dir in extr_dirs:
        extr_name = os.path.basename(extr_dir)
        # Manage overwriting
        if os.path.isdir(extr_dir):
            if overwrite:
                LOGGER.debug("Already existing extracted %s. It will be overwritten as asked.", extr_names)
                remove(extr_dir)
            else:
                LOGGER.debug("Already existing extracted %s. It won't be overwritten.", extr_names)

        else:
            LOGGER.info("Extracting %s", extr_names)
            # Inside docker, extracting files is really slow -> copy the archive in a tmp directory
            tmp = None
            if sys_utils.in_docker():
                # Create a tmp directory
                tmp = tempfile.TemporaryDirectory()
                copy(file_path, tmp.name)
                file_path = os.path.join(tmp.name, os.path.basename(file_path))
                tmp_extr_output = tmp.name

                # Recreate dir with tmp output
                tmp_extr_dir = os.path.join(tmp_extr_output, extr_name)
            else:
                tmp_extr_output = output
                tmp_extr_dir = extr_dir

            if file_path.endswith(".zip"):
                members = [name for name in arch.namelist() if extr_name in name]
            else:
                members = arch.getmembers()  # Always extract all files for TAR data

                # Tar files do not contain a file tree
                tmp_extr_output = tmp_extr_dir

            # Extract product
            try:
                os.makedirs(tmp_extr_dir, exist_ok=True)
                arch.extractall(path=tmp_extr_output, members=members)
            except tarfile.ReadError as ex:
                raise TypeError("Impossible to extract {}".format(file_path)) from ex

            # Copy back if we are running inside docker and clean tmp dir
            if tmp is not None:
                copy(tmp_extr_dir, extr_dir)
                tmp.cleanup()

    # Close archive
    arch.close()

    # Return str for compatibility reasons
    if len(extr_dirs) == 1:
        extr_dirs = extr_dirs[0]

    return extr_dirs


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
    LOGGER.info("Unzipping products in %s", output)
    progress_bar = tqdm(archives)
    extracts = []
    for arch in progress_bar:
        progress_bar.set_description('Unzipping products {}'.format(os.path.basename(arch)))
        extracts.append(extract_file(arch, output, overwrite))

    return extracts


def archive(folder_path: str, archive_path: str, fmt: str = "zip") -> str:
    """
    Archive a folder recursively.

    Args:
        folder_path (str): Folder to archive
        archive_path (str): Archive path, with or without extension
        fmt (str): Format of the archive, used by `shutil.make_archive`. Choose between [zip, tar, gztar, bztar, xztar]
    Returns:
        str: Archive filename
    """
    # Shutil make_archive needs a path without extension
    archive_base = os.path.splitext(archive_path)[0]

    # Archive the folder
    archive_fn = shutil.make_archive(archive_base,
                                     format=fmt,
                                     root_dir=os.path.dirname(folder_path),
                                     base_dir=os.path.basename(folder_path))

    return archive_fn


def add_to_zip(zip_path: str, dirs_to_add: Union[list, str]) -> None:
    """
    Add folders to an already existing zip file (recursively).

    Args:
        zip_path (str): Already existing zip file
        dirs_to_add (Union[list, str]): Directories to add
    """
    # Check if existing zipfile
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Non existing {zip_path}")

    # Convert to list if needed
    if isinstance(dirs_to_add, str):
        dirs_to_add = [dirs_to_add]

    # Add all folders to the existing zip
    # Forced to use ZipFile because make_archive only works with one folder and not existing zipfile
    with zipfile.ZipFile(zip_path, "a") as zip_file:
        progress_bar = tqdm(dirs_to_add)
        for dir_to_add in progress_bar:
            progress_bar.set_description('Adding {} to {}'.format(os.path.basename(dir_to_add),
                                                                  os.path.basename(zip_path)))
            for root, _, files in os.walk(dir_to_add):
                base_path = os.path.join(dir_to_add, '..')

                # Write dir (in namelist at least)
                zip_file.write(root, os.path.relpath(root, base_path))

                # Write files
                for file in files:
                    zip_file.write(os.path.join(root, file),
                                   os.path.relpath(os.path.join(root, file),
                                                   os.path.join(dir_to_add, '..')))


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
    Delete a file or a directory (recursively) using `shutil.rmtree` or `os.remove`.

    Args:
        path (str): Path to be removed
    """
    if not os.path.exists(path):
        LOGGER.debug("Non existing %s", path)

    elif os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            LOGGER.debug("Impossible to remove the directory %s", path, exc_info=True)

    elif os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            LOGGER.debug("Impossible to remove the file %s", path, exc_info=True)


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
    except shutil.Error:
        LOGGER.debug(exc_info=True)
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


# subclass JSONDecoder
class CustomDecoder(JSONDecoder):
    """ Decoder for JSON with methods for datetimes """

    # pylint: disable=W0221
    # Override the default method
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    # pylint: disable=E0202, R0201
    # - An attribute defined in json.decoder line 319 hides this method (method-hidden)
    # - Method could be a function (no-self-use)
    def object_hook(self, obj: Any):
        """
        Overload of object_hook function that deals with `datetime.datetime`

        Args:
            obj (dict): Dict containing objects to decode from JSON

        Returns:
            dict: Dict with decoded object
        """
        for key, val in obj.items():
            if isinstance(val, str):
                try:
                    # Date -> Encoder saves dates as isoformat
                    obj[key] = date.fromisoformat(val)
                except ValueError:
                    try:
                        # Datetime -> Encoder saves datetimes as isoformat
                        obj[key] = datetime.fromisoformat(val)
                    except ValueError:
                        obj[key] = val
            else:
                obj[key] = val
        return obj


# subclass JSONEncoder
class CustomEncoder(JSONEncoder):
    """ Encoder for JSON with methods for datetimes and np.int64 """

    # pylint: disable=W0221
    def default(self, obj):
        """ Overload of the default method """
        if isinstance(obj, (date, datetime)):
            out = obj.isoformat()
        elif isinstance(obj, np.int64):
            out = int(obj)
        elif isinstance(obj, Enum):
            out = obj.value
        else:
            out = json.JSONEncoder.default(self, obj)

        return out


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
        data = json.load(file, cls=CustomDecoder)
        if print_file:
            LOGGER.debug("Configuration file %s contains:\n%s",
                         json_file,
                         json.dumps(data, indent=3, cls=CustomEncoder))
    return data


def save_json(output_json: str, json_dict: dict) -> None:
    """
    Save a JSON file, with datetime and numpy types management.

    Args:
        output_json (str): Output file
        json_dict (dict): Json dictionary
    """

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
