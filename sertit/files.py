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
""" Tools for paths and files """

import hashlib
import json
import logging
import os
import pickle
import pprint
import re
import shutil
import tarfile
import tempfile
import zipfile
from datetime import date, datetime
from enum import Enum
from json import JSONDecoder, JSONEncoder
from pathlib import Path
from typing import Any, Union

import numpy as np
from cloudpathlib import AnyPath, CloudPath
from lxml import etree
from tqdm import tqdm

from sertit import misc
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def get_root_path() -> Union[CloudPath, Path]:
    """
    Get the root path of the current disk:

    - On Linux this returns `/`
    - On Windows this returns `C:\\` or whatever the current drive is

    ```python
    >>> get_root_path()
    "/" on Linux
    "C:\\" on Windows (if you run this code from the C: drive)
    ```
    """
    return AnyPath(os.path.abspath(os.sep))


def listdir_abspath(directory: Union[str, CloudPath, Path]) -> list:
    """
    Get absolute path of all files in the given directory.

    It is the same function than `os.listdir` but returning absolute paths.

    ```python
    >>> folder = "."
    >>> listdir_abspath(folder)
    ['D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\files.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\logs.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\misc.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\network.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\rasters_rio.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\strings.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\vectors.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\version.py',
    'D:\\_SERTIT_UTILS\\sertit-utils\\sertit\\__init__.py']
    ```

    Args:
        directory (Union[str, CloudPath, Path]): Relative or absolute path to the directory to be scanned

    Returns:
        str: Absolute path of all files in the given directory
    """
    dirpath = AnyPath(directory)

    return list(dirpath.iterdir())


def to_abspath(
    path: Union[str, CloudPath, Path], create: bool = True
) -> Union[CloudPath, Path]:
    """
    Return the absolute path of the specified path and check if it exists

    If not:

    - If it is a file (aka has an extension), it raises an exception
    - If it is a folder, it creates it

    To be used with argparse to retrieve the absolute path of a file, like:

    ```python
    >>> parser = argparse.ArgumentParser()
    >>> # Add config file path key
    >>> parser.add_argument("--config",
                            help="Config file path (absolute or relative)",
                            type=to_abspath)
    ```

    Args:
        path (Union[str, CloudPath, Path]): Path as a string (relative or absolute)
        create (bool): Create directory if not existing

    Returns:
        Union[CloudPath, Path]: Absolute path
    """
    abs_path = AnyPath(path).resolve()

    if not abs_path.exists():
        if abs_path.suffix:
            # If the path specifies a file (with extension), it raises an exception
            raise FileNotFoundError(f"Non existing file: {abs_path}")

        # If the path specifies a folder, it creates it
        if create:
            abs_path.mkdir()

    return abs_path


def real_rel_path(
    path: Union[str, CloudPath, Path], start: Union[str, CloudPath, Path]
) -> Union[CloudPath, Path]:
    """
    Gives the real relative path from a starting folder.
    (and not just adding `..\..` between the start and the target)

    ```python
    >>> path = r'D:\_SERTIT_UTILS\sertit-utils\sertit'
    >>> start = os.path.join(".", "..", "..")
    >>> real_rel_path(path, start)
    'sertit-utils\\sertit'
    ```

    Args:
        path (Union[str, CloudPath, Path]): Path to make relative
        start (Union[str, CloudPath, Path]): Start, the path being relative from this folder.

    Returns:
        Relative path
    """
    path = AnyPath(path)
    start = AnyPath(start)
    if not isinstance(path, CloudPath) and not isinstance(start, CloudPath):
        rel_path = AnyPath(os.path.relpath(path.parent, start), path.name)
    else:
        rel_path = path

    return rel_path


def extract_file(
    file_path: Union[str, CloudPath, Path],
    output: Union[str, CloudPath, Path],
    overwrite: bool = False,
) -> Union[list, CloudPath, Path]:
    """
    Extract an archived file (zip or others). Overwrites if specified.
    For zipfiles, in case of multiple folders archived, pay attention that what is returned is the first folder.

    ```python
    >>> file_path = 'D:\\path\\to\\zip.zip'
    >>> output = 'D:\\path\\to\\output'
    >>> extract_file(file_path, output, overwrite=True)
    D:\\path\\to\\output\zip'
    ```

    Args:
        file_path (str): Archive file path
        output (str): Output where to put the extracted file
        overwrite (bool): Overwrite found extracted files

    Returns:
        Union[list, CloudPath, Path]: Extracted file paths (as str if only one)
    """
    # Convert to path
    file_path = AnyPath(file_path)
    output = AnyPath(output)

    # In case a folder is given, returns it (this means that the file is already extracted)
    if file_path.is_dir():
        return file_path

    # Manage archive type
    if file_path.suffix == ".zip":
        # Manage the case with several directories inside one zipfile
        arch = zipfile.ZipFile(file_path, "r")
        extr_names = list({path.split("/")[0] for path in arch.namelist()})
    elif file_path.suffix == ".tar" or file_path.suffixes == [".tar", ".gz"]:
        # Tar files have no subdirectories, so create one
        extr_names = [get_filename(file_path)]
        arch = tarfile.open(file_path, "r")
    else:
        raise TypeError(
            f"Only .zip, .tar and .tar.gz files can be extracted, not {file_path}"
        )

    # Get extracted list
    extr_dirs = [output.joinpath(extr_name) for extr_name in extr_names]

    # Loop over basedirs from inside the archive
    for extr_dir in extr_dirs:
        extr_name = extr_dir.name
        # Manage overwriting
        if extr_dir.is_dir():
            if overwrite:
                LOGGER.debug(
                    "Already existing extracted %s. It will be overwritten as asked.",
                    extr_names,
                )
                remove(extr_dir)
            else:
                LOGGER.debug(
                    "Already existing extracted %s. It won't be overwritten.",
                    extr_names,
                )

        else:
            LOGGER.info("Extracting %s", extr_names)
            # Inside docker, extracting files is really slow -> copy the archive in a tmp directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                if misc.in_docker():
                    # Create a tmp directory
                    copy(file_path, tmp_dir)
                    file_path = os.path.join(tmp_dir, os.path.basename(file_path))
                    tmp_extr_output = tmp_dir

                    # Recreate dir with tmp output
                    tmp_extr_dir = os.path.join(tmp_extr_output, extr_name)
                else:
                    tmp_extr_output = output
                    tmp_extr_dir = extr_dir

                if str(file_path).endswith(".zip"):
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
                    raise TypeError(f"Impossible to extract {file_path}") from ex

                # Copy back if we are running inside docker and clean tmp dir
                if misc.in_docker():
                    copy(tmp_extr_dir, extr_dir)

    # Close archive
    arch.close()

    # Return str for compatibility reasons
    if len(extr_dirs) == 1:
        extr_dirs = extr_dirs[0]

    return extr_dirs


def extract_files(
    archives: list, output: Union[str, CloudPath, Path], overwrite: bool = False
) -> list:
    """
    Extract all archived files. Overwrites if specified.

    ```python
    >>> file_path = ['D:\\path\\to\\zip1.zip', 'D:\\path\\to\\zip2.zip']
    >>> output = 'D:\\path\\to\\output'
    >>> extract_files(file_path, output, overwrite=True)
    ['D:\\path\\to\\output\zip1', 'D:\\path\\to\\output\zip2']
    ```

    Args:
        archives (list of str): List of archives to be extracted
        output (str): Output folder where extracted files will be written
        overwrite (bool): Overwrite found extracted files

    Returns:
        list: Extracted files (even pre-existing ones)
    """
    LOGGER.info("Extracting products in %s", output)
    progress_bar = tqdm(archives)
    extracts = []
    for arch in progress_bar:
        progress_bar.set_description(f"Extracting product {os.path.basename(arch)}")
        extracts.append(extract_file(arch, output, overwrite))

    return extracts


def get_archived_file_list(archive_path: Union[str, CloudPath, Path]) -> list:
    """
    Get the list of all the files contained in an archive.

    ```python
    >>> arch_path = 'D:\\path\\to\\zip.zip'
    >>> get_archived_file_list(arch_path, file_regex)
    ['file_1.txt', 'file_2.tif', 'file_3.xml', 'file_4.geojson']
    ```

    Args:
        archive_path (Union[str, CloudPath, Path]): Archive path

    Returns:
        list: All files contained in the given archive
    """
    archive_path = AnyPath(archive_path)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zip_ds:
            file_list = [f.filename for f in zip_ds.filelist]
    else:
        try:
            with tarfile.open(archive_path) as tar_ds:
                tar_mb = tar_ds.getmembers()
                file_list = [mb.name for mb in tar_mb]
        except tarfile.ReadError as ex:
            raise TypeError(f"Impossible to open archive: {archive_path}") from ex

    return file_list


def get_archived_rio_path(
    archive_path: Union[str, CloudPath, Path], file_regex: str, as_list: bool = False
) -> Union[list, CloudPath, Path]:
    """
    Get archived file path from inside the archive, to be read with rasterio:

    - `zip+file://{zip_path}!{file_name}`
    - `tar+file://{tar_path}!{file_name}`

    See [here](https://rasterio.readthedocs.io/en/latest/topics/datasets.html?highlight=zip#dataset-identifiers)
    for more information.

    .. WARNING::
        It wont be readable by pandas, geopandas or xmltree !

    .. WARNING::
        If `as_list` is `False`, it will only return the first file matched !

    You can use this [site](https://regexr.com/) to build your regex.

    ```python
    >>> arch_path = 'D:\\path\\to\\zip.zip'
    >>> file_regex = '.*dir.*file_name'  # Use .* for any character
    >>> path = get_archived_tif_path(arch_path, file_regex)
    'zip+file://D:\\path\\to\\output\zip!dir/filename.tif'
    >>> rasterio.open(path)
    <open DatasetReader name='zip+file://D:\\path\\to\\output\zip!dir/filename.tif' mode='r'>
    ```

    Args:
        archive_path (Union[str, CloudPath, Path]): Archive path
        file_regex (str): File regex (used by re) as it can be found in the getmembers() list
        as_list (bool): If true, returns a list (including all found files). If false, returns only the first match

    Returns:
        Union[list, str]: Band path that can be read by rasterio
    """
    # Get file list
    archive_path = AnyPath(archive_path)
    file_list = get_archived_file_list(archive_path)

    if archive_path.suffix in [".tar", ".zip"]:
        prefix = archive_path.suffix[-3:]
    elif archive_path.suffix == ".tar.gz":
        raise TypeError(
            ".tar.gz files are too slow to be read from inside the archive. Please extract them instead."
        )
    else:
        raise TypeError("Only .zip and .tar files can be read from inside its archive.")

    # Search for file
    regex = re.compile(file_regex)
    archived_band_path = list(filter(regex.match, file_list))
    if not archived_band_path:
        raise FileNotFoundError(
            f"Impossible to find file {file_regex} in {get_filename(archive_path)}"
        )

    # Convert to rio path
    if isinstance(archive_path, CloudPath):
        archived_band_path = [
            f"{prefix}+file+{archive_path}!{path}" for path in archived_band_path
        ]
    else:
        archived_band_path = [
            f"{prefix}+file://{archive_path}!{path}" for path in archived_band_path
        ]

    # Convert to str if needed
    if not as_list:
        archived_band_path = archived_band_path[0]

    return archived_band_path


def read_archived_xml(
    archive_path: Union[str, CloudPath, Path], xml_regex: str
) -> etree._Element:
    """
    Read archived XML from `zip` or `tar` archives.

    You can use this [site](https://regexr.com/) to build your regex.

    ```python
    >>> arch_path = 'D:\\path\\to\\zip.zip'
    >>> file_regex = '.*dir.*file_name'  # Use .* for any character
    >>> read_archived_xml(arch_path, file_regex)
    <Element LANDSAT_METADATA_FILE at 0x1c90007f8c8>
    ```

    Args:
        archive_path (Union[str, CloudPath, Path]): Archive path
        xml_regex (str): XML regex (used by re) as it can be found in the getmembers() list

    Returns:
         etree._Element: XML file
    """
    archive_path = AnyPath(archive_path)

    # Compile regex
    regex = re.compile(xml_regex)

    # Open tar and zip XML
    try:
        if archive_path.suffix == ".tar":
            with tarfile.open(archive_path) as tar_ds:
                tar_mb = tar_ds.getmembers()
                name_list = [mb.name for mb in tar_mb]
                band_name = list(filter(regex.match, name_list))[0]
                tarinfo = [mb for mb in tar_mb if mb.name == band_name][0]
                xml_str = tar_ds.extractfile(tarinfo).read()
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as zip_ds:
                name_list = [f.filename for f in zip_ds.filelist]
                band_name = list(filter(regex.match, name_list))[0]
                xml_str = zip_ds.read(band_name)

        elif archive_path.suffix == ".tar.gz":
            raise TypeError(
                ".tar.gz files are too slow to read from inside the archive. Please extract them instead."
            )
        else:
            raise TypeError(
                "Only .zip and .tar files can be read from inside its archive."
            )
    except IndexError:
        raise FileNotFoundError(
            f"Impossible to find XML file {xml_regex} in {get_filename(archive_path)}"
        )

    return etree.fromstring(xml_str)


def archive(
    folder_path: Union[str, CloudPath, Path],
    archive_path: Union[str, CloudPath, Path],
    fmt: str = "zip",
) -> Union[CloudPath, Path]:
    """
    Archives a folder recursively.

    ```python
    >>> folder_path = 'D:\\path\\to\\folder_to_archive'
    >>> archive_path = 'D:\\path\\to\\output'
    >>> archive = archive(folder_path, archive_path, fmt="gztar")
    'D:\\path\\to\\output\\folder_to_archive.tar.gz'
    ```

    Args:
        folder_path (Union[str, CloudPath, Path]): Folder to archive
        archive_path (Union[str, CloudPath, Path]): Archive path, with or without extension
        fmt (str): Format of the archive, used by `shutil.make_archive`. Choose between [zip, tar, gztar, bztar, xztar]
    Returns:
        str: Archive filename
    """
    archive_path = AnyPath(archive_path)
    folder_path = AnyPath(folder_path)

    # Shutil make_archive needs a path without extension
    archive_base = os.path.splitext(archive_path)[0]

    # Archive the folder
    archive_fn = shutil.make_archive(
        archive_base,
        format=fmt,
        root_dir=folder_path.parent,
        base_dir=folder_path.name,
    )

    return AnyPath(archive_fn)


def add_to_zip(
    zip_path: Union[str, CloudPath, Path],
    dirs_to_add: Union[list, str, CloudPath, Path],
) -> None:
    """
    Add folders to an already existing zip file (recursively).

    ```python
    >>> zip_path = 'D:\\path\\to\\zip.zip'
    >>> dirs_to_add = ['D:\\path\\to\\dir1', 'D:\\path\\to\\dir2']
    >>> add_to_zip(zip_path, dirs_to_add)
    >>> # zip.zip contains 2 more folders, dir1 and dir2
    ```

    Args:
        zip_path (Union[str, CloudPath, Path]): Already existing zip file
        dirs_to_add (Union[list, str]): Directories to add
    """
    zip_path = AnyPath(zip_path)

    # Check if existing zipfile
    if not zip_path.is_file():
        raise FileNotFoundError(f"Non existing {zip_path}")

    # Convert to list if needed
    if not isinstance(dirs_to_add, list):
        dirs_to_add = [dirs_to_add]

    # Add all folders to the existing zip
    # Forced to use ZipFile because make_archive only works with one folder and not existing zipfile
    with zipfile.ZipFile(zip_path, "a") as zip_file:
        progress_bar = tqdm(dirs_to_add)
        for dir_to_add in progress_bar:
            progress_bar.set_description(
                f"Adding {os.path.basename(dir_to_add)} to {os.path.basename(zip_path)}"
            )
            for root, _, files in os.walk(dir_to_add):
                base_path = os.path.join(dir_to_add, "..")

                # Write dir (in namelist at least)
                zip_file.write(root, os.path.relpath(root, base_path))

                # Write files
                for file in files:
                    zip_file.write(
                        os.path.join(root, file),
                        os.path.relpath(
                            os.path.join(root, file), os.path.join(dir_to_add, "..")
                        ),
                    )


def get_filename(file_path: Union[str, CloudPath, Path]) -> str:
    """
    Get file name (without extension) from file path, ie:

    ```python
    >>> file_path = 'D:\\path\\to\\filename.zip'
    >>> get_file_name(file_path)
    'filename'
    ```

    Args:
        file_path (Union[str, CloudPath, Path]): Absolute or relative file path (the file doesn't need to exist)

    Returns:
        str: File name (without extension)
    """
    file_path = AnyPath(file_path)

    # We need to avoid splitext because of nested extensions such as .tar.gz
    return file_path.name.split(".")[0]


def remove(path: Union[str, CloudPath, Path]) -> None:
    """
    Deletes a file or a directory (recursively) using `shutil.rmtree` or `os.remove`.

    ```python
    >>> path_to_remove = 'D:\\path\\to\\remove'  # Could also be a file
    >>> remove(path_to_remove)
    path_to_remove deleted
    ```

    Args:
        path (Union[str, CloudPath, Path]): Path to be removed
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
    directory: Union[str, CloudPath, Path],
    name_with_wildcard: str = "*",
    extension: str = None,
) -> None:
    """
    Remove files corresponding to a pattern from a directory.

    ```python
    >>> directory = 'D:\\path\\to\\folder'
    >>> os.listdir(directory)
    ["huhu.exe", "blabla.geojson", "haha.txt", "blabla"]

    >>> remove(directory, "blabla*")
    >>> os.listdir(directory)
    ["huhu.exe", "haha.txt"] # Removes also directories

    >>> remove(directory, "*", extension="txt")
    >>> os.listdir(directory)
    ["huhu.exe"]
    ```

    Args:
        directory (Union[str, CloudPath, Path]): Directory where to find the files
        name_with_wildcard (str): Filename (wildcards accepted)
        extension (str): Extension wanted, optional. With or without point. (yaml or .yaml accepted)
    """
    directory = AnyPath(directory)
    if extension and not extension.startswith("."):
        extension = "." + extension

    file_list = directory.glob(name_with_wildcard + extension)
    for file in file_list:
        remove(file)


def copy(
    src: Union[str, CloudPath, Path], dst: Union[str, CloudPath, Path]
) -> Union[CloudPath, Path]:
    """
    Copy a file or a directory (recursively) with `copytree` or `copy2`.

    ```python
    >>> src = 'D:\\path\\to\\copy'
    >>> dst = 'D:\\path\\to\\output'
    >>> copy(src, dst)
    copydir 'D:\\path\\to\\output\\copy'

    >>> src = 'D:\\path\\to\\copy.txt'
    >>> dst = 'D:\\path\\to\\output\\huhu.txt'
    >>> copyfile = copy(src, dst)
    'D:\\path\\to\\output\\huhu.txt' but with the content of copy.txt
    ```

    Args:
        src (Union[str, CloudPath, Path]): Source Path
        dst (Union[str, CloudPath, Path]): Destination Path (file or folder)

    Returns:
        Union[CloudPath, Path]: New path
    """
    src = AnyPath(src)
    out = None
    try:
        if src.is_dir():
            out = AnyPath(shutil.copytree(src, dst))
        elif os.path.isfile(src):
            out = AnyPath(shutil.copy2(src, dst))
    except shutil.Error:
        LOGGER.debug(exc_info=True)
        out = src
        # eg. source or destination doesn't exist
    except IOError as ex:
        raise IOError(f"Copy error: {ex.strerror}") from ex

    return out


def find_files(
    names: Union[list, str],
    root_paths: Union[list, str, CloudPath, Path],
    max_nof_files: int = -1,
    get_as_str: bool = False,
) -> Union[list, str]:
    """
    Returns matching files recursively from a list of root paths.

    Regex are allowed (using glob)

    ```python
    >>> root_path = 'D:\\root'
    >>> dir1_path = 'D:\\root\\dir1'
    >>> dir2_path = 'D:\\root\\dir2'

    >>> os.listdir(dir1_path)
    ["haha.txt", "huhu.txt", "hoho.txt"]
    >>> os.listdir(dir2_path)
    ["huhu.txt", "hehe.txt"]

    >>> find_files("huhu.txt", root_path)
    ['D:\\root\\dir1\\huhu.txt', 'D:\\root\\dir2\\huhu.txt']

    >>> find_files("huhu.txt", root_path, max_nof_files=1)
    ['D:\\root\\dir1\\huhu.txt']

    >>> find_files("huhu.txt", root_path, max_nof_files=1, get_as_str=True)
    found = 'D:\\root\\dir1\\huhu.txt'
    ```

    Args:
        names (Union[list, str]): File names.
        root_paths (Union[list, str]): Root paths
        max_nof_files (int): Maximum number of files (set to -1 for unlimited)
        get_as_str (bool): if only one file is found, it can be retrieved as a string instead of a list

    Returns:
        list: File name
    """
    paths = []

    # Transform to list
    if not isinstance(names, list):
        names = [names]

    if not isinstance(root_paths, list):
        root_paths = [root_paths]

    try:
        for root_path in root_paths:
            root_path = AnyPath(root_path)
            for name in names:
                paths += list(root_path.glob(f"**/*{name}*"))

    except StopIteration:
        pass

    # Check if found
    if not paths:
        raise FileNotFoundError(f"Files {names} not found in {root_paths}")

    if max_nof_files > 0:
        paths = paths[:max_nof_files]

    LOGGER.debug(
        "Paths found in %s for filenames %s:\n%s",
        root_paths,
        names,
        pprint.pformat(paths),
    )

    # Get str if needed
    if len(paths) == 1 and get_as_str:
        paths = paths[0]

    return paths


# subclass JSONDecoder
class CustomDecoder(JSONDecoder):
    """Decoder for JSON with methods for datetimes"""

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
        elif isinstance(obj, (CloudPath, Path)):
            out = str(obj)
        else:
            out = json.JSONEncoder.default(self, obj)

        return out


def read_json(json_file: Union[str, CloudPath, Path], print_file: bool = True) -> dict:
    """
    Read a JSON file

    ```python
    >>> json_path = 'D:\\path\\to\\json.json'
    >>> read_json(json_path, print_file=False)
    {"A": 1, "B": 2}
    ```

    Args:
        json_file (Union[str, CloudPath, Path]): Path to JSON file
        print_file (bool):  Print the configuration file

    Returns:
        dict: JSON data
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


def save_json(output_json: Union[str, CloudPath, Path], json_dict: dict) -> None:
    """
    Save a JSON file, with datetime, numpy types and Enum management.

    ```python
    >>> output_json = 'D:\\path\\to\\json.json'
    >>> json_dict = {"A": np.int64(1), "B": datetime.today(), "C": SomeEnum.some_name}
    >>> save_json(output_json, json_dict)
    ```

    Args:
        output_json (Union[str, CloudPath, Path]): Output file
        json_dict (dict): Json dictionary
    """

    with open(output_json, "w") as output_config_file:
        json.dump(json_dict, output_config_file, indent=3, cls=CustomEncoder)


def save_obj(obj: Any, path: Union[str, CloudPath, Path]) -> None:
    """
    Save an object as a pickle (can save any Python objects).

    ```python
    >>> output_pkl = 'D:\\path\\to\\pickle.pkl'
    >>> pkl_dict = {"A": np.ones([3, 3]),
                    "B": datetime.today(),
                    "C": SomeEnum.some_name}
    >>> save_json(output_pkl, pkl_dict)
    ```

    Args:
        obj (Any): Any object serializable
        path (Union[str, CloudPath, Path]): Path where to write the pickle
    """
    with open(path, "wb+") as file:
        pickle.dump(obj, file)


def load_obj(path: Union[str, CloudPath, Path]) -> Any:
    """
    Load a pickled object.

    ```python
    >>> output_pkl = 'D:\\path\\to\\pickle.pkl'
    >>> load_obj(output_pkl)
    {"A": np.ones([3, 3]), "B": datetime.today(), "C": SomeEnum.some_name}
    ```

    Args:
        path (Union[str, CloudPath, Path]): Path of the pickle
    Returns:
        object (Any): Pickled object

    """
    with open(path, "rb") as file:
        return pickle.load(file)


# too many arguments
# pylint: disable=R0913
def get_file_in_dir(
    directory: Union[str, CloudPath, Path],
    pattern_str: str,
    extension: str = None,
    filename_only: bool = False,
    get_list: bool = False,
    exact_name: bool = False,
) -> Union[CloudPath, Path, list]:
    """
    Get one or all matching files (pattern + extension) from inside a directory.

    Note that the pattern is a regex with glob's convention, ie. `*pattern*`.

    If `exact_name` is `False`, the searched pattern will be `*{pattern}*.{extension}`, else `{pattern}.{extension}`.

    ```python
    >>> directory = 'D:\\path\\to\\dir'
    >>> os.listdir(directory)
    ["haha.txt", "huhu1.txt", "huhu1.geojson", "hoho.txt"]

    >>> get_file_in_dir(directory, "huhu")
    'D:\\path\\to\\dir\\huhu1.geojson'

    >>> get_file_in_dir(directory, "huhu", extension="txt")
    'D:\\path\\to\\dir\\huhu1.txt'

    >>> get_file_in_dir(directory, "huhu", get_list=True)
    ['D:\\path\\to\\dir\\huhu1.txt', 'D:\\path\\to\\dir\\huhu1.geojson']

    >>> get_file_in_dir(directory, "huhu", filename_only=True, get_list=True)
    ['huhu1.txt', 'huhu1.geojson']

    >>> get_file_in_dir(directory, "huhu", get_list=True, exact_name=True)
    []
    ```

    Args:
        directory (str): Directory where to find the files
        pattern_str (str): Pattern wanted as a string, with glob's convention.
        extension (str): Extension wanted, optional. With or without point. (`yaml` or `.yaml` accepted)
        filename_only (bool): Get only the filename
        get_list (bool): Get the whole list of matching files
        exact_name (bool): Get the exact name (without adding `*` before and after the given pattern)

    Returns:
        Union[CloudPath, Path, list]: File
    """
    directory = AnyPath(directory)

    # Glob pattern
    if exact_name:
        glob_pattern = pattern_str
    else:
        glob_pattern = "*" + pattern_str + "*"
    if extension:
        if not extension.startswith("."):
            extension = "." + extension
        glob_pattern += extension

    # Search for the pattern in the directory
    file_list = list(directory.glob(glob_pattern))

    if len(file_list) == 0:
        raise FileNotFoundError(
            f"File with pattern {glob_pattern} not found in {directory}"
        )

    # Return list, file path or file name
    if get_list:
        file = file_list
    else:
        if len(file_list) > 1:
            LOGGER.warning(
                "More than one file corresponding to the pattern %s has been found here %s. "
                "Only the first item will be returned.",
                glob_pattern,
                directory,
            )
        file = file_list[0]
        if filename_only:
            file = file.name

    return file


# pylint: disable=E1121
def hash_file_content(file_content: str, len_param: int = 5) -> str:
    """
    Hash a file into a unique str.

    ```python
    >>> read_json("path\\to\\json.json")
    {"A": 1, "B": 2}

    >>> hash_file_content(str(file_content))
    "d3fad5bdf9"
    ```

    Args:
        file_content (str): File content
        len_param (int): Length parameter for the hash (length of the key will be 2x this number)

    Returns:
        str: Hashed file content
    """
    hasher = hashlib.shake_256()
    hasher.update(str.encode(file_content))
    return hasher.hexdigest(len_param)
