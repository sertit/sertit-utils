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

import glob
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
from typing import Any, Union

import numpy as np
from lxml import etree
from tqdm import tqdm

from sertit import misc
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def get_root_path() -> str:
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
    return os.path.abspath(os.sep)


def listdir_abspath(directory: str) -> list:
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
        directory (str): Relative or absolute path to the directory to be scanned

    Returns:
        str: Absolute path of all files in the given directory
    """
    return [
        os.path.abspath(os.path.join(directory, file)) for file in os.listdir(directory)
    ]


def to_abspath(path_str: str, create: bool = True) -> str:
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
        path_str (str): Path as a string (relative or absolute)
        create (bool): Create directory if not existing

    Returns:
        str: Absolute path
    """
    abs_path = os.path.abspath(path_str)

    if not os.path.exists(abs_path):
        if os.path.splitext(abs_path)[1]:
            # If the path specifies a file (with extension), it raises an exception
            raise FileNotFoundError(f"Non existing file: {abs_path}")

        # If the path specifies a folder, it creates it
        if create:
            os.makedirs(abs_path)

    return abs_path


def real_rel_path(path: str, start: str) -> str:
    """
    Gives the real relative path from a starting folder.
    (and not just adding `..\..` between the start and the target)

    ```python
    >>> path = 'D:\\_SERTIT_UTILS\\sertit-utils\\sertit'
    >>> start = os.path.join("..", "..")
    >>> real_rel_path(path, start)
    'sertit-utils\\sertit'
    ```

    Args:
        path (str): Path to make relative
        start (str): Start, the path being relative from this folder.

    Returns:
        Relative path
    """
    return os.path.join(
        os.path.relpath(os.path.dirname(path), start), os.path.basename(path)
    )


def extract_file(
    file_path: str, output: str, overwrite: bool = False
) -> Union[list, str]:
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
        Union[list, str]: Extracted file paths (as str if only one)
    """
    # In case a folder is given, returns it (this means that the file is already extracted)
    if os.path.isdir(file_path):
        return file_path

    # Manage archive type
    if file_path.endswith(".zip"):
        # Manage the case with several directories inside one zipfile
        arch = zipfile.ZipFile(file_path, "r")
        extr_names = list({path.split("/")[0] for path in arch.namelist()})
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tar"):
        # Tar files have no subdirectories, so create one
        extr_names = [get_filename(file_path)]
        arch = tarfile.open(file_path, "r")
    else:
        raise TypeError(
            f"Only .zip, .tar and .tar.gz files can be extracted, not {file_path}"
        )

    # Get extracted list
    extr_dirs = [os.path.join(output, extr_name) for extr_name in extr_names]

    # Loop over basedirs from inside the archive
    for extr_dir in extr_dirs:
        extr_name = os.path.basename(extr_dir)
        # Manage overwriting
        if os.path.isdir(extr_dir):
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


def extract_files(archives: list, output: str, overwrite: bool = False) -> list:
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


def get_archived_file_list(archive_path) -> list:
    """
    Get the list of all the files contained in an archive.

    ```python
    >>> arch_path = 'D:\\path\\to\\zip.zip'
    >>> get_archived_file_list(arch_path, file_regex)
    ['file_1.txt', 'file_2.tif', 'file_3.xml', 'file_4.geojson']
    ```

    Args:
        archive_path (str): Archive path

    Returns:
        list: All files contained in the given archive
    """
    if archive_path.endswith(".zip"):
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
    archive_path: str, file_regex: str, as_list: bool = False
) -> Union[list, str]:
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
        archive_path (str): Archive path
        file_regex (str): File regex (used by re) as it can be found in the getmembers() list
        as_list (bool): If true, returns a list (including all found files). If false, returns only the first match

    Returns:
        Union[list, str]: Band path that can be read by rasterio
    """
    # Get file list
    file_list = get_archived_file_list(archive_path)

    if archive_path.endswith(".tar") or archive_path.endswith(".zip"):
        prefix = archive_path[-3:]
    elif archive_path.endswith(".tar.gz"):
        raise TypeError(
            ".tar.gz files are too slow to read from inside the archive. Please extract them instead."
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
    archived_band_path = [
        f"{prefix}+file://{archive_path}!{path}" for path in archived_band_path
    ]

    # Convert to str if needed
    if not as_list:
        archived_band_path = archived_band_path[0]

    return archived_band_path


def read_archived_xml(archive_path: str, xml_regex: str) -> etree._Element:
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
        archive_path (str): Archive path
        xml_regex (str): XML regex (used by re) as it can be found in the getmembers() list

    Returns:
         etree._Element: XML file
    """
    # Compile regex
    regex = re.compile(xml_regex)

    # Open tar and zip XML
    try:
        if archive_path.endswith(".tar"):
            with tarfile.open(archive_path) as tar_ds:
                tar_mb = tar_ds.getmembers()
                name_list = [mb.name for mb in tar_mb]
                band_name = list(filter(regex.match, name_list))[0]
                tarinfo = [mb for mb in tar_mb if mb.name == band_name][0]
                xml_str = tar_ds.extractfile(tarinfo).read()
        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as zip_ds:
                name_list = [f.filename for f in zip_ds.filelist]
                band_name = list(filter(regex.match, name_list))[0]
                xml_str = zip_ds.read(band_name)

        elif archive_path.endswith(".tar.gz"):
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


def read_archived_vector(archive_path: str, vector_regex: str):
    """
    Read archived vector from `zip` or `tar` archives.

    You can use this [site](https://regexr.com/) to build your regex.

    ```python
    >>> arch_path = 'D:\\path\\to\\zip.zip'
    >>> files.read_archived_vector(arch_path, ".*map-overlay\.kml")
                           Name  ...                                           geometry
    0  Sentinel-1 Image Overlay  ...  POLYGON ((0.85336 42.24660, -2.32032 42.65493,...
    ```

    Args:
        archive_path (str): Archive path
        vector_regex (str): Vector regex (used by re) as it can be found in the getmembers() list

    Returns:
        gpd.GeoDataFrame: Vector
    """
    # Import here as we don't want geopandas to pollute this file import
    import geopandas as gpd

    # Get file list
    file_list = get_archived_file_list(archive_path)
    if archive_path.endswith(".tar") or archive_path.endswith(".zip"):
        prefix = archive_path[-3:]
    elif archive_path.endswith(".tar.gz"):
        raise TypeError(
            ".tar.gz files are too slow to read from inside the archive. Please extract them instead."
        )
    else:
        raise TypeError("Only .zip and .tar files can be read from inside its archive.")

    # Open tar and zip vectors
    try:
        regex = re.compile(vector_regex)
        archived_vect_path = list(filter(regex.match, file_list))[0]
        archived_vect_path = f"{prefix}://{archive_path}!{archived_vect_path}"
        if archived_vect_path.endswith("kml"):
            from sertit import vectors

            vectors.set_kml_driver()

        vect = gpd.read_file(archived_vect_path)
    except IndexError:
        raise FileNotFoundError(
            f"Impossible to find vector {vector_regex} in {get_filename(archive_path)}"
        )

    return vect


def archive(folder_path: str, archive_path: str, fmt: str = "zip") -> str:
    """
    Archives a folder recursively.

    ```python
    >>> folder_path = 'D:\\path\\to\\folder_to_archive'
    >>> archive_path = 'D:\\path\\to\\output'
    >>> archive = archive(folder_path, archive_path, fmt="gztar")
    'D:\\path\\to\\output\\folder_to_archive.tar.gz'
    ```

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
    archive_fn = shutil.make_archive(
        archive_base,
        format=fmt,
        root_dir=os.path.dirname(folder_path),
        base_dir=os.path.basename(folder_path),
    )

    return archive_fn


def add_to_zip(zip_path: str, dirs_to_add: Union[list, str]) -> None:
    """
    Add folders to an already existing zip file (recursively).

    ```python
    >>> zip_path = 'D:\\path\\to\\zip.zip'
    >>> dirs_to_add = ['D:\\path\\to\\dir1', 'D:\\path\\to\\dir2']
    >>> add_to_zip(zip_path, dirs_to_add)
    >>> # zip.zip contains 2 more folders, dir1 and dir2
    ```

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


def get_filename(file_path: str) -> str:
    """
    Get file name (without extension) from file path, ie:

    ```python
    >>> file_path = 'D:\\path\\to\\filename.zip'
    >>> get_file_name(file_path)
    'filename'
    ```

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
    Deletes a file or a directory (recursively) using `shutil.rmtree` or `os.remove`.

    ```python
    >>> path_to_remove = 'D:\\path\\to\\remove'  # Could also be a file
    >>> remove(path_to_remove)
    path_to_remove deleted
    ```

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


def remove_by_pattern(
    directory: str, name_with_wildcard: str = "*", extension: str = None
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
        directory (str): Directory where to find the files
        name_with_wildcard (str): Filename (wildcards accepted)
        extension (str): Extension wanted, optional. With or without point. (yaml or .yaml accepted)
    """

    if extension and not extension.startswith("."):
        extension = "." + extension

    pattern_str = os.path.join(directory, name_with_wildcard + extension)
    file_list = glob.glob(pattern_str, recursive=False)
    for file in file_list:
        remove(file)


def copy(src: str, dst: str) -> str:
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
        raise IOError(f"Copy error: {ex.strerror}") from ex

    return out


def find_files(
    names: Union[list, str],
    root_paths: Union[list, str],
    max_nof_files: int = -1,
    get_as_str: bool = False,
) -> Union[list, str]:
    """
    Returns matching files recursively from a list of root paths.

    Regex are not allowed, only exact matches are looking for (using `os.walk()`)

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
        raise FileNotFoundError(f"Files {names} not found in {root_paths}")

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
        else:
            out = json.JSONEncoder.default(self, obj)

        return out


def read_json(json_file: str, print_file: bool = True) -> dict:
    """
    Read a JSON file

    ```python
    >>> json_path = 'D:\\path\\to\\json.json'
    >>> read_json(json_path, print_file=False)
    {"A": 1, "B": 2}
    ```

    Args:
        json_file (str): Path to JSON file
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


def save_json(output_json: str, json_dict: dict) -> None:
    """
    Save a JSON file, with datetime, numpy types and Enum management.

    ```python
    >>> output_json = 'D:\\path\\to\\json.json'
    >>> json_dict = {"A": np.int64(1), "B": datetime.today(), "C": SomeEnum.some_name}
    >>> save_json(output_json, json_dict)
    ```

    Args:
        output_json (str): Output file
        json_dict (dict): Json dictionary
    """

    with open(output_json, "w") as output_config_file:
        json.dump(json_dict, output_config_file, indent=3, cls=CustomEncoder)


def save_obj(obj: Any, path: str) -> None:
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
        path (str): Path where to write the pickle
    """
    with open(path, "wb+") as file:
        pickle.dump(obj, file)


def load_obj(path: str) -> Any:
    """
    Load a pickled object.

    ```python
    >>> output_pkl = 'D:\\path\\to\\pickle.pkl'
    >>> load_obj(output_pkl)
    {"A": np.ones([3, 3]), "B": datetime.today(), "C": SomeEnum.some_name}
    ```

    Args:
        path (str): Path of the pickle
    Returns:
        object (Any): Pickled object

    """
    with open(path, "rb") as file:
        return pickle.load(file)


# too many arguments
# pylint: disable=R0913
def get_file_in_dir(
    directory: str,
    pattern_str: str,
    extension: str = None,
    filename_only: bool = False,
    get_list: bool = False,
    exact_name: bool = False,
) -> Union[str, list]:
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
        Union[str, list]: File
    """
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
    file_list = glob.glob(os.path.join(directory, glob_pattern))

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
            file = os.path.basename(file)

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
