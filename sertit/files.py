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

import dill
import numpy as np
from lxml import etree, html
from tqdm import tqdm

from sertit import AnyPath, logs, path
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


def extract_file(
    file_path: AnyPathStrType,
    output: AnyPathStrType,
    overwrite: bool = False,
) -> AnyPathType:
    """
    Extract an archived file (zip or others). Overwrites if specified.
    If the archive don't contain a root directory with the name of the archive without the extension, create it

    Args:
        file_path (str): Archive file path
        output (str): Output where to put the extracted directory
        overwrite (bool): Overwrite found extracted directory

    Returns:
        AnyPathType: Extracted directory paths

    Example:
        >>> file_path = 'D:/path/to/zip.zip'
        >>> output = 'D:/path/to/output'
        >>> extract_file(file_path, output, overwrite=True)
        D:/path/to/output/zip'
    """
    # Convert to path
    file_path = AnyPath(file_path)
    output = AnyPath(output)

    # In case a folder is given, returns it (this means that the file is already extracted)
    if file_path.is_dir():
        return file_path

    # Beware with .SEN3 and .SAFE extensions
    archive_output = output.joinpath(path.get_filename(file_path))

    # In case not overwrite and the extracted directory already exists
    if not overwrite and archive_output.exists():
        LOGGER.debug(
            "Already existing extracted %s. It won't be overwritten.",
            archive_output,
        )
        return archive_output

    def extract_sub_dir(arch, filename_list):
        top_level_files = list({item.split("/")[0] for item in filename_list})

        # When the only root directory in the archive has the right name, we don't have to create it
        if len(top_level_files) == 1 and archive_output.name == path.get_filename(
            top_level_files[0]
        ):
            arch.extractall(archive_output.parent)
            archive_output.parent.joinpath(top_level_files[0]).rename(archive_output)
        else:
            arch.extractall(archive_output)

    # Manage archive type
    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path, "r") as zip_file:
            extract_sub_dir(zip_file, zip_file.namelist())
    elif file_path.suffix == ".tar" or file_path.suffixes == [".tar", ".gz"]:
        with tarfile.open(file_path, "r") as tar_file:
            extract_sub_dir(tar_file, tar_file.getnames())
    elif file_path.suffix == ".7z":
        try:
            import py7zr

            with py7zr.SevenZipFile(file_path, "r") as z7_file:
                extract_sub_dir(z7_file, z7_file.getnames())
        except ModuleNotFoundError as exc:
            raise TypeError("Please install 'py7zr' to extract .7z files") from exc
    else:
        raise TypeError(
            f"Only .zip, .tar, .tar.gz and .7z files can be extracted, not {file_path}"
        )

    return archive_output


def extract_files(
    archives: list, output: AnyPathStrType, overwrite: bool = False
) -> list:
    """
    Extract all archived files. Overwrites if specified.

    Example:
        >>> file_path = ['D:/path/to/zip1.zip', 'D:/path/to/zip2.zip']
        >>> output = 'D:/path/to/output'
        >>> extract_files(file_path, output, overwrite=True)
        ['D:/path/to/output.zip1', 'D:/path/to/output.zip2']

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


def get_archived_file_list(archive_path: AnyPathStrType) -> list:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get the list of all the files contained in an archive.

    Args:
        archive_path (AnyPathStrType): Archive path

    Returns:
        list: All files contained in the given archive

    Example:
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> get_archived_file_list(arch_path, file_regex)
        ['file_1.txt', 'file_2.tif', 'file_3.xml', 'file_4.geojson']
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_archived_file_list(archive_path)


def get_archived_path(
    archive_path: AnyPathStrType, file_regex: str, as_list: bool = False
) -> Union[list, AnyPathType]:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get archived file path from inside the archive.

    .. WARNING::
        If :code:`as_list` is :code:`False`, it will only return the first file matched !

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Example:
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> file_regex = '.*dir.*file_name'  # Use .* for any character
        >>> path = get_archived_path(arch_path, file_regex)
        'dir/filename.tif'

    Args:
        archive_path (AnyPathStrType): Archive path
        file_regex (str): File regex (used by re) as it can be found in the getmembers() list
        as_list (bool): If true, returns a list (including all found files). If false, returns only the first match

    Returns:
        Union[list, str]: Path from inside the zipfile
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_archived_path(archive_path, file_regex, as_list)


def get_archived_rio_path(
    archive_path: AnyPathStrType, file_regex: str, as_list: bool = False
) -> Union[list, AnyPathType]:
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.path` instead of :py:mod:`sertit.files`

    Get archived file path from inside the archive, to be read with rasterio:

    - :code:`zip+file://{zip_path}!{file_name}`
    - :code:`tar+file://{tar_path}!{file_name}`


    See `here <https://rasterio.readthedocs.io/en/latest/topics/datasets.html?highlight=zip#dataset-identifiers>`_
    for more information.

    .. WARNING::
        It won't be readable by pandas, geopandas or xmltree !

    .. WARNING::
        If :code:`as_list` is :code:`False`, it will only return the first file matched !

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Args:
        archive_path (AnyPathStrType): Archive path
        file_regex (str): File regex (used by re) as it can be found in the getmembers() list
        as_list (bool): If true, returns a list (including all found files). If false, returns only the first match

    Returns:
        Union[list, str]: Band path that can be read by rasterio

    Example:
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> file_regex = '.*dir.*file_name'  # Use .* for any character
        >>> path = get_archived_tif_path(arch_path, file_regex)
        'zip+file://D:/path/to/output.zip!dir/filename.tif'
        >>> rasterio.open(path)
        <open DatasetReader name='zip+file://D:/path/to/output.zip!dir/filename.tif' mode='r'>
    """
    logs.deprecation_warning(
        "This function is deprecated. Import it from 'sertit.path' instead of 'sertit.files'"
    )
    return path.get_archived_rio_path(archive_path, file_regex, as_list)


def read_archived_file(
    archive_path: AnyPathStrType, regex: str, file_list: list = None
) -> bytes:
    """
    Read archived file (in bytes) from :code:`zip` or :code:`tar` archives.

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Args:
        archive_path (AnyPathStrType): Archive path
        regex (str): Regex (used by re) as it can be found in the getmembers() list
        file_list (list): List of files contained in the archive. Optional, if not given it will be re-computed.

    Returns:
         bytes: Archived file in bytes
    """
    archive_path = AnyPath(archive_path)

    # Compile regex
    regex = re.compile(regex)

    # Open tar and zip XML
    try:
        if archive_path.suffix == ".tar":
            with tarfile.open(archive_path) as tar_ds:
                # file_list is not very useful for TAR files...
                if file_list is None:
                    tar_mb = tar_ds.getmembers()
                    file_list = [mb.name for mb in tar_mb]
                name = list(filter(regex.match, file_list))[0]
                tarinfo = tar_ds.getmember(name)
                file_str = tar_ds.extractfile(tarinfo).read()
        elif archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as zip_ds:
                if file_list is None:
                    file_list = [f.filename for f in zip_ds.filelist]
                name = list(filter(regex.match, file_list))[0]
                file_str = zip_ds.read(name)

        elif archive_path.suffix == ".tar.gz":
            raise TypeError(
                ".tar.gz files are too slow to read from inside the archive. Please extract them instead."
            )
        else:
            raise TypeError(
                "Only .zip and .tar files can be read from inside its archive."
            )
    except IndexError as exc:
        raise FileNotFoundError(
            f"Impossible to find file {regex} in {path.get_filename(archive_path)}"
        ) from exc

    return file_str


def read_archived_xml(
    archive_path: AnyPathStrType, regex: str = None, file_list: list = None, **kwargs
) -> etree._Element:
    """
    Read archived XML from :code:`zip` or :code:`tar` archives.

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Args:
        archive_path (AnyPathStrType): Archive path
        regex (str): XML regex (used by re) as it can be found in the getmembers() list
        file_list (list): List of files contained in the archive. Optional, if not given it will be re-computed.

    Returns:
         etree._Element: XML file

    Example:
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> file_regex = '.*dir.*file_name'  # Use .* for any character
        >>> read_archived_xml(arch_path, file_regex)
        <Element LANDSAT_METADATA_FILE at 0x1c90007f8c8>
    """
    if regex is None:
        logs.deprecation_warning(
            "'xml_regex' is deprecated, please use 'regex' instead."
        )
        regex = kwargs.pop("xml_regex")

    xml_bytes = read_archived_file(archive_path, regex=regex, file_list=file_list)

    return etree.fromstring(xml_bytes)


def read_archived_html(
    archive_path: AnyPathStrType, regex: str, file_list: list = None
) -> html.HtmlElement:
    """
    Read archived HTML from :code:`zip` or :code:`tar` archives.

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Args:
        archive_path (AnyPathStrType): Archive path
        regex (str): HTML regex (used by re) as it can be found in the getmembers() list
        file_list (list): List of files contained in the archive. Optional, if not given it will be re-computed.

    Returns:
         html._Element: HTML file

    Example:
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> file_regex = '.*dir.*file_name'  # Use .* for any character
        >>> read_archived_html(arch_path, file_regex)
        <Element html at 0x1c90007f8c8>
    """
    html_bytes = read_archived_file(archive_path, regex, file_list=file_list)

    return html.fromstring(html_bytes)


def archive(
    folder_path: AnyPathStrType,
    archive_path: AnyPathStrType,
    fmt: str = "zip",
) -> AnyPathType:
    """
    Archives a folder recursively.

    Args:
        folder_path (AnyPathStrType): Folder to archive
        archive_path (AnyPathStrType): Archive path, with or without extension
        fmt (str): Format of the archive, used by :code:`shutil.make_archive`. Choose between [zip, tar, gztar, bztar, xztar]

    Returns:
        str: Archive filename

    Example:
        >>> folder_path = 'D:/path/to/folder_to_archive'
        >>> archive_path = 'D:/path/to/output'
        >>> archive = archive(folder_path, archive_path, fmt="gztar")
        'D:/path/to/output/folder_to_archive.tar.gz'
    """
    archive_path = AnyPath(archive_path)
    folder_path = AnyPath(folder_path)

    tmp_dir = None
    if path.is_cloud_path(folder_path):
        tmp_dir = tempfile.TemporaryDirectory()
        folder_path = folder_path.download_to(tmp_dir.name)

    # Shutil make_archive needs a path without extension
    archive_base = os.path.splitext(archive_path)[0]

    # Archive the folder
    archive_fn = shutil.make_archive(
        archive_base,
        format=fmt,
        root_dir=folder_path.parent,
        base_dir=folder_path.name,
    )

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return AnyPath(archive_fn)


def add_to_zip(
    zip_path: AnyPathStrType,
    dirs_to_add: Union[list, AnyPathStrType],
) -> AnyPathType:
    """
    Add folders to an already existing zip file (recursively).

    Args:
        zip_path (AnyPathStrType): Already existing zip file
        dirs_to_add (Union[list, AnyPathStrType]): Directories to add

    Returns:
        AnyPathType: Updated zip_path

    Example:
        >>> zip_path = 'D:/path/to/zip.zip'
        >>> dirs_to_add = ['D:/path/to/dir1', 'D:/path/to/dir2']
        >>> add_to_zip(zip_path, dirs_to_add)
        zip.zip contains 2 more folders, dir1 and dir2
    """
    zip_path = AnyPath(zip_path)

    # If the zip is on the cloud, cache it (zipfile doesn't like cloud paths)
    if path.is_cloud_path(zip_path):
        zip_path = AnyPath(zip_path.fspath)

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
        for dir_to_add_path in progress_bar:
            # Just to be sure, use str instead of Paths
            if isinstance(dir_to_add_path, Path):
                dir_to_add = str(dir_to_add_path)
            elif path.is_cloud_path(dir_to_add_path):
                dir_to_add = dir_to_add_path.fspath
            else:
                dir_to_add = dir_to_add_path

            progress_bar.set_description(
                f"Adding {os.path.basename(dir_to_add)} to {os.path.basename(zip_path)}"
            )
            tmp = tempfile.TemporaryDirectory()
            if os.path.isfile(dir_to_add):
                dir_to_add = extract_file(dir_to_add, tmp.name)

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

            # Clean tmp
            tmp.cleanup()

    return zip_path


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
        out = src.download_to(dst)
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
