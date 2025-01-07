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
"""Tools for paths"""

import errno
import logging
import os
import pprint
import re
import tarfile
import tempfile
import zipfile
from typing import Any, Union

from sertit import AnyPath, logs
from sertit.logs import SU_NAME
from sertit.types import AnyPathStrType, AnyPathType

LOGGER = logging.getLogger(SU_NAME)


def get_root_path() -> AnyPathType:
    """
    Get the root path of the current disk:

    - On Linux this returns :code:`/`
    - On Windows this returns :code:`C:/` or whatever the current drive is

    Example:
        >>> get_root_path()
        "/" on Linux
        "C:/" on Windows (if you run this code from the C: drive)
    """
    return AnyPath(os.path.abspath(os.sep))


def listdir_abspath(directory: AnyPathStrType) -> list:
    """
    Get absolute path of all files in the given directory.

    It is the same function as :code:`os.listdir` but returning absolute paths.

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
    dirpath = AnyPath(directory)

    return list(dirpath.iterdir())


def to_abspath(
    raw_path: AnyPathStrType,
    create: bool = True,
    raise_file_not_found: bool = True,
) -> AnyPathType:
    """
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
        >>> )
    """
    abs_path = AnyPath(raw_path).resolve()

    if not abs_path.exists():
        if abs_path.suffix:
            if raise_file_not_found:
                # If the path specifies a file (with extension), it raises an exception
                raise FileNotFoundError(f"Non existing file: {abs_path}")

        # If the path specifies a folder, it creates it
        elif create:
            abs_path.mkdir()

    return abs_path


def real_rel_path(raw_path: AnyPathStrType, start: AnyPathStrType) -> AnyPathType:
    """
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
    raw_path = AnyPath(raw_path)
    start = AnyPath(start)
    if not is_cloud_path(raw_path) and not is_cloud_path(start):
        rel_path = AnyPath(os.path.relpath(raw_path.parent, start), raw_path.name)
    else:
        rel_path = raw_path

    return rel_path


def get_archived_file_list(archive_path: AnyPathStrType) -> list:
    """
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
            raise tarfile.ReadError(
                f"Impossible to open archive: {archive_path}"
            ) from ex

    return file_list


def get_archived_path(
    archive_path: AnyPathStrType,
    regex: str,
    as_list: bool = False,
    case_sensitive: bool = False,
    file_list: list = None,
    **kwargs,
) -> Union[list, AnyPathType]:
    """
    Get archived file path from inside the archive.

    .. WARNING::
        If :code:`as_list` is :code:`False`, it will only return the first file matched !

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Args:
        archive_path (AnyPathStrType): Archive path
        regex (str): File regex (used by re) as it can be found in the getmembers() list
        as_list (bool): If true, returns a list (including all found files). If false, returns only the first match
        case_sensitive (bool): If true, the regex is case-sensitive.
        file_list (list): List of files to get archived from. Optional, if not given it will be re-computed.

    Returns:
        Union[list, str]: Path from inside the zipfile

    Example:
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> file_regex = '.*dir.*file_name'  # Use .* for any character
        >>> path = get_archived_path(arch_path, file_regex)
        'dir/filename.tif'
    """
    if regex is None:
        logs.deprecation_warning(
            "'file_regex' is deprecated, please use 'regex' instead."
        )
        regex = kwargs.pop("file_regex")

    # Get file list
    archive_path = AnyPath(archive_path)

    # Offer the ability to give the file list directly, as this operation is expensive when done with large archives stored on the cloud
    if file_list is None:
        file_list = get_archived_file_list(archive_path)

    # Search for file
    re_rgx = re.compile(regex) if case_sensitive else re.compile(regex, re.IGNORECASE)
    archived_band_paths = list(filter(re_rgx.match, file_list))
    if not archived_band_paths:
        raise FileNotFoundError(
            f"Impossible to find file {regex} in {get_filename(archive_path)}"
        )

    # Convert to str if needed
    if not as_list:
        archived_band_paths = archived_band_paths[0]

    return archived_band_paths


def get_archived_rio_path(
    archive_path: AnyPathStrType,
    regex: str,
    as_list: bool = False,
    file_list: list = None,
    **kwargs,
) -> Union[list, AnyPathType]:
    """
    Get archived file path from inside the archive, to be read with rasterio:

    - :code:`zip+file://{zip_path}!{file_name}`
    - :code:`tar+file://{tar_path}!{file_name}`


    See `here <https://rasterio.readthedocs.io/en/latest/topics/datasets.html?highlight=zip#dataset-identifiers>`_
    for more information.

    .. WARNING::
        It wont be readable by pandas, geopandas or xmltree !

    .. WARNING::
        If :code:`as_list` is :code:`False`, it will only return the first file matched !

    You can use this `site <https://regexr.com/>`_ to build your regex.

    Args:
        archive_path (AnyPathStrType): Archive path
        regex (str): File regex (used by re) as it can be found in the getmembers() list
        as_list (bool): If true, returns a list (including all found files). If false, returns only the first match
        file_list (list): List of files contained in the archive. Optional, if not given it will be re-computed.

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
    if regex is None:
        logs.deprecation_warning(
            "'file_regex' is deprecated, please use 'regex' instead."
        )
        regex = kwargs.pop("file_regex")

    archive_path = AnyPath(archive_path)
    if archive_path.suffix in [".tar", ".zip"]:
        prefix = archive_path.suffix[-3:]
    elif archive_path.suffix == ".tar.gz":
        raise TypeError(
            ".tar.gz files are too slow to be read from inside the archive. Please extract them instead."
        )
    else:
        raise TypeError("Only .zip and .tar files can be read from inside its archive.")

    # Search for file
    archived_band_paths = get_archived_path(
        archive_path, regex=regex, as_list=True, file_list=file_list
    )

    # Convert to rio path
    if is_cloud_path(archive_path):
        archived_band_paths = [
            f"{prefix}+file+{archive_path}!{path}" for path in archived_band_paths
        ]
    else:
        # archived_band_paths = [
        #     f"{prefix}+file://{archive_path}!{path}" for path in archived_band_paths
        # ]
        archived_band_paths = [
            f"/vsi{prefix}/{archive_path}/{path}" for path in archived_band_paths
        ]

    # Convert to str if needed
    if not as_list:
        archived_band_paths = archived_band_paths[0]

    return archived_band_paths


def get_filename(file_path: AnyPathStrType, other_exts: Union[list, str] = None) -> str:
    """
    Get file name (without extension) from file path, ie:

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
    file_path = AnyPath(file_path)

    # We need to avoid splitext because of nested extensions such as .tar.gz
    multi_exts = [".tar.gz", ".SAFE.zip", ".SEN3.zip"]

    if other_exts is not None:
        if not isinstance(other_exts, list):
            other_exts = [other_exts]

        multi_exts += other_exts

    if any([str(file_path).endswith(ext) for ext in multi_exts]):
        filename = file_path.name.split(".")[0]
    else:
        # Manage correctly the cases like HLS.L30.T42RVR.2022240T055634.v2.0.B01.tif files...
        filename = file_path.stem

    # get_archived_rio_path returns zip+file://{zip_path}!{file_name}
    if ".zip!" in filename:
        filename = filename.split("!")[1]

    return filename


def get_ext(file_path: AnyPathStrType) -> str:
    """
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
    file_path = AnyPath(file_path)

    # We need to avoid splitext because of nested extensions such as .tar.gz
    return ".".join(file_path.name.split(".")[1:])


def find_files(
    names: Union[list, str],
    root_paths: Union[list, AnyPathStrType],
    max_nof_files: int = -1,
    get_as_str: bool = False,
) -> Union[list, str]:
    """
    Returns matching files recursively from a list of root paths.

    Regex are allowed (using glob)

    Args:
        names (Union[list, str]): File names.
        root_paths (Union[list, str]): Root paths
        max_nof_files (int): Maximum number of files (set to -1 for unlimited)
        get_as_str (bool): if only one file is found, it can be retrieved as a string instead of a list

    Returns:
        list: File name

    Example:
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
        >>>
        >>> find_files("huhu.txt", root_path, max_nof_files=1, get_as_str=True)
        found = 'D:/root/dir1/huhu.txt'
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
    directory = AnyPath(directory)

    # Glob pattern
    glob_pattern = pattern_str if exact_name else "*" + pattern_str + "*"
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


def is_writable(dir_path: AnyPathStrType):
    """
    Determine whether the directory is writeable or not.

    Args:
        dir_path (AnyPathStrType): Directory path

    Returns:
        bool: True if the directory is writable
    """
    try:
        with tempfile.TemporaryFile(dir=str(dir_path)):
            pass
    except (OSError, FileNotFoundError) as e:
        if e.errno in [
            errno.EACCES,
            errno.EEXIST,
            errno.EROFS,
            errno.ENOENT,
            errno.EINVAL,
        ]:  # 2, 13, 17, 30, 22
            return False
        e.filename = dir_path
        raise
    return True


def is_cloud_path(path: AnyPathStrType):
    """
    Determine whether the path corresponds to a file stored on the cloud or not.

    Args:
        path (AnyPathStrType): File path

    Returns:
        bool: True if the file is store on the cloud.
    """
    try:
        from cloudpathlib import CloudPath

        return isinstance(AnyPath(path), CloudPath)
    except Exception:
        return False


def is_path(path: Any) -> bool:
    """
    Determine whether the path corresponds to a file stored on the cloud or not.

    Args:
        path (AnyPathStrType): File path

    Returns:
        bool: True if the file is store on the cloud.
    """
    from pathlib import Path

    from cloudpathlib import CloudPath

    return isinstance(path, (str, Path, CloudPath))
