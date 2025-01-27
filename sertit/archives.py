import logging
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Union

from lxml import etree, html
from tqdm import tqdm

from sertit import AnyPath, path, s3
from sertit.logs import SU_NAME
from sertit.types import AnyPathStrType, AnyPathType

LOGGER = logging.getLogger(SU_NAME)


@contextmanager
def open_zipfile(file_path, mode="r"):
    if path.is_cloud_path(file_path):
        file_path = s3.read(file_path)

    with zipfile.ZipFile(file_path, mode) as zip_file:
        yield zip_file


@contextmanager
def open_tarfile(file_path, mode="r"):
    if path.is_cloud_path(file_path):
        args = {"fileobj": s3.read(file_path), "mode": mode}
    else:
        args = {"name": file_path, "mode": mode}
    with tarfile.open(**args) as tar_file:
        yield tar_file


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
        with open_zipfile(file_path) as zip_file:
            extract_sub_dir(zip_file, zip_file.namelist())
    elif file_path.suffix == ".tar" or file_path.suffixes == [".tar", ".gz"]:
        with open_tarfile(file_path) as tar_file:
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
            with open_tarfile(archive_path) as tar_ds:
                # file_list is not very useful for TAR files...
                if file_list is None:
                    tar_mb = tar_ds.getmembers()
                    file_list = [mb.name for mb in tar_mb]
                name = list(filter(regex.match, file_list))[0]
                tarinfo = tar_ds.getmember(name)
                file_str = tar_ds.extractfile(tarinfo).read()
        elif archive_path.suffix == ".zip":
            with open_zipfile(archive_path) as zip_ds:
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

    # with zipfile.ZipFile(archive_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
    #     for f in folder_path.glob("**"):
    #         zipf.write(f, f.relative_to(folder_path.name))

    tmp_dir = None
    if path.is_cloud_path(folder_path):
        tmp_dir = tempfile.TemporaryDirectory()
        folder_path = s3.download(folder_path, tmp_dir.name)

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

    try:
        arch = AnyPath(archive_fn, **folder_path.storage_options)
    except AttributeError:
        arch = AnyPath(archive_fn)

    return arch


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

    with tempfile.TemporaryDirectory() as tmp_dir:
        # If the zip is on the cloud, cache it (zipfile doesn't like cloud paths)
        if path.is_cloud_path(zip_path):
            raise NotImplementedError(
                "Impossible (for now) to update a zip stored in the cloud!"
            )

        # Check if existing zipfile
        if not zip_path.is_file():
            raise FileNotFoundError(f"Non existing {zip_path}")

        # Convert to list if needed
        if not isinstance(dirs_to_add, list):
            dirs_to_add = [dirs_to_add]

        # Add all folders to the existing zip
        # Forced to use ZipFile because make_archive only works with one folder and not existing zipfile
        with open_zipfile(zip_path, "a") as zip_file:
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
                if os.path.isfile(dir_to_add):
                    dir_to_add = extract_file(dir_to_add, tmp_dir)

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

    return zip_path


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

    is_zip = archive_path.suffix == ".zip"
    archive_fn = path.get_filename(archive_path)
    if is_zip:
        with open_zipfile(archive_path) as zip_ds:
            file_list = [f.filename for f in zip_ds.filelist]
    else:
        try:
            with open_tarfile(archive_path) as tar_ds:
                tar_mb = tar_ds.getmembers()
                file_list = [mb.name for mb in tar_mb]
        except tarfile.ReadError as ex:
            raise tarfile.ReadError(f"Impossible to open archive: {archive_fn}") from ex

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
            f"Impossible to find file {regex} in {path.get_filename(archive_path)}"
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
    if path.is_cloud_path(archive_path):
        archived_band_paths = [
            f"{prefix}+file+{archive_path}!{p}" for p in archived_band_paths
        ]
    else:
        # archived_band_paths = [
        #     f"{prefix}+file://{archive_path}!{path}" for path in archived_band_paths
        # ]
        archived_band_paths = [
            f"/vsi{prefix}/{archive_path}/{p}" for p in archived_band_paths
        ]

    # Convert to str if needed
    if not as_list:
        archived_band_paths = archived_band_paths[0]

    return archived_band_paths
