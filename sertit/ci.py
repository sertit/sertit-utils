# -*- coding: utf-8 -*-
# Copyright 2023, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""
CI tools

You can use :code:`assert_raster_equal` only if you have installed sertit[full] or sertit[rasters]
"""
import filecmp
import logging
import os
import pprint
from doctest import Example
from functools import wraps
from pathlib import Path
from typing import Any, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, CloudPath, S3Client
from lxml import etree, html
from lxml.doctestcompare import LHTMLOutputChecker, LXMLOutputChecker

from sertit import files, vectors
from sertit.logs import SU_NAME

AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_S3_ENDPOINT = "s3.unistra.fr"

LOGGER = logging.getLogger(SU_NAME)


def get_mnt_path() -> str:
    """
    Return mounting directory :code:`/mnt`.

    .. WARNING::
        This won't work on Windows !

    .. code-block:: python

        >>> get_mnt_path()
        '/mnt'

    Returns:
        str: Mounting directory
    """
    return r"/mnt"


def _get_db_path(db_nb=2) -> str:
    """
    Returns DSx database0x path

    - :code:`/mnt/ds2_dbx` when mounted (docker...)
    - :code:`\\ds2\database0x` on windows
    """
    db_path = f"{get_mnt_path()}/ds2_db{db_nb}"

    if not os.path.isdir(db_path):
        db_path = rf"\\DS2\database0{db_nb}"

    if not os.path.isdir(db_path):
        raise NotADirectoryError(f"Impossible to open ds2/database0{db_nb}!")
    return db_path


def get_db2_path() -> str:
    """
    Returns DS2 database02 path

    - :code:`/mnt/ds2_db2` when mounted (docker...)
    - :code:`\\ds2\database02` on windows

    .. code-block:: python

        >>> get_db2_path()
        '/mnt/ds2_db2'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(2)


def get_db3_path() -> str:
    """
    Returns DS2 database03 path

    - :code:`/mnt/ds2_db3` when mounted (docker...)
    - :code:`\\ds2\database03` on windows

    .. code-block:: python

        >>> get_db3_path()
        '/mnt/ds2_db3'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(3)


def get_db4_path() -> str:
    """
    Returns DS2 database04 path

    - :code:`/mnt/ds2_db4` when mounted (docker...)
    - :code:`\\ds2\database04` on windows

    Returns:
        str: Mounted directory
    """
    return _get_db_path(4)


def assert_val(val_1: Any, val_2: Any, field: str) -> None:
    """
    Compare two values corresponding to a field

    Args:
        val_1 (Any): Value 1
        val_2 (Any): Value 2
        field (str): Field to compare
    """
    assert val_1 == val_2, f"{field} incoherent:\n{val_1} != {val_2}"


def assert_field(dict_1: dict, dict_2: dict, field: str) -> None:
    """
    Compare two fields of a dictionary

    Args:
        dict_1 (dict): Dict 1
        dict_2 (dict): Dict 2
        field (str): Field to compare
    """
    assert_val(dict_1[field], dict_2[field], field)


def assert_files_equal(
    file_1: Union[str, Path, CloudPath], file_2: Union[str, Path, CloudPath]
):
    """
    Assert to files are equal by hashing its content

    Args:
        file_1 (str): Path to file 1
        file_2 (str): Path to file 2
    """
    if isinstance(file_1, CloudPath):
        file_1 = file_1.fspath

    if isinstance(file_2, CloudPath):
        file_2 = file_2.fspath

    with open(str(file_1), "r") as f1:
        with open(str(file_2), "r") as f2:
            assert files.hash_file_content(f1.read()) == files.hash_file_content(
                f2.read()
            )


def assert_meta(meta_1: dict, meta_2: dict, tf_precision: float = 1e-9):
    """
    Compare rasterio metadata

    Args:
        meta_1 (dict): Metadata 1
        meta_2 (dict): Metadata 2
        tf_precision (float): Transform precision (in transform units)
    """
    # Driver
    assert_field(meta_1, meta_2, "driver")
    assert_field(meta_1, meta_2, "dtype")
    assert_field(meta_1, meta_2, "nodata")
    assert_field(meta_1, meta_2, "width")
    assert_field(meta_1, meta_2, "height")
    assert_field(meta_1, meta_2, "count")
    assert_field(meta_1, meta_2, "crs")

    assert meta_1["transform"].almost_equals(
        meta_2["transform"], precision=tf_precision
    ), f'transform incoherent:\n{meta_1["transform"]}\n!=\n{meta_2["transform"]}'


def assert_raster_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path]
) -> None:
    """
    Assert that two rasters are equal.

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> assert_raster_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong

    Args:
        path_1 (Union[str, CloudPath, Path]): Raster 1
        path_2 (Union[str, CloudPath, Path]): Raster 2
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_equal."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1:
        with rasterio.open(str(path_2)) as ds_2:
            # Metadata
            assert_meta(ds_1.meta, ds_2.meta)

            # Assert equal
            np.testing.assert_array_equal(ds_1.read(), ds_2.read())


def assert_raster_almost_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path], decimal=7
) -> None:
    """
    Assert that two rasters are almost equal.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array and a precision of 10^-{decimal} on the transform

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_almost_equal(path, path2)
        >>> # Raises AssertionError if sth goes wrong

    Args:
        path_1 (Union[str, CloudPath, Path]): Raster 1
        path_2 (Union[str, CloudPath, Path]): Raster 2
        decimal (int): Number of decimal
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_almost_equal."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1:
        with rasterio.open(str(path_2)) as ds_2:
            # Metadata
            assert_meta(ds_1.meta, ds_2.meta, tf_precision=10**-decimal)

            # Assert almost equal
            errors = []
            for i in range(ds_1.count):

                desc = (
                    f": {ds_1.descriptions[i]}"
                    if ds_1.descriptions[i] is not None
                    else ""
                )
                LOGGER.info(f"Checking Band {i + 1}{desc}")
                try:
                    marr_1 = ds_1.read(i + 1)
                    marr_2 = ds_2.read(i + 1)
                    np.testing.assert_array_almost_equal(
                        marr_1, marr_2, decimal=decimal
                    )
                except AssertionError:
                    text = f"Band {i + 1}{desc} failed"
                    errors.append(text)
                    LOGGER.error(text, exc_info=True)

            if errors:
                raise AssertionError(errors)


def assert_raster_almost_equal_magnitude(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path], decimal=2
) -> None:
    """
    Assert that two rasters are almost equal, with the decimal taken on the scientif representation of the array.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array divided by the order of magnitude of the array and a precision of 10^-{decimal} on the transform

    i.e. `decimal=2, mean(array) = 15.2, true = 13.2687, false = 13.977, comparison: 1.32687, false <-> 1.3977 => false (2 != 9)`

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_almost_equal_magnitude(path, path2)
        >>> # Raises AssertionError if sth goes wrong

    Args:
        path_1 (Union[str, CloudPath, Path]): Raster 1
        path_2 (Union[str, CloudPath, Path]): Raster 2
        decimal (int): Number of decimal
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_almost_equal."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1:
        with rasterio.open(str(path_2)) as ds_2:
            # Metadata
            assert_meta(ds_1.meta, ds_2.meta, tf_precision=10**-decimal)

            # Assert almost equal
            errors = []
            for i in range(ds_1.count):

                desc = (
                    f": {ds_1.descriptions[i]}"
                    if ds_1.descriptions[i] is not None
                    else ""
                )
                LOGGER.info(f"Checking Band {i + 1}{desc}")
                try:
                    marr_1 = ds_1.read(i + 1)
                    marr_2 = ds_2.read(i + 1)

                    # Manage better the number of (decimals are for a magnitude of 0)
                    magnitude = np.floor(np.log10(abs(np.nanmedian(marr_1))))
                    if np.isinf(magnitude):
                        magnitude = 0

                    np.testing.assert_array_almost_equal(
                        marr_1 / 10**magnitude,
                        marr_2 / 10**magnitude,
                        decimal=decimal,
                    )
                except AssertionError:
                    text = f"Band {i + 1}{desc} failed"
                    errors.append(text)
                    LOGGER.error(text, exc_info=True)

            if errors:
                raise AssertionError(errors)


def assert_raster_max_mismatch(
    path_1: Union[str, CloudPath, Path],
    path_2: Union[str, CloudPath, Path],
    max_mismatch_pct=0.5,
) -> None:
    """
    Assert that two rasters are almost equal.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array and a precision of 10^-9 on the transform

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_max_mismatch(path, path2)
        >>> # Raises AssertionError if sth goes wrong

    Args:
        path_1 (Union[str, CloudPath, Path]): Raster 1
        path_2 (Union[str, CloudPath, Path]): Raster 2
        max_mismatch_pct (float): Maximum of element mismatch in %
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_max_mismatch."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1:
        with rasterio.open(str(path_2)) as ds_2:
            # Metadata
            assert_meta(ds_1.meta, ds_2.meta)

            # Compute the number of mismatch
            nof_mismatch = np.count_nonzero(ds_1.read() != ds_2.read())
            nof_elements = ds_1.count * ds_1.width * ds_1.height
            pct_mismatch = nof_mismatch / nof_elements * 100.0
            assert pct_mismatch < max_mismatch_pct, (
                f"Too many mismatches !\n"
                f"Number of mismatches: {nof_mismatch} / {nof_elements},\n"
                f"Percentage of mismatches: {pct_mismatch}% > {max_mismatch_pct}%,"
            )


def assert_dir_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path]
) -> None:
    """
    Assert that two directories are equal.

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/rasters"
        >>> assert_dir_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong

    Args:
        path_1 (str): Directory 1
        path_2 (str): Directory 2
    """
    path_1 = AnyPath(path_1)
    path_2 = AnyPath(path_2)
    assert path_1.is_dir(), f"{path_1} is not a directory!"
    assert path_2.is_dir(), f"{path_2} is not a directory!"

    dcmp = filecmp.dircmp(path_1, path_2)
    try:
        assert (
            dcmp.left_only == []
        ), f"More files in {path_1}!\n{pprint.pformat(list(dcmp.left_only))}"
        assert (
            dcmp.right_only == []
        ), f"More files in {path_2}!\n{pprint.pformat(list(dcmp.right_only))}"
    except FileNotFoundError:
        files_1 = [AnyPath(path).name for path in AnyPath(path_1).iterdir()]
        files_2 = [AnyPath(path).name for path in AnyPath(path_2).iterdir()]

        for f1 in files_1:
            assert (
                f1 in files_2
            ), f"File missing!\n{f1} not in {pprint.pformat(files_2)}"

        for f2 in files_2:
            assert (
                f2 in files_1
            ), f"File missing!\n{f2} not in {pprint.pformat(files_1)}"


def assert_geom_equal(
    geom_1: Union[str, CloudPath, Path, "gpd.GeoDataFrame"],
    geom_2: Union[str, CloudPath, Path, "gpd.GeoDataFrame"],
) -> None:
    """
    Assert that two geometries are equal
    (do not check equality between geodataframe as they may differ on other fields).

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/vectors/aoi.geojson"
        >>> assert_geom_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong

    .. WARNING::
        Only checks:
         - valid geometries
         - length of GeoDataFrame
         - CRS

    Args:
        geom_1 (Union[str, CloudPath, Path, "gpd.GeoDataFrame"]): Geometry 1
        geom_2 (Union[str, CloudPath, Path, "gpd.GeoDataFrame"]): Geometry 2
    """

    if not isinstance(geom_1, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geom_1 = vectors.read(geom_1)
    if not isinstance(geom_2, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geom_2 = vectors.read(geom_2)

    assert len(geom_1) == len(
        geom_2
    ), f"Non equal geometry lengths!\n{len(geom_1)} != {len(geom_2)}"
    assert (
        geom_1.crs == geom_2.crs
    ), f"Non equal geometry CRS!\n{geom_1.crs} != {geom_2.crs}"

    for idx in range(len(geom_1)):
        curr_geom_1 = geom_1.geometry.iat[idx]
        curr_geom_2 = geom_2.geometry.iat[idx]

        # If valid geometries, assert that the both are equal
        if curr_geom_1.is_valid and curr_geom_2.is_valid:
            assert curr_geom_1.equals(
                curr_geom_2
            ), f"Non equal geometries!\n{curr_geom_1} != {curr_geom_2}"


def assert_geom_almost_equal(
    geom_1: Union[str, CloudPath, Path, "gpd.GeoDataFrame"],
    geom_2: Union[str, CloudPath, Path, "gpd.GeoDataFrame"],
    decimal=9,
) -> None:
    """
    Assert that two geometries are equal
    (do not check equality between geodataframe as they may differ on other fields).

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/vectors/aoi.geojson"
        >>> assert_geom_almost_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong

    .. WARNING::
        Only checks:
         - valid geometries
         - length of GeoDataFrame
         - CRS

    Args:
        geom_1 (Union[str, CloudPath, Path, "gpd.GeoDataFrame"]): Geometry 1
        geom_2 (Union[str, CloudPath, Path, "gpd.GeoDataFrame"]): Geometry 2
        decimal (int): Number of decimal
    """

    if not isinstance(geom_1, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geom_1 = vectors.read(geom_1)
    if not isinstance(geom_2, (gpd.GeoDataFrame, gpd.GeoSeries)):
        geom_2 = vectors.read(geom_2)

    assert len(geom_1) == len(
        geom_2
    ), f"Non equal geometry lengths!\n{len(geom_1)} != {len(geom_2)}"
    assert (
        geom_1.crs == geom_2.crs
    ), f"Non equal geometry CRS!\n{geom_1.crs} != {geom_2.crs}"

    for idx in range(len(geom_1)):
        curr_geom_1 = geom_1.geometry.iat[idx]
        curr_geom_2 = geom_2.geometry.iat[idx]

        # If valid geometries, assert that the both are equal
        if curr_geom_1.is_valid and curr_geom_2.is_valid:
            assert curr_geom_1.equals_exact(
                curr_geom_2, tolerance=10**-decimal
            ), f"Non equal geometries!\n{curr_geom_1} != {curr_geom_2}"


def assert_xml_equal(xml_elem_1: etree._Element, xml_elem_2: etree._Element) -> None:
    """
    Assert that 2 XML (as etree Elements) are equal.

    -> Useful for pytests.

    Args:
        xml_elem_1 (etree._Element): 1st Element
        xml_elem_2 (etree._Element): 2nd Element
    """
    str_1 = etree.tostring(xml_elem_1, encoding="unicode")
    str_2 = etree.tostring(xml_elem_2, encoding="unicode")
    checker = LXMLOutputChecker()
    if not checker.check_output(str_1, str_2, 0):
        message = checker.output_difference(Example("", str_1), str_2, 0)
        raise AssertionError(message)


def assert_html_equal(xml_elem_1: etree._Element, xml_elem_2: etree._Element) -> None:
    """
    Assert that 2 XML (as etree Elements) are equal.

    -> Useful for pytests.

    Args:
        xml_elem_1 (etree._Element): 1st Element
        xml_elem_2 (etree._Element): 2nd Element
    """
    str_1 = html.tostring(xml_elem_1, encoding="unicode")
    str_2 = html.tostring(xml_elem_2, encoding="unicode")
    checker = LHTMLOutputChecker()
    if not checker.check_output(str_1, str_2, 0):
        message = checker.output_difference(Example("", str_1), str_2, 0)
        raise AssertionError(message)


def assert_xr_encoding_attrs(
    xda_1: Union[xr.DataArray, xr.Dataset], xda_2: Union[xr.DataArray, xr.Dataset]
):
    """
    Assert that the attributes and the encoding of xarray.DataArray/set are the same

    Args:
        xda_1 (Union[xr.DataArray, xr.Dataset]): First xarray
        xda_2 (Union[xr.DataArray, xr.Dataset]): First xarray
    """
    # Attributes
    try:
        assert xda_1.attrs == xda_2.attrs
    except AssertionError:
        try:
            for key, val in xda_1.attrs.items():
                if not key.startswith("_"):
                    assert (
                        xda_1.attrs[key] == xda_2.attrs[key]
                    ), f"{xda_1.attrs[key]=} != {xda_2.attrs[key]=}"

            for key, val in xda_2.attrs.items():
                if not key.startswith("_"):
                    assert (
                        xda_1.attrs[key] == xda_2.attrs[key]
                    ), f"{xda_1.attrs[key]=} != {xda_2.attrs[key]=}"
        except KeyError:
            raise AssertionError

    # Encoding
    try:
        assert xda_1.encoding == xda_2.encoding
    except AssertionError:
        try:
            for key, val in xda_1.encoding.items():
                if not key.startswith("_"):
                    assert (
                        xda_1.encoding[key] == xda_2.encoding[key]
                    ), f"{xda_1.encoding[key]=} != {xda_2.encoding[key]=}"

            for key, val in xda_2.encoding.items():
                if not key.startswith("_"):
                    assert (
                        xda_1.encoding[key] == xda_2.encoding[key]
                    ), f"{xda_1.encoding[key]=} != {xda_2.encoding[key]=}"
        except KeyError:
            raise AssertionError


def reduce_verbosity(other_loggers: list = None) -> None:
    """
    Reduce verbosity for other loggers (setting them to WARNING)

        Args:
            other_loggers (list): Other loggers to reduce verosity

    """
    loggers = [
        "boto3",
        "botocore",
        "botocore.hooks",
        "botocore.auth",
        "shapely",
        "shapely.geos",
        "fiona",
        "rasterio",
        "urllib3",
        "s3transfer",
        "pyproj",
        "matplotlib",
        "distributed",
        "asyncio",
        "bokeh",
    ]
    if other_loggers:
        loggers += other_loggers

    # Unique logger names
    for logger in list(set(loggers)):
        logging.getLogger(logger).setLevel(logging.WARNING)


def s3_env(*args, **kwargs):
    """
    Create S3 compatible storage environment
    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
    """
    import rasterio

    use_s3 = kwargs["use_s3_env_var"]
    function = args[0]

    @wraps(function)
    def s3_env_wrapper():
        """S3 environment wrapper"""
        if int(os.getenv(use_s3, 1)) and os.getenv(AWS_SECRET_ACCESS_KEY):
            # Define S3 client for S3 paths
            define_s3_client()
            os.environ[use_s3] = "1"
            LOGGER.info("Using S3 files")
            with rasterio.Env(
                CPL_CURL_VERBOSE=False,
                AWS_VIRTUAL_HOSTING=False,
                AWS_S3_ENDPOINT=AWS_S3_ENDPOINT,
                GDAL_DISABLE_READDIR_ON_OPEN=False,
            ):
                function()

        else:
            os.environ[use_s3] = "0"
            LOGGER.info("Using on disk files")
            function()

    return s3_env_wrapper


def define_s3_client():
    """
    Define S3 client
    """
    # ON S3
    client = S3Client(
        endpoint_url=f"https://{AWS_S3_ENDPOINT}",
        aws_access_key_id=os.getenv(AWS_ACCESS_KEY_ID),
        aws_secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY),
    )
    client.set_as_default_client()
