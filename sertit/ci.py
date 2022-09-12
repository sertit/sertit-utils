# -*- coding: utf-8 -*-
# Copyright 2022, SERTIT-ICube - France, https://sertit.unistra.fr/
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
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
from cloudpathlib import AnyPath, CloudPath
from lxml import etree, html
from lxml.doctestcompare import LHTMLOutputChecker, LXMLOutputChecker

from sertit import vectors


def get_mnt_path() -> str:
    """
    Return mounting directory :code::code:`/mnt`.

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
    Return mounted directory :code::code:`/mnt/ds2_db2` which corresponds to :code::code:`/ds2/database02`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    .. code-block:: python

        >>> get_db_path(db_nb=2)
        '/mnt/ds2_db2'
    """
    db_path = f"{get_mnt_path()}/ds2_db{db_nb}"

    if not os.path.isdir(db_path):
        raise NotADirectoryError(f"Directory not found: {db_path}")

    return db_path


def get_db2_path() -> str:
    """
    Return mounted directory :code::code:`/mnt/ds2_db2` which corresponds to :code::code:`/ds2/database02`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    .. code-block:: python

        >>> get_db2_path()
        '/mnt/ds2_db2'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(2)


def get_db3_path() -> str:
    """
    Return mounted directory :code::code:`/mnt/ds2_db3` which corresponds to :code::code:`/ds2/database03`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    .. code-block:: python

        >>> get_db3_path()
        '/mnt/ds2_db3'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(3)


def get_db4_path() -> str:
    """
    Return mounted directory :code:`/mnt/ds2_db4` which corresponds to :code:`/ds2/database04`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    .. code-block:: python

        >>> get_db4_path()
        '/mnt/ds2_db4'

    Returns:
        str: Mounted directory
    """
    return _get_db_path(4)


def _assert_field(dict_1: dict, dict_2: dict, field: str) -> None:
    """
    Compare two fields of a dictionary

    Args:
        dict_1 (dict): Dict 1
        dict_2 (dict): Dict 2
        field (str): Field to compare
    """
    assert (
        dict_1[field] == dict_2[field]
    ), f"{field} incoherent:\n{dict_1[field]} != {dict_2[field]}"


def _assert_meta(meta_1, meta_2):
    """
    Compare rasterio metadata

    Args:
        meta_1 (dict): Metadata 1
        meta_2 (dict): Metadata 2
    """
    # Driver
    _assert_field(meta_1, meta_2, "driver")
    _assert_field(meta_1, meta_2, "dtype")
    _assert_field(meta_1, meta_2, "nodata")
    _assert_field(meta_1, meta_2, "width")
    _assert_field(meta_1, meta_2, "height")
    _assert_field(meta_1, meta_2, "count")
    _assert_field(meta_1, meta_2, "crs")

    assert meta_1["transform"].almost_equals(
        meta_2["transform"], precision=1e-9
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
            _assert_meta(ds_1.meta, ds_2.meta)

            # Assert equal
            np.testing.assert_array_equal(ds_1.read(), ds_2.read())


def assert_raster_almost_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path], decimal=7
) -> None:
    """
    Assert that two rasters are almost equal.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array and a precision of 10^-9 on the transform

    Useful for pytests.

    .. code-block:: python

        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_equal(path, path2)
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
            _assert_meta(ds_1.meta, ds_2.meta)

            # Assert almost equal
            np.testing.assert_almost_equal(ds_1.read(), ds_2.read(), decimal=decimal)


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
        >>> assert_raster_equal(path, path2)
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
            _assert_meta(ds_1.meta, ds_2.meta)

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
        assert (
            files_1 == files_2
        ), f"Files non equal!\n{pprint.pformat(files_1)} != {pprint.pformat(files_2)}"


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
                curr_geom_2, tolerance=0.5 * 10 ** decimal
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


def reduce_verbosity(other_loggers: list = None) -> None:
    """ Reduce verbosity for other loggers """
    loggers = [
        "boto3",
        "botocore",
        "shapely",
        "fiona",
        "rasterio",
        "urllib3",
        "s3transfer",
        "pyproj",
        "matplotlib",
    ]
    if other_loggers:
        loggers += other_loggers

    # Unique logger names
    for logger in list(set(loggers)):
        logging.getLogger(logger).setLevel(logging.WARNING)
