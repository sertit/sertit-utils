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
"""
CI tools

You can use `assert_raster_equal` only if you have installed sertit[full] or sertit[rasters]
"""
import filecmp
import os
from doctest import Example
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import rasterio
from cloudpathlib import AnyPath, CloudPath
from lxml import etree
from lxml.doctestcompare import LXMLOutputChecker


def get_mnt_path() -> str:
    """
    Return mounting directory `/mnt`.

    .. WARNING::
        This won't work on Windows !

    ```python
    >>> get_mnt_path()
    '/mnt'
    ```

    Returns:
        str: Mounting directory
    """
    return r"/mnt"


def _get_db_path(db_nb=2) -> str:
    """
    Return mounted directory `/mnt/ds2_db2` which corresponds to `\\ds2\database02`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    ```python
    >>> get_db_path(db_nb=2)
    '/mnt/ds2_db2'
    ```
    """
    db_path = f"{get_mnt_path()}/ds2_db{db_nb}"

    if not os.path.isdir(db_path):
        raise NotADirectoryError(f"Directory not found: {db_path}")

    return db_path


def get_db2_path() -> str:
    """
    Return mounted directory `/mnt/ds2_db2` which corresponds to `\\ds2\database02`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    ```python
    >>> get_db2_path()
    '/mnt/ds2_db2'
    ```

    Returns:
        str: Mounted directory
    """
    return _get_db_path(2)


def get_db3_path() -> str:
    """
    Return mounted directory `/mnt/ds2_db3` which corresponds to `\\ds2\database03`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    ```python
    >>> get_db3_path()
    '/mnt/ds2_db3'
    ```

    Returns:
        str: Mounted directory
    """
    return _get_db_path(3)


def get_db4_path() -> str:
    """
    Return mounted directory `/mnt/ds2_db4` which corresponds to `\\ds2\database04`.

    .. WARNING::
        Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    ```python
    >>> get_db4_path()
    '/mnt/ds2_db4'
    ```

    Returns:
        str: Mounted directory
    """
    return _get_db_path(4)


def assert_raster_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path]
) -> None:
    """
    Assert that two rasters are equal.

    -> Useful for pytests.

    ```python
    >>> path = r"CI\DATA\rasters\raster.tif"
    >>> assert_raster_equal(path, path)
    >>> # Raises AssertionError if sth goes wrong
    ```

    Args:
        path_1 (Union[str, CloudPath, Path]): Raster 1
        path_2 (Union[str, CloudPath, Path]): Raster 2
    """
    with rasterio.open(str(path_1)) as dst_1:
        with rasterio.open(str(path_2)) as dst_2:
            assert dst_1.meta == dst_2.meta
            np.testing.assert_array_equal(dst_1.read(), dst_2.read())


def assert_raster_almost_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path], decimal=7
) -> None:
    """
    Assert that two rasters are almost equal.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of `1E{decimal}` on the array and a precision of 10^-9 on the transform

    -> Useful for pytests.

    ```python
    >>> path = r"CI\DATA\rasters\raster.tif"
    >>> path2 = r"CI\DATA\rasters\raster_almost.tif"
    >>> assert_raster_equal(path, path2)
    >>> # Raises AssertionError if sth goes wrong
    ```

    Args:
        path_1 (Union[str, CloudPath, Path]): Raster 1
        path_2 (Union[str, CloudPath, Path]): Raster 2
    """

    with rasterio.open(str(path_1)) as dst_1:
        with rasterio.open(str(path_2)) as dst_2:
            assert dst_1.meta["driver"] == dst_2.meta["driver"]
            assert dst_1.meta["dtype"] == dst_2.meta["dtype"]
            assert dst_1.meta["nodata"] == dst_2.meta["nodata"]
            assert dst_1.meta["width"] == dst_2.meta["width"]
            assert dst_1.meta["height"] == dst_2.meta["height"]
            assert dst_1.meta["count"] == dst_2.meta["count"]
            assert dst_1.meta["crs"] == dst_2.meta["crs"]
            assert dst_1.meta["transform"].almost_equals(
                dst_2.meta["transform"], precision=1e-9
            )
            np.testing.assert_almost_equal(dst_1.read(), dst_2.read(), decimal=decimal)


def assert_dir_equal(
    path_1: Union[str, CloudPath, Path], path_2: Union[str, CloudPath, Path]
) -> None:
    """
    Assert that two directories are equal.

    # Useful for pytests.

    ```python
    >>> path = r"CI\DATA\rasters"
    >>> assert_dir_equal(path, path)
    >>> # Raises AssertionError if sth goes wrong
    ```

    Args:
        path_1 (str): Directory 1
        path_2 (str): Directory 2
    """
    path_1 = AnyPath(path_1)
    path_2 = AnyPath(path_2)
    assert path_1.is_dir()
    assert path_2.is_dir()

    dcmp = filecmp.dircmp(path_1, path_2)
    try:
        assert dcmp.left_only == []
        assert dcmp.right_only == []
    except FileNotFoundError:
        assert [AnyPath(path).name for path in AnyPath(path_1).iterdir()] == [
            AnyPath(path).name for path in AnyPath(path_2).iterdir()
        ]

    # def assert_archive_equal(path_1: str, path_2: str) -> None:


#     """
#     Assert that two archives are equal, by creating hashes that should be equal
#
#     Args:
#         path_1 (str): Archive 1
#         path_2 (str): Archive 2
#     """
#     filecmp.cmp(path_1, path_2)
#
#     file_1 = hashlib.sha256(open(path_1, 'rb').read()).digest()
#     file_2 = hashlib.sha256(open(path_2, 'rb').read()).digest()
#     assert file_1 == file_2


def assert_geom_equal(geom_1: gpd.GeoDataFrame, geom_2: gpd.GeoDataFrame) -> None:
    """
    Assert that two geometries are equal
    (do not check equality between geodataframe as they may differ on other fields).

    -> Useful for pytests.

    ```python
    >>> path = r"CI\DATA\vectors\aoi.geojson"
    >>> assert_geom_equal(path, path)
    >>> # Raises AssertionError if sth goes wrong
    ```

    .. WARNING::
        Only checks:
         - valid geometries
         - length of GeoDataFrame
         - CRS

    Args:
        geom_1 (gpd.GeoDataFrame): Geometry 1
        geom_2 (gpd.GeoDataFrame): Geometry 2
    """
    assert len(geom_1) == len(geom_2)
    assert geom_1.crs == geom_2.crs
    for idx in range(len(geom_1)):
        if geom_1.geometry.iat[idx].is_valid and geom_2.geometry.iat[idx].is_valid:
            # If valid geometries, assert that the both are equal
            assert geom_1.geometry.iat[idx].equals(geom_2.geometry.iat[idx])


def assert_xml_equal(xml_elem_1: etree._Element, xml_elem_2: etree._Element) -> None:
    """
    Assert that 2 XML (as etree Elements) are equal.

    -> Useful for pytests.

    Args:
        xml_elem_1 (etree._Element): 1st Element
        xml_elem_2 (etree._Element): 2nd Element
    """
    str_1 = etree.tounicode(xml_elem_1)
    str_2 = etree.tounicode(xml_elem_2)
    checker = LXMLOutputChecker()
    if not checker.check_output(str_1, str_2, 0):
        message = checker.output_difference(Example("", str_1), str_2, 0)
        raise AssertionError(message)
