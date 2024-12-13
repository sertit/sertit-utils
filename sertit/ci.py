# Copyright 2024, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""

import filecmp
import logging
import pprint
import tempfile
from doctest import Example
from typing import Any, Union

import geopandas as gpd
import numpy as np
from lxml import etree, html
from lxml.doctestcompare import LHTMLOutputChecker, LXMLOutputChecker
from shapely import force_2d, normalize
from shapely.testing import assert_geometries_equal

from sertit import AnyPath, files, path, s3, unistra
from sertit.logs import SU_NAME, deprecation_warning
from sertit.types import AnyPathStrType, AnyXrDataStructure

LOGGER = logging.getLogger(SU_NAME)

# Alias for compatibility (don't deprecate them)
AWS_ACCESS_KEY_ID = s3.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = s3.AWS_SECRET_ACCESS_KEY
AWS_S3_ENDPOINT = s3.AWS_S3_ENDPOINT


def s3_env(*args, **kwargs):
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.unistra` instead of :py:mod:`sertit.ci`
    """
    deprecation_warning(
        "This function is deprecated. Import it from 'sertit.unistra' instead of 'sertit.ci'"
    )
    return unistra.s3_env(*args, **kwargs)


def define_s3_client():
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.unistra` instead of :py:mod:`sertit.ci`
    """
    deprecation_warning(
        "This function is deprecated. Import it from 'sertit.unistra' instead of 'sertit.ci'"
    )
    return unistra.define_s3_client()


def get_db2_path():
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.unistra` instead of :py:mod:`sertit.ci`
    """
    deprecation_warning(
        "This function is deprecated. Import it from 'sertit.unistra' instead of 'sertit.ci'"
    )
    return unistra.get_db2_path()


def get_db3_path():
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.unistra` instead of :py:mod:`sertit.ci`
    """
    deprecation_warning(
        "This function is deprecated. Import it from 'sertit.unistra' instead of 'sertit.ci'"
    )
    return unistra.get_db3_path()


def get_db4_path():
    """
    .. deprecated:: 1.30.0
       Import it from :py:mod:`sertit.unistra` instead of :py:mod:`sertit.ci`
    """
    deprecation_warning(
        "This function is deprecated. Import it from 'sertit.unistra' instead of 'sertit.ci'"
    )
    return unistra.get_db4_path()


def assert_val(val_1: Any, val_2: Any, field: str) -> None:
    """
    Compare two values corresponding to a field

    Args:
        val_1 (Any): Value 1
        val_2 (Any): Value 2
        field (str): Field to compare
    """
    desc = f"{field} incoherent:\n{val_1} != {val_2}"

    # Manage None as value
    if val_2 is None or val_1 is None:
        assert val_1 is val_2, desc
    elif val_2 is np.nan or val_1 is np.nan:
        assert np.isnan(val_1) and np.isnan(val_2), desc
    else:
        try:
            assert val_1 == val_2, desc
        except ValueError:
            assert all(val_1 == val_2), desc


def assert_field(dict_1: dict, dict_2: dict, field: str) -> None:
    """
    Compare two fields of a dictionary

    Args:
        dict_1 (dict): Dict 1
        dict_2 (dict): Dict 2
        field (str): Field to compare
    """
    assert_val(dict_1[field], dict_2[field], field)


def assert_files_equal(file_1: AnyPathStrType, file_2: AnyPathStrType):
    """
    Assert to files are equal by hashing its content

    Args:
        file_1 (str): Path to file 1
        file_2 (str): Path to file 2
    """
    with file_1.open("r") as f1, file_2.open("r") as f2:
        assert files.hash_file_content(f1.read()) == files.hash_file_content(f2.read())


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


def assert_raster_equal(path_1: AnyPathStrType, path_2: AnyPathStrType) -> None:
    """
    Assert that two rasters are equal.

    Useful for pytests.

    Args:
        path_1 (AnyPathStrType): Raster 1
        path_2 (AnyPathStrType): Raster 2

    Example:
        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> assert_raster_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_equal."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1, rasterio.open(str(path_2)) as ds_2:
        # Metadata
        assert_meta(ds_1.meta, ds_2.meta)

        # Assert equal
        np.testing.assert_array_equal(ds_1.read(), ds_2.read())


def assert_raster_almost_equal(
    path_1: AnyPathStrType, path_2: AnyPathStrType, decimal=7
) -> None:
    """
    Assert that two rasters are almost equal.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array and a precision of 10^-{decimal} on the transform

    Useful for pytests.

    Args:
        path_1 (AnyPathStrType): Raster 1
        path_2 (AnyPathStrType): Raster 2
        decimal (int): Number of decimal

    Example:
        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_almost_equal(path, path2)
        >>> # Raises AssertionError if sth goes wrong
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_almost_equal."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1, rasterio.open(str(path_2)) as ds_2:
        # Metadata
        assert_meta(ds_1.meta, ds_2.meta, tf_precision=10**-decimal)

        # Assert almost equal
        errors = []
        for i in range(ds_1.count):
            desc = (
                f": {ds_1.descriptions[i]}" if ds_1.descriptions[i] is not None else ""
            )
            LOGGER.info(f"Checking Band {i + 1}{desc}")
            try:
                marr_1 = ds_1.read(i + 1)
                marr_2 = ds_2.read(i + 1)
                np.testing.assert_array_almost_equal(marr_1, marr_2, decimal=decimal)
            except AssertionError:
                text = f"Band {i + 1}{desc} failed"
                errors.append(text)
                LOGGER.error(text, exc_info=True)

        if errors:
            raise AssertionError(errors)


def assert_raster_almost_equal_magnitude(
    path_1: AnyPathStrType, path_2: AnyPathStrType, decimal=2
) -> None:
    """
    Assert that two rasters are almost equal, with the decimal taken on the scientif representation of the array.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array divided by the order of magnitude of the array and a precision of 10^-{decimal} on the transform

    i.e. `decimal=2, mean(array) = 15.2, true = 13.2687, false = 13.977, comparison: 1.32687, false <-> 1.3977 => false (2 != 9)`

    Useful for pytests.

    Args:
        path_1 (AnyPathStrType): Raster 1
        path_2 (AnyPathStrType): Raster 2
        decimal (int): Number of decimal

    Example:
        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_almost_equal_magnitude(path, path2)
        >>> # Raises AssertionError if sth goes wrong
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_almost_equal."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1, rasterio.open(str(path_2)) as ds_2:
        # Metadata
        assert_meta(ds_1.meta, ds_2.meta, tf_precision=10**-decimal)

        # Assert almost equal
        errors = []
        for i in range(ds_1.count):
            desc = (
                f": {ds_1.descriptions[i]}" if ds_1.descriptions[i] is not None else ""
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
    path_1: AnyPathStrType,
    path_2: AnyPathStrType,
    max_mismatch_pct=0.5,
) -> None:
    """
    Assert that two rasters are almost equal.
    (everything is equal except the transform and the arrays that are almost equal)

    Accepts an offset of :code:`1E{decimal}` on the array and a precision of 10^-9 on the transform

    Useful for pytests.

    Args:
        path_1 (AnyPathStrType): Raster 1
        path_2 (AnyPathStrType): Raster 2
        max_mismatch_pct (float): Maximum of element mismatch in %

    Example:
        >>> path = r"CI/DATA/rasters/raster.tif"
        >>> path2 = r"CI/DATA/rasters/raster_almost.tif"
        >>> assert_raster_max_mismatch(path, path2)
        >>> # Raises AssertionError if sth goes wrong
    """
    try:
        import rasterio
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' to use assert_raster_max_mismatch."
        ) from ex

    with rasterio.open(str(path_1)) as ds_1, rasterio.open(str(path_2)) as ds_2:
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


def assert_dir_equal(path_1: AnyPathStrType, path_2: AnyPathStrType) -> None:
    """
    Assert that two directories are equal.

    Useful for pytests.

    Args:
        path_1 (str): Directory 1
        path_2 (str): Directory 2

    Example:
        >>> path = r"CI/DATA/rasters"
        >>> assert_dir_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong
    """
    path_1 = AnyPath(path_1)
    path_2 = AnyPath(path_2)
    assert path_1.is_dir(), f"{path_1} is not a directory!"
    assert path_2.is_dir(), f"{path_2} is not a directory!"

    with tempfile.TemporaryDirectory() as tmpdir:
        if path.is_cloud_path(path_1):
            path_1 = s3.download(path_1, tmpdir)
        if path.is_cloud_path(path_2):
            path_2 = s3.download(path_2, tmpdir)

        dcmp = filecmp.dircmp(path_1, path_2)
        try:
            assert (
                dcmp.left_only == []
            ), f"More files in {path_1}!\n{pprint.pformat(list(dcmp.left_only))}"
            assert (
                dcmp.right_only == []
            ), f"More files in {path_2}!\n{pprint.pformat(list(dcmp.right_only))}"
        except FileNotFoundError:
            files_1 = [p.name for p in path_1.iterdir()]
            files_2 = [p.name for p in path_2.iterdir()]

            for f1 in files_1:
                assert (
                    f1 in files_2
                ), f"File missing!\n{f1} not in {pprint.pformat(files_2)}"

            for f2 in files_2:
                assert (
                    f2 in files_1
                ), f"File missing!\n{f2} not in {pprint.pformat(files_1)}"


def assert_geom_equal(
    geom_1: Union[AnyPathStrType, "gpd.GeoDataFrame"],
    geom_2: Union[AnyPathStrType, "gpd.GeoDataFrame"],
    ignore_z=True,
) -> None:
    """
    Assert that two geometries are equal
    (do not check equality between geodataframe as they may differ on other fields).

    Useful for pytests.

    Args:
        geom_1 (Union[AnyPathStrType, "gpd.GeoDataFrame"]): Geometry 1
        geom_2 (Union[AnyPathStrType, "gpd.GeoDataFrame"]): Geometry 2
        ignore_z (bool): Ignore Z coordinate

    Warning:
        Only checks:
         - valid geometries
         - length of GeoDataFrame
         - CRS

    Example:
        >>> path = r"CI/DATA/vectors/aoi.geojson"
        >>> assert_geom_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong
    """
    try:
        from sertit import vectors
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' and 'geopandas' to use the 'vectors' package."
        ) from ex

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
        curr_geom_1 = normalize(geom_1.geometry.iat[idx])
        curr_geom_2 = normalize(geom_2.geometry.iat[idx])

        if ignore_z:
            curr_geom_1 = force_2d(curr_geom_1)
            curr_geom_2 = force_2d(curr_geom_2)

        # If valid geometries, assert that the both are equal
        if curr_geom_1.is_valid and curr_geom_2.is_valid:
            try:
                assert curr_geom_1.equals(
                    curr_geom_2
                ), f"Non equal geometries!\n{curr_geom_1} != {curr_geom_2}"
            except AssertionError:
                # Get tolerance
                tol = 1e-7 if geom_1.crs.is_geographic else 1e-3

                # This functions tests differently than equals
                assert_geometries_equal(
                    curr_geom_1,
                    curr_geom_2,
                    tolerance=tol,
                    equal_none=False,
                    equal_nan=False,
                    normalize=True,
                )


def assert_geom_almost_equal(
    geom_1: Union[AnyPathStrType, "gpd.GeoDataFrame"],
    geom_2: Union[AnyPathStrType, "gpd.GeoDataFrame"],
    decimal=9,
    ignore_z=True,
) -> None:
    """
    Assert that two geometries are equal
    (do not check equality between geodataframe as they may differ on other fields).

    Useful for pytests.

    Args:
        geom_1 (Union[AnyPathStrType, "gpd.GeoDataFrame"]): Geometry 1
        geom_2 (Union[AnyPathStrType, "gpd.GeoDataFrame"]): Geometry 2
        decimal (int): Number of decimal
        ignore_z (bool): Ignore Z coordinate

    Warning:
        Only checks:
         - valid geometries
         - length of GeoDataFrame
         - CRS

    Example:
        >>> path = r"CI/DATA/vectors/aoi.geojson"
        >>> assert_geom_almost_equal(path, path)
        >>> # Raises AssertionError if sth goes wrong
    """
    try:
        from sertit import vectors
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'rasterio' and 'geopandas' to use the 'vectors' package."
        ) from ex

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
        curr_geom_1 = normalize(geom_1.geometry.iat[idx])
        curr_geom_2 = normalize(geom_2.geometry.iat[idx])

        if ignore_z:
            curr_geom_1 = force_2d(curr_geom_1)
            curr_geom_2 = force_2d(curr_geom_2)

        # If valid geometries, assert that the both are equal
        if curr_geom_1.is_valid and curr_geom_2.is_valid:
            try:
                assert curr_geom_1.equals_exact(
                    curr_geom_2, tolerance=10**-decimal
                ), f"Non equal geometries!\n{curr_geom_1} != {curr_geom_2}"
            except AssertionError:
                # This functions tests differently than equals
                assert_geometries_equal(
                    curr_geom_1,
                    curr_geom_2,
                    tolerance=10**-decimal,
                    equal_none=False,
                    equal_nan=False,
                    normalize=True,
                )


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
    xda_1: AnyXrDataStructure,
    xda_2: AnyXrDataStructure,
    unchecked_attr: Union[list, str] = None,
):
    """
    Assert that the attributes and the encoding of xarray.DataArray/set are the same

    Args:
        xda_1 (AnyXrDataStructure): First xarray
        xda_2 (AnyXrDataStructure): Second xarray
        unchecked_attr (Union[list, str]): Don't check this list of attributes
    """
    if unchecked_attr is None:
        unchecked_attr = []
    elif not isinstance(unchecked_attr, list):
        unchecked_attr = [unchecked_attr]

    # Attributes
    try:
        assert xda_1.attrs == xda_2.attrs
    except AssertionError:
        try:
            for key, _ in xda_1.attrs.items():
                if not key.startswith("_") and key not in unchecked_attr:
                    assert (
                        xda_1.attrs[key] == xda_2.attrs[key]
                    ), f"{xda_1.attrs[key]=} != {xda_2.attrs[key]=}"

            for key, _ in xda_2.attrs.items():
                if not key.startswith("_") and key not in unchecked_attr:
                    assert (
                        xda_1.attrs[key] == xda_2.attrs[key]
                    ), f"{xda_1.attrs[key]=} != {xda_2.attrs[key]=}"
        except KeyError as exc:
            raise AssertionError(
                f"Missing key {exc} in attributes of one DataArray/Dataset"
            ) from exc

    # Encoding
    try:
        assert xda_1.encoding == xda_2.encoding
    except AssertionError:
        try:
            for key, _ in xda_1.encoding.items():
                if not key.startswith("_") and key not in unchecked_attr:
                    assert (
                        xda_1.encoding[key] == xda_2.encoding[key]
                    ), f"{xda_1.encoding[key]=} != {xda_2.encoding[key]=}"

            for key, _ in xda_2.encoding.items():
                if not key.startswith("_") and key not in unchecked_attr:
                    assert (
                        xda_1.encoding[key] == xda_2.encoding[key]
                    ), f"{xda_1.encoding[key]=} != {xda_2.encoding[key]=}"
        except KeyError as exc:
            raise AssertionError(
                f"Missing key {exc} in attributes of one DataArray/Dataset"
            ) from exc


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
        "asyncio",
        "bokeh",
        "numba",
        "distributed",
        "distributed.worker",
        "distributed.scheduler",
        "distributed.nanny",
        "distributed.core",
        "distributed.http.proxy",
        "distributed.batched",
    ]
    if other_loggers:
        loggers += other_loggers

    # Unique logger names
    for logger in list(set(loggers)):
        logging.getLogger(logger).setLevel(logging.WARNING)
