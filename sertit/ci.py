"""
\\ds2\database02 → /mnt/ds2_db2
\\ds2\database03 → /mnt/ds2_db3
\\ds2\database04 → /mnt/ds2_db4
"""
import filecmp
import os

import numpy as np
import geopandas as gpd
import rasterio


def get_mnt_path() -> str:
    """
    Return mounting directory `/mnt`.

    **WARNING**:  This won't work on Windows !

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

    **WARNING**: Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

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

    **WARNING**: Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

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

    **WARNING**: Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

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

    **WARNING**: Use it carefully (OK in CI) as this directory may not exist ! This won't work on Windows !

    ```python
    >>> get_db4_path()
    '/mnt/ds2_db4'
    ```

    Returns:
        str: Mounted directory
    """
    return _get_db_path(4)


def assert_raster_equal(path_1: str, path_2: str) -> None:
    """
    Assert that two rasters are equal.

    Useful in pytests.

    ```python
    >>> path = r"CI\DATA\rasters\raster.tif"
    >>> assert_raster_equal(path, path)
    >>> # Raises AssertError if sth goes wrong
    ```

    Args:
        path_1 (str): Raster 1
        path_2 (str): Raster 2
    """
    with rasterio.open(path_1) as dst_1:
        with rasterio.open(path_2) as dst_2:
            assert dst_1.meta == dst_2.meta
            np.testing.assert_array_equal(dst_1.read(), dst_2.read())


def assert_dir_equal(path_1: str, path_2: str) -> None:
    """
    Assert that two directories are equal.

    Useful in pytests.

    ```python
    >>> path = r"CI\DATA\rasters"
    >>> assert_dir_equal(path, path)
    >>> # Raises AssertError if sth goes wrong
    ```

    Args:
        path_1 (str): Directory 1
        path_2 (str): Directory 2
    """
    dcmp = filecmp.dircmp(path_1, path_2)
    assert os.path.isdir(path_1)
    assert os.path.isdir(path_2)
    assert dcmp.left_only == []
    assert dcmp.right_only == []


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

    Useful in pytests.

    ```python
    >>> path = r"CI\DATA\vectors\aoi.geojson"
    >>> assert_geom_equal(path, path)
    >>> # Raises AssertError if sth goes wrong
    ```

    **WARNING**: only checks:
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
