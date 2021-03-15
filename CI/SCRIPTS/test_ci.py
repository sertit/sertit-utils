""" Script testing the files """
import os
import pytest

import geopandas as gpd

from CI.SCRIPTS.script_utils import RASTER_DATA, FILE_DATA, GEO_DATA
from sertit import ci
from sertit.misc import in_docker


def test_assert():
    """ Test CI functions """
    # Dirs
    dir2 = os.path.join(FILE_DATA, "core")

    ci.assert_dir_equal(FILE_DATA, FILE_DATA)
    with pytest.raises(AssertionError):
        ci.assert_dir_equal(FILE_DATA, dir2)

    # Vector
    vector_path = os.path.join(GEO_DATA, "aoi.geojson")
    vector2_path = os.path.join(GEO_DATA, "aoi2.geojson")

    vec_df = gpd.read_file(vector_path)
    vec2_df = gpd.read_file(vector2_path)

    ci.assert_geom_equal(vec_df, vec_df)
    with pytest.raises(AssertionError):
        ci.assert_geom_equal(vec_df, vec2_df)

    # Rasters
    raster_path = os.path.join(RASTER_DATA, "raster.tif")
    raster2_path = os.path.join(RASTER_DATA, "raster_masked.tif")

    ci.assert_raster_equal(raster_path, raster_path)
    with pytest.raises(AssertionError):
        ci.assert_raster_equal(raster_path, raster2_path)


def test_mnt():
    """ Test mounted directories """
    assert ci.get_db2_path() == '/mnt/ds2_db2'
    assert ci.get_db3_path() == '/mnt/ds2_db3'
    assert ci.get_db4_path() == '/mnt/ds2_db4'

    with pytest.raises(NotADirectoryError):
        ci._get_db_path(5)
