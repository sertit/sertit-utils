""" Script testing the files """
import os
import pytest

import geopandas as gpd
from lxml import etree

from CI.SCRIPTS.script_utils import RASTER_DATA, FILE_DATA, GEO_DATA
from sertit import ci


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

    # XML
    xml_folder = os.path.join(FILE_DATA, "LM05_L1TP_200030_20121230_20200820_02_T2_CI")
    xml_path = os.path.join(xml_folder, "LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml")
    xml_bad_path = os.path.join(xml_folder, "false_xml.xml")
    xml_ok = etree.parse(xml_path).getroot()
    xml_nok = etree.parse(xml_bad_path).getroot()

    ci.assert_xml_equal(xml_ok, xml_ok)
    with pytest.raises(AssertionError):
        ci.assert_xml_equal(xml_ok, xml_nok)


def test_mnt():
    """ Test mounted directories """
    assert ci.get_db2_path() == '/mnt/ds2_db2'
    assert ci.get_db3_path() == '/mnt/ds2_db3'
    assert ci.get_db4_path() == '/mnt/ds2_db4'

    with pytest.raises(NotADirectoryError):
        ci._get_db_path(5)
