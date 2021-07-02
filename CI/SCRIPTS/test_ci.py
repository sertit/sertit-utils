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
""" Script testing the CI """
import os
import tempfile

import pytest
from cloudpathlib import CloudPath
from lxml import etree

from CI.SCRIPTS.script_utils import files_path, rasters_path, s3_env, vectors_path
from sertit import ci, misc, rasters_rio, vectors


@s3_env
def test_assert_dir():
    """Test CI functions"""
    # Dirs
    dir2 = files_path().joinpath("core")

    ci.assert_dir_equal(files_path(), files_path())
    with pytest.raises(AssertionError):
        ci.assert_dir_equal(files_path(), dir2)


@s3_env
def test_assert_vect():
    """Test CI functions"""
    # Vector
    vector_path = vectors_path().joinpath("aoi.geojson")
    vector2_path = vectors_path().joinpath("aoi2.geojson")

    vec_df = vectors.read(vector_path)
    vec2_df = vectors.read(vector2_path)

    assert not vec_df.empty
    ci.assert_geom_equal(vec_df, vec_df)
    with pytest.raises(AssertionError):
        ci.assert_geom_equal(vec_df, vec2_df)


@s3_env
def test_assert_raster():
    # Rasters
    raster_path = rasters_path().joinpath("raster.tif")
    raster2_path = rasters_path().joinpath("raster_masked.tif")

    ci.assert_raster_equal(raster_path, raster_path)
    with pytest.raises(AssertionError):
        ci.assert_raster_equal(raster_path, raster2_path)

    # Rasters almost equal
    with tempfile.TemporaryDirectory() as tmp:
        # Read and convert raster to float
        arr, meta = rasters_rio.read(raster_path)
        arr = arr.astype(float)
        meta["dtype"] = float
        raster_float_path = os.path.join(tmp, "raster_float.tif")
        rasters_rio.write(arr, meta, raster_float_path)

        # Slightly change it
        offset = 1e-07
        arr += offset
        meta["transform"].shear(offset, offset)
        raster_almost_path = os.path.join(tmp, "raster_almost.tif")
        rasters_rio.write(arr, meta, raster_almost_path)

        ci.assert_raster_almost_equal(raster_float_path, raster_almost_path)
        with pytest.raises(AssertionError):
            raster2_path = rasters_path().joinpath("raster_masked.tif")
            ci.assert_raster_almost_equal(raster_path, raster2_path)

        with pytest.raises(AssertionError):
            offset = 1e-05
            arr += offset
            meta["transform"].shear(offset, offset)
            raster_too_much_path = os.path.join(tmp, "raster_too_much.tif")
            rasters_rio.write(arr, meta, raster_too_much_path)
            ci.assert_raster_almost_equal(raster_float_path, raster_too_much_path)


@s3_env
def test_assert_xml():
    # XML
    xml_folder = files_path().joinpath("LM05_L1TP_200030_20121230_20200820_02_T2_CI")
    xml_path = xml_folder.joinpath("LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml")
    xml_bad_path = xml_folder.joinpath("false_xml.xml")

    if isinstance(files_path(), CloudPath):
        xml_path = xml_path.fspath
        xml_bad_path = xml_bad_path.fspath

    xml_ok = etree.parse(str(xml_path)).getroot()
    xml_nok = etree.parse(str(xml_bad_path)).getroot()

    ci.assert_xml_equal(xml_ok, xml_ok)
    with pytest.raises(AssertionError):
        ci.assert_xml_equal(xml_ok, xml_nok)


@pytest.mark.skipif(not misc.in_docker(), reason="Only works in docker")
def test_mnt():
    """Test mounted directories"""
    assert ci.get_db2_path() == "/mnt/ds2_db2"
    assert ci.get_db3_path() == "/mnt/ds2_db3"
    assert ci.get_db4_path() == "/mnt/ds2_db4"

    with pytest.raises(NotADirectoryError):
        ci._get_db_path(5)
