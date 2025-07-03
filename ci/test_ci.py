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
"""Script testing the CI"""

import os
import tempfile

import numpy as np
import pytest
from lxml import etree

from ci.script_utils import files_path, rasters_path, s3_env, vectors_path
from sertit import ci, path, rasters, rasters_rio, vectors

ci.reduce_verbosity()


@s3_env
def test_assert_base():
    """Test CI functions"""
    # assert_val
    ci.assert_val("a", "a", "same string")
    ci.assert_val(None, None, "both None")
    ci.assert_val(np.nan, np.nan, "Nans")
    ci.assert_val([np.nan, np.nan], [np.nan, np.nan], "different vals (list of nans)")
    with pytest.raises(AssertionError):
        ci.assert_val("a", "b", "different string")

    with pytest.raises(AssertionError):
        ci.assert_val("a", None, "only one None")

    # assert_field
    dict_1 = {
        "a": 1,
    }
    dict_2 = {
        "a": 2,
    }
    ci.assert_field(dict_1, dict_1, "a")
    with pytest.raises(AssertionError):
        ci.assert_field(dict_1, dict_2, "a")

    ci.assert_val([1, 2], [1, 2], "same list")
    with pytest.raises(AssertionError):
        ci.assert_val([1, 2], [3, 2], "different list")

    with pytest.raises(AssertionError):
        ci.assert_val(None, 3, "different vals (None)")

    with pytest.raises(AssertionError):
        ci.assert_val(np.nan, 3, "different vals (nans)")

    with pytest.raises(AssertionError):
        ci.assert_val(np.nan, None, "different vals (nans + None)")


@s3_env
def test_assert_dir():
    """Test CI functions"""
    # Dirs
    dir2 = files_path().joinpath("core")

    ci.assert_dir_equal(files_path(), files_path())
    with pytest.raises(AssertionError):
        ci.assert_dir_equal(files_path(), dir2)


@s3_env
def test_assert_files():
    """Test CI functions"""
    ok_path = files_path().joinpath("productPreview.html")
    false_path = files_path().joinpath("false.html")

    ci.assert_files_equal(ok_path, ok_path)
    ci.assert_files_equal(str(ok_path), str(ok_path))
    with pytest.raises(AssertionError):
        ci.assert_files_equal(ok_path, false_path)


@s3_env
def test_assert_vect():
    """Test CI functions"""
    # Vector
    vector_path = vectors_path().joinpath("aoi.geojson")
    vector2_path = vectors_path().joinpath("aoi2.geojson")

    vec_df = vectors.read(vector_path)
    vec2_df = vectors.read(vector2_path)

    assert not vec_df.empty
    ci.assert_geom_equal(vec_df, vector_path)
    with pytest.raises(AssertionError):
        ci.assert_geom_equal(vector_path, vec2_df)

    assert not vec_df.empty
    ci.assert_geom_almost_equal(vec_df, vector_path)
    with pytest.raises(AssertionError):
        ci.assert_geom_almost_equal(vector_path, vec2_df)


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

        # Almost equal
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

        # Max mismatch
        ci.assert_raster_max_mismatch(raster_path, raster_path, max_mismatch_pct=0.001)
        with pytest.raises(AssertionError):
            ci.assert_raster_max_mismatch(raster_float_path, raster_almost_path)

        # Magnitude
        ci.assert_raster_almost_equal_magnitude(raster_float_path, raster_almost_path)
        with pytest.raises(AssertionError):
            raster2_path = rasters_path().joinpath("raster_masked.tif")
            ci.assert_raster_almost_equal_magnitude(raster_path, raster2_path)

        with pytest.raises(AssertionError):
            offset = 1e-02
            arr += offset
            meta["transform"].shear(offset, offset)
            raster_too_much_path = os.path.join(tmp, "raster_too_much.tif")
            rasters_rio.write(arr, meta, raster_too_much_path)
            ci.assert_raster_almost_equal_magnitude(
                raster_float_path, raster_too_much_path, decimal=3
            )

    # Test encoding
    rast_1_xr = rasters.read(raster_path)
    rast_2_xr = rasters.read(raster2_path)
    rast_2_xr.encoding["add_field"] = True
    ci.assert_xr_encoding_attrs(rast_1_xr, rast_1_xr)
    with pytest.raises(AssertionError):
        ci.assert_xr_encoding_attrs(rast_1_xr, rast_2_xr)


@s3_env
def test_dim():
    """Test CI functions"""
    # Dim file
    dim_path = rasters_path().joinpath("dim_file.dim")
    dim_path_only_coh = rasters_path().joinpath("dim_file_only_coh.dim")
    dim_path_spk_only_coh = rasters_path().joinpath("dim_file_Spk_only_coh.dim")
    dim_path_different_image = rasters_path().joinpath("dim_file_other.dim")

    ci.assert_dim_file_equal(dim_path, dim_path)

    # Test if error when different number of bands
    with pytest.raises(AssertionError):
        ci.assert_dim_file_equal(dim_path, dim_path_only_coh)

    # Test if error when completely different
    with pytest.raises(AssertionError):
        ci.assert_dim_file_equal(dim_path, dim_path_different_image)

    # Test if error with same name different bands
    with pytest.raises(AssertionError):
        ci.assert_dim_file_equal(dim_path_only_coh, dim_path_spk_only_coh)


@s3_env
def test_assert_xml():
    # XML
    xml_folder = files_path().joinpath("LM05_L1TP_200030_20121230_20200820_02_T2_CI")
    xml_path = xml_folder.joinpath("LM05_L1TP_200030_20121230_20200820_02_T2_MTL.xml")
    xml_bad_path = xml_folder.joinpath("false_xml.xml")

    if path.is_cloud_path(files_path()):
        xml_path = xml_path.fspath
        xml_bad_path = xml_bad_path.fspath

    xml_ok = etree.parse(str(xml_path)).getroot()
    xml_nok = etree.parse(str(xml_bad_path)).getroot()

    ci.assert_xml_equal(xml_ok, xml_ok)
    with pytest.raises(AssertionError):
        ci.assert_xml_equal(xml_ok, xml_nok)


@s3_env
def test_assert_html():
    # HTML
    html_path = files_path().joinpath("productPreview.html")
    html_bad_path = files_path().joinpath("false.html")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if path.is_cloud_path(files_path()):
            html_path = html_path.download_to(tmp_dir)
            html_bad_path = html_bad_path.download_to(tmp_dir)

        html_ok = etree.parse(str(html_path)).getroot()
        html_nok = etree.parse(str(html_bad_path)).getroot()

        ci.assert_xml_equal(html_ok, html_ok)
        with pytest.raises(AssertionError):
            ci.assert_xml_equal(html_ok, html_nok)
