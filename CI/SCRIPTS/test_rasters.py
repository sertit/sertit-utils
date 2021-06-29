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
""" Script testing raster functions (with XARRAY) """
import os
import shutil
import tempfile

import numpy as np
import pytest
import rasterio
import xarray as xr

from CI.SCRIPTS.script_utils import rasters_path, s3_env
from sertit import ci, files, rasters, vectors


@s3_env
def test_rasters():
    """Test raster functions"""

    # Rasters
    raster_path = rasters_path().joinpath("raster.tif")
    raster_masked_path = rasters_path().joinpath("raster_masked.tif")
    raster_cropped_xarray_path = rasters_path().joinpath("raster_cropped_xarray.tif")
    raster_sieved_path = rasters_path().joinpath("raster_sieved.tif")
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_merged_gtiff_path = rasters_path().joinpath("raster_merged.tif")

    # Vectors
    mask_path = rasters_path().joinpath("raster_mask.geojson")
    extent_path = rasters_path().joinpath("extent.geojson")
    footprint_path = rasters_path().joinpath("footprint.geojson")
    vect_truth_path = rasters_path().joinpath("vector.geojson")
    diss_truth_path = rasters_path().joinpath("dissolved.geojson")
    nodata_truth_path = rasters_path().joinpath("nodata.geojson")
    valid_truth_path = rasters_path().joinpath("valid.geojson")

    # Create tmp file
    # VRT needs to be build on te same disk
    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir = rasters_path().joinpath("OUTPUT_XARRAY")
        # os.makedirs(tmp_dir, exist_ok=True)

        # Get Extent
        extent = rasters.get_extent(raster_path)
        truth_extent = vectors.read(extent_path)
        ci.assert_geom_equal(extent, truth_extent)

        # Get Footprint
        footprint = rasters.get_footprint(raster_path)
        truth_footprint = vectors.read(footprint_path)
        ci.assert_geom_equal(footprint, truth_footprint)

        with rasterio.open(str(raster_path)) as dst:
            dst_dtype = dst.meta["dtype"]
            # Read
            xda = rasters.read(raster_path)
            xda_1 = rasters.read(raster_path, resolution=dst.res[0])
            xda_2 = rasters.read(raster_path, resolution=[dst.res[0], dst.res[1]])
            xda_3 = rasters.read(raster_path, size=(xda_1.rio.width, xda_1.rio.height))
            xda_4 = rasters.read(raster_path, resolution=dst.res[0] / 2)
            xda_5 = rasters.read(raster_path, indexes=1)
            assert xda_4.shape[-2] == xda.shape[-2] * 2
            assert xda_4.shape[-1] == xda.shape[-1] * 2
            with pytest.raises(ValueError):
                rasters.read(dst, resolution=[20, 20, 20])

            # Create xr.Dataset
            name = files.get_filename(dst.name)
            xds = xr.Dataset({name: xda})

            assert xda.shape == (dst.count, dst.height, dst.width)
            assert xda.encoding["dtype"] == dst_dtype
            assert xds[name].shape == xda.shape
            assert xda_1.rio.crs == dst.crs
            assert xda_1.rio.transform() == dst.transform
            np.testing.assert_array_equal(xda_1, xda_2)
            np.testing.assert_array_equal(xda_1, xda_3)
            np.testing.assert_array_equal(xda_1, xda_5)

            # Write
            xda_out = os.path.join(tmp_dir, "test_xda.tif")
            xds_out = os.path.join(tmp_dir, "test_xds.tif")
            rasters.write(xda, xda_out, dtype=dst_dtype)
            rasters.write(xds, xds_out, dtype=dst_dtype)
            assert os.path.isfile(xda_out)

            # Mask
            xda_masked = os.path.join(tmp_dir, "test_mask_xda.tif")
            xds_masked = os.path.join(tmp_dir, "test_mask_xds.tif")
            mask = vectors.read(mask_path)
            mask_xda = rasters.mask(xda, mask.geometry)
            mask_xds = rasters.mask(xda, mask)
            rasters.write(mask_xda, xda_masked, dtype=np.uint8)
            rasters.write(mask_xds, xds_masked, dtype=np.uint8)

            # Paint
            xda_paint_true = os.path.join(tmp_dir, "test_paint_true_xda.tif")
            xda_paint_false = os.path.join(tmp_dir, "test_paint_false_xda.tif")
            xds_paint_true = os.path.join(tmp_dir, "test_paint_true_xds.tif")
            xds_paint_false = os.path.join(tmp_dir, "test_paint_false_xds.tif")
            mask = vectors.read(mask_path)
            paint_true_xda = rasters.paint(xda, mask.geometry, value=600, invert=True)
            paint_false_xda = rasters.paint(xda, mask.geometry, value=600, invert=False)
            paint_true_xds = rasters.paint(xda, mask, value=600, invert=True)
            paint_false_xds = rasters.paint(xda, mask, value=600, invert=False)
            rasters.write(paint_true_xda, xda_paint_true, dtype=np.uint8)
            rasters.write(paint_false_xda, xda_paint_false, dtype=np.uint8)
            rasters.write(paint_true_xds, xds_paint_true, dtype=np.uint8)
            rasters.write(paint_false_xds, xds_paint_false, dtype=np.uint8)

            # Crop
            xda_cropped = os.path.join(tmp_dir, "test_crop_xda.tif")
            xds_cropped = os.path.join(tmp_dir, "test_crop_xds.tif")
            crop_xda = rasters.crop(xda, mask.geometry)
            crop_xds = rasters.crop(xds, mask)
            rasters.write(crop_xda, xda_cropped, dtype=np.uint8)
            rasters.write(crop_xds, xds_cropped, dtype=np.uint8)

            # Sieve
            xda_sieved = os.path.join(tmp_dir, "test_sieved_xda.tif")
            xds_sieved = os.path.join(tmp_dir, "test_sieved_xds.tif")
            sieve_xda = rasters.sieve(xda, sieve_thresh=20, connectivity=4)
            sieve_xds = rasters.sieve(xds, sieve_thresh=20, connectivity=4)
            rasters.write(sieve_xda, xda_sieved, dtype=np.uint8)
            rasters.write(sieve_xds, xds_sieved, dtype=np.uint8)

            # Collocate
            coll_xda = rasters.collocate(xda, xda)  # Just hope that it doesnt crash
            xr.testing.assert_equal(coll_xda, xda)
            coll_xds = rasters.collocate(xds, xds)  # Just hope that it doesnt crash
            xr.testing.assert_equal(coll_xds, xds)

            # Merge GTiff
            raster_merged_gtiff_out = os.path.join(tmp_dir, "test_merged.tif")
            rasters.merge_gtiff(
                [raster_path, raster_to_merge_path],
                raster_merged_gtiff_out,
                method="max",
            )

            # Vectorize
            val = 2
            vect = rasters.vectorize(raster_path)
            vect_xds = rasters.vectorize(xds)
            vect_val = rasters.vectorize(raster_path, values=val)
            vect_val_diss = rasters.vectorize(raster_path, values=val, dissolve=True)
            vect_val_disc = rasters.vectorize(
                raster_path, values=[1, 255], keep_values=False
            )
            vect.to_file(os.path.join(tmp_dir, "test_vector.geojson"), driver="GeoJSON")
            vect_truth = vectors.read(vect_truth_path)
            diss_truth = vectors.read(diss_truth_path)
            ci.assert_geom_equal(vect, vect_truth)
            ci.assert_geom_equal(vect_xds[name], vect_truth)
            ci.assert_geom_equal(vect_val, vect_truth.loc[vect_truth.raster_val == val])
            ci.assert_geom_equal(vect_val_diss, diss_truth)
            ci.assert_geom_equal(
                vect_val_disc, vect_truth.loc[vect_truth.raster_val == val]
            )

            # Get valid vec
            valid_vec = rasters.get_valid_vector(raster_path)
            valid_vec_xds = rasters.get_valid_vector(xds)
            valid_truth = vectors.read(valid_truth_path)
            ci.assert_geom_equal(valid_vec, valid_truth)
            ci.assert_geom_equal(valid_vec_xds[name], valid_truth)

            # Get nodata vec
            nodata_vec = rasters.get_nodata_vector(raster_path)
            nodata_vec_xds = rasters.get_nodata_vector(xds)
            nodata_truth = vectors.read(nodata_truth_path)
            ci.assert_geom_equal(nodata_vec, nodata_truth)
            ci.assert_geom_equal(nodata_vec_xds[name], nodata_truth)

        # Tests
        ci.assert_raster_equal(raster_path, xda_out)
        ci.assert_raster_equal(xda_masked, raster_masked_path)
        ci.assert_raster_equal(xda_cropped, raster_cropped_xarray_path)
        ci.assert_raster_equal(xda_sieved, raster_sieved_path)
        ci.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)

        ci.assert_raster_equal(raster_path, xds_out)
        ci.assert_raster_equal(xds_masked, raster_masked_path)
        ci.assert_raster_equal(xds_cropped, raster_cropped_xarray_path)
        ci.assert_raster_equal(xds_sieved, raster_sieved_path)


@s3_env
@pytest.mark.skipif(
    shutil.which("gdalbuildvrt") is None,
    reason="Only works if gdalbuildvrt can be found.",
)
def test_vrt():
    raster_merged_vrt_path = rasters_path().joinpath("raster_merged.vrt")
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_path = rasters_path().joinpath("raster.tif")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Merge VRT
        raster_merged_vrt_out = os.path.join(tmp_dir, "test_merged.vrt")
        rasters.merge_vrt([raster_path, raster_to_merge_path], raster_merged_vrt_out)
        ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)


@s3_env
def test_write():
    raster_path = rasters_path().joinpath("raster.tif")
    raster_xds = rasters.read(raster_path)

    nodata = {
        np.uint8: 255,
        np.int8: -128,
        np.uint16: 65535,
        np.int16: -9999,
        np.uint32: 65535,
        np.int32: 65535,
        np.float32: -9999,
        np.float64: -9999,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = os.path.join(tmp_dir, "test_nodata.tif")

        for dtype, nodata_val in nodata.items():
            print(dtype.__name__)
            rasters.write(raster_xds, test_path, dtype=dtype)
            with rasterio.open(test_path) as ds:
                assert ds.meta["dtype"] == dtype or ds.meta["dtype"] == dtype.__name__
                assert ds.meta["nodata"] == nodata_val


def test_dim():
    """Test on BEAM-DIMAP function"""
    dim_path = rasters_path().joinpath("DIM.dim")
    assert rasters.get_dim_img_path(dim_path) == rasters_path().joinpath(
        "DIM.data", "dim.img"
    )


def test_bit():
    """Test bit arrays"""
    # Bit
    np_ones = xr.DataArray(np.ones((1, 2, 2), dtype=np.uint16))
    ones = rasters.read_bit_array(np_ones, bit_id=0)
    zeros = rasters.read_bit_array(np_ones, bit_id=list(np.arange(1, 15)))
    assert (np_ones.data == ones).all()
    for arr in zeros:
        assert (np_ones.data == 1 + arr).all()

    # Bit
    np_ones = xr.DataArray(np.ones((1, 2, 2), dtype=np.uint8))
    ones = rasters.read_bit_array(np_ones, bit_id=0)
    zeros = rasters.read_bit_array(np_ones, bit_id=list(np.arange(1, 7)))
    assert (np_ones.data == ones).all()
    for arr in zeros:
        assert (np_ones.data == 1 + arr).all()

    # Bit
    np_ones = xr.DataArray(np.ones((1, 2, 2), dtype=np.uint32))
    ones = rasters.read_bit_array(np_ones, bit_id=0)
    zeros = rasters.read_bit_array(np_ones, bit_id=list(np.arange(1, 31)))
    assert (np_ones.data == ones).all()
    for arr in zeros:
        assert (np_ones.data == 1 + arr).all()

    # uint8
    np_ones = xr.DataArray(np.ones((1, 2, 2), dtype=np.uint8))
    ones = rasters.read_uint8_array(np_ones, bit_id=0)
    zeros = rasters.read_uint8_array(np_ones, bit_id=list(np.arange(1, 7)))
    assert (np_ones.data == ones).all()
    for arr in zeros:
        assert (np_ones.data == 1 + arr).all()

    # uint8 from floats
    np_ones = xr.DataArray(np.ones((1, 2, 2), dtype=float))
    ones = rasters.read_uint8_array(np_ones, bit_id=0)
    zeros = rasters.read_uint8_array(np_ones, bit_id=list(np.arange(1, 7)))
    assert (np_ones.data == ones).all()
    for arr in zeros:
        assert (np_ones.data == 1 + arr).all()


@s3_env
def test_xarray_fct():
    """ Test xarray functions """
    # Set nodata
    A = xr.DataArray(dims=("x", "y"), data=[[1, 0, 0], [0, 0, 0]])
    nodata = xr.DataArray(
        dims=("x", "y"), data=[[1, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )
    A_nodata = rasters.set_nodata(A, 0)

    xr.testing.assert_equal(A_nodata, nodata)

    # Mtd
    raster_path = rasters_path().joinpath("raster.tif")
    xda = rasters.read(raster_path)
    xda_sum = xda + xda
    xda_sum = rasters.set_metadata(xda_sum, xda, "sum")

    assert xda_sum.rio.crs == xda.rio.crs
    assert np.isnan(xda_sum.rio.nodata)
    assert xda_sum.rio.encoded_nodata == xda.rio.encoded_nodata
    assert xda_sum.attrs == xda.attrs
    assert xda_sum.encoding == xda.encoding
    assert xda_sum.rio.transform() == xda.rio.transform()
    assert xda_sum.rio.width == xda.rio.width
    assert xda_sum.rio.height == xda.rio.height
    assert xda_sum.rio.count == xda.rio.count
    assert xda_sum.name == "sum"


def test_where():
    """ Test overloading of xr.where function """
    A = xr.DataArray(dims=("x", "y"), data=[[1, 0, 5], [np.nan, 0, 0]])
    mask_A = rasters.where(A > 3, 0, 1, A, new_name="mask_A")

    np.testing.assert_equal(np.isnan(A.data), np.isnan(mask_A.data))
    assert A.attrs == mask_A.attrs
    np.testing.assert_equal(
        mask_A.data, np.array([[1.0, 1.0, 0.0], [np.nan, 1.0, 1.0]])
    )
