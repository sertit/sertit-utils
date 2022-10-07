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
""" Script testing raster function (with rasterio) """
import os
import shutil
import tempfile

import numpy as np
import pytest
import rasterio
import shapely

from CI.SCRIPTS.script_utils import rasters_path, s3_env
from sertit import ci, rasters_rio, vectors

ci.reduce_verbosity()


@s3_env
def test_rasters_rio():
    """Test raster functions"""
    # Rasters
    raster_path = rasters_path().joinpath("raster.tif")
    raster_masked_path = rasters_path().joinpath("raster_masked.tif")
    raster_cropped_path = rasters_path().joinpath("raster_cropped.tif")
    raster_sieved_path = rasters_path().joinpath("raster_sieved.tif")
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_merged_gtiff_path = rasters_path().joinpath("raster_merged.tif")

    # Vectors
    mask_path = rasters_path().joinpath("raster_mask.geojson")
    extent_path = rasters_path().joinpath("extent.geojson")
    footprint_path = rasters_path().joinpath("footprint.geojson")
    if shapely.__version__ >= "1.8a1":
        vect_truth_path = rasters_path().joinpath("vector.geojson")
        diss_truth_path = rasters_path().joinpath("dissolved.geojson")
    else:
        print("USING OLD VECTORS")
        vect_truth_path = rasters_path().joinpath("vector_old.geojson")
        diss_truth_path = rasters_path().joinpath("dissolved_old.geojson")
    nodata_truth_path = rasters_path().joinpath("nodata.geojson")
    valid_truth_path = rasters_path().joinpath("valid.geojson")

    # Create tmp file
    # VRT needs to be build on te same disk
    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir = rasters_path().joinpath("OUTPUT_RASTERIO")
        # os.makedirs(tmp_dir, exist_ok=True)

        # Get Extent
        extent = rasters_rio.get_extent(raster_path)
        truth_extent = vectors.read(extent_path)
        ci.assert_geom_equal(extent, truth_extent)

        # Get Footprint
        footprint = rasters_rio.get_footprint(raster_path)
        truth_footprint = vectors.read(footprint_path)
        ci.assert_geom_equal(footprint, truth_footprint)

        # Read
        raster, meta = rasters_rio.read(raster_path)
        with rasterio.open(str(raster_path)) as dst:
            raster_1, meta1 = rasters_rio.read(dst, resolution=20)
            raster_2, _ = rasters_rio.read(dst, resolution=[20, 20])
            raster_3, _ = rasters_rio.read(dst, size=(meta1["width"], meta1["height"]))
            raster_4, _ = rasters_rio.read(raster_path, indexes=[1])
            with pytest.raises(ValueError):
                rasters_rio.read(dst, resolution=[20, 20, 20])

            assert raster.shape == (dst.count, dst.height, dst.width)
            assert meta["crs"] == dst.crs
            assert meta["transform"] == dst.transform
            np.testing.assert_array_equal(raster_1, raster_2)
            np.testing.assert_array_equal(raster_1, raster_3)
            np.testing.assert_array_equal(raster, raster_4)  # 2D array

            # Write
            raster_out = os.path.join(tmp_dir, "test.tif")
            rasters_rio.write(raster, meta, raster_out)
            assert os.path.isfile(raster_out)

            # Mask
            raster_masked_out = os.path.join(tmp_dir, "test_mask.tif")
            mask = vectors.read(mask_path)
            mask_arr, mask_meta = rasters_rio.mask(dst, mask.geometry)
            rasters_rio.write(mask_arr, mask_meta, raster_masked_out)

            # Crop
            raster_cropped_out = os.path.join(tmp_dir, "test_crop.tif")
            crop_arr, crop_meta = rasters_rio.crop(dst, mask)
            rasters_rio.write(crop_arr, crop_meta, raster_cropped_out)

            # Sieve
            sieve_out = os.path.join(tmp_dir, "test_sieved.tif")
            sieve_arr, sieve_meta = rasters_rio.sieve(
                raster, meta, sieve_thresh=20, connectivity=4
            )
            rasters_rio.write(sieve_arr, sieve_meta, sieve_out, nodata=255)

            # Collocate
            coll_arr, coll_meta = rasters_rio.collocate(
                meta, raster, meta
            )  # Just hope that it doesnt crash
            assert coll_meta == meta

            # Merge GTiff
            raster_merged_gtiff_out = os.path.join(tmp_dir, "test_merged.tif")
            rasters_rio.merge_gtiff(
                [raster_path, raster_to_merge_path],
                raster_merged_gtiff_out,
                method="max",
            )

            # Vectorize
            val = 2
            vect = rasters_rio.vectorize(raster_path)
            vect_val = rasters_rio.vectorize(raster_path, values=val)
            vect_val_diss = rasters_rio.vectorize(
                raster_path, values=val, dissolve=True
            )
            vect_val_disc = rasters_rio.vectorize(
                raster_path, values=[1, 255], keep_values=False
            )
            vect_truth = vectors.read(vect_truth_path)
            diss_truth = vectors.read(diss_truth_path)
            ci.assert_geom_equal(vect, vect_truth)
            ci.assert_geom_equal(vect_val, vect_truth.loc[vect_truth.raster_val == val])
            ci.assert_geom_equal(vect_val_diss, diss_truth)
            ci.assert_geom_equal(
                vect_val_disc, vect_truth.loc[vect_truth.raster_val == val]
            )

            # Get valid vec
            valid_vec = rasters_rio.get_valid_vector(raster_path)
            valid_truth = vectors.read(valid_truth_path)
            ci.assert_geom_equal(valid_vec, valid_truth)

            # Get nodata vec
            nodata_vec = rasters_rio.get_nodata_vector(raster_path)
            nodata_truth = vectors.read(nodata_truth_path)
            ci.assert_geom_equal(nodata_vec, nodata_truth)

        # Tests
        ci.assert_raster_equal(raster_path, raster_out)
        ci.assert_raster_equal(raster_masked_out, raster_masked_path)
        ci.assert_raster_equal(raster_cropped_out, raster_cropped_path)
        ci.assert_raster_equal(sieve_out, raster_sieved_path)
        ci.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)


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
        rasters_rio.merge_vrt(
            [raster_path, raster_to_merge_path], raster_merged_vrt_out
        )
        ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)

        os.remove(raster_merged_vrt_out)

        rasters_rio.merge_vrt(
            [raster_path, raster_to_merge_path], raster_merged_vrt_out, abs_path=True
        )
        ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)


def test_dim():
    """Test on BEAM-DIMAP function"""
    dim_path = rasters_path().joinpath("DIM.dim")
    assert rasters_rio.get_dim_img_path(dim_path) == rasters_path().joinpath(
        "DIM.data", "dim.img"
    )


def test_bit():
    """Test bit arrays"""
    np_ones = np.ones((1, 2, 2), dtype=np.uint16)
    ones = rasters_rio.read_bit_array(np_ones, bit_id=0)
    zeros = rasters_rio.read_bit_array(np_ones, bit_id=list(np.arange(1, 15)))
    assert (np_ones == ones).all()
    for arr in zeros:
        assert (np_ones == 1 + arr).all()


@s3_env
def test_dem_fct():
    """ Test DEM fct, ie. slope and hillshade"""
    # Paths IN
    dem_path = rasters_path().joinpath("dem.tif")
    hlsd_path = rasters_path().joinpath("hillshade.tif")
    slope_path = rasters_path().joinpath("slope.tif")
    slope_r_path = rasters_path().joinpath("slope_r.tif")
    slope_p_path = rasters_path().joinpath("slope_p.tif")

    # Create tmp file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Path OUT
        hlsd_path_out = os.path.join(tmp_dir, "hillshade_out.tif")
        slope_path_out = os.path.join(tmp_dir, "slope.tif")
        slope_r_path_out = os.path.join(tmp_dir, "slope_r.tif")
        slope_p_path_out = os.path.join(tmp_dir, "slope_p.tif")

        # Compute
        hlsd, meta = rasters_rio.hillshade(dem_path, 34.0, 45.2)
        rasters_rio.write(hlsd, meta, hlsd_path_out)

        slp, meta = rasters_rio.slope(dem_path)
        rasters_rio.write(slp, meta, slope_path_out)

        slp_r, meta = rasters_rio.slope(dem_path, in_pct=False, in_rad=True)
        rasters_rio.write(slp_r, meta, slope_r_path_out)

        slp_p, meta = rasters_rio.slope(dem_path, in_pct=True)
        rasters_rio.write(slp_p, meta, slope_p_path_out)

        # Test
        ci.assert_raster_almost_equal(hlsd_path, hlsd_path_out, decimal=4)
        ci.assert_raster_almost_equal(slope_path, slope_path_out, decimal=4)
        ci.assert_raster_almost_equal(slope_r_path, slope_r_path_out, decimal=4)
        ci.assert_raster_almost_equal(slope_p_path, slope_p_path_out, decimal=4)


@s3_env
def test_reproj():
    """ Test reproject fct """
    dem_path = rasters_path().joinpath(
        "Copernicus_DSM_10_N43_00_W003_00_DEM_resampled.tif"
    )
    raster_path = rasters_path().joinpath("raster.tif")
    reproj_path = rasters_path().joinpath("reproj_out.tif")

    with rasterio.open(str(dem_path)) as src:
        with rasterio.open(str(raster_path)) as dst:
            dst_arr, dst_meta = rasters_rio.reproject_match(
                dst.meta, src.read(), src.meta
            )

            # from dst
            assert dst.meta["driver"] == dst_meta["driver"]
            assert dst.meta["width"] == dst_meta["width"]
            assert dst.meta["height"] == dst_meta["height"]
            assert dst.meta["transform"] == dst_meta["transform"]

            # from src
            assert src.meta["count"] == dst_meta["count"]
            assert src.meta["nodata"] == dst_meta["nodata"]
            assert src.meta["dtype"] == dst_meta["dtype"]

            with tempfile.TemporaryDirectory() as tmp_dir:
                path_out = os.path.join(tmp_dir, "out.tif")
                rasters_rio.write(dst_arr, dst_meta, path_out)

                ci.assert_raster_almost_equal(path_out, reproj_path, decimal=4)


@s3_env
def test_rasterize():
    """ Test rasterize fct """
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_path = rasters_path().joinpath("raster.tif")
    raster_true_bin_path = rasters_path().joinpath("rasterized_bin.tif")
    raster_true_path = rasters_path().joinpath("rasterized.tif")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Binary vector
        out_bin_path = os.path.join(tmp_dir, "out_bin.tif")
        rast_bin, bin_meta = rasters_rio.rasterize(raster_path, vec_path)
        rasters_rio.write(rast_bin, bin_meta, out_bin_path)

        ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_path, decimal=4)

        # Vector
        out_path = os.path.join(tmp_dir, "out.tif")
        rast, meta = rasters_rio.rasterize(
            raster_path, vec_path, "raster_val", dtype=np.uint8
        )
        rasters_rio.write(rast, meta, out_path)

        ci.assert_raster_almost_equal(raster_true_path, out_path, decimal=4)
