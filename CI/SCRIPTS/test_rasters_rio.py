""" Script testing the rasters """
import os
import tempfile

import pytest
import rasterio
import numpy as np
import geopandas as gpd
from CI.SCRIPTS.script_utils import RASTER_DATA, get_ci_data_path
from sertit import rasters_rio, ci


def test_rasters_rio():
    """ Test raster functions """
    raster_path = os.path.join(RASTER_DATA, "raster.tif")
    raster_masked_path = os.path.join(RASTER_DATA, "raster_masked.tif")
    raster_cropped_path = os.path.join(RASTER_DATA, "raster_cropped.tif")
    raster_sieved_path = os.path.join(RASTER_DATA, "raster_sieved.tif")
    raster_to_merge_path = os.path.join(RASTER_DATA, "raster_to_merge.tif")
    raster_merged_gtiff_path = os.path.join(RASTER_DATA, "raster_merged.tif")
    raster_merged_vrt_path = os.path.join(RASTER_DATA, "raster_merged.vrt")
    mask_path = os.path.join(RASTER_DATA, "raster_mask.geojson")
    extent_path = os.path.join(RASTER_DATA, "extent.geojson")
    footprint_path = os.path.join(RASTER_DATA, "footprint.geojson")
    vect_truth_path = os.path.join(RASTER_DATA, "vector.geojson")
    nodata_truth_path = os.path.join(RASTER_DATA, "nodata.geojson")
    valid_truth_path = os.path.join(RASTER_DATA, "valid.geojson")

    # Create tmp file
    # VRT needs to be build on te same disk
    with tempfile.TemporaryDirectory(prefix=get_ci_data_path()) as tmp_dir:
        # Get Extent
        extent = rasters_rio.get_extent(raster_path)
        truth_extent = gpd.read_file(extent_path)
        ci.assert_geom_equal(extent, truth_extent)

        # Get Footprint
        footprint = rasters_rio.get_footprint(raster_path)
        truth_footprint = gpd.read_file(footprint_path)
        ci.assert_geom_equal(footprint, truth_footprint)

        with rasterio.open(raster_path) as dst:
            # Read
            raster, meta = rasters_rio.read(dst)
            raster_1, meta1 = rasters_rio.read(dst, resolution=20)
            raster_2, _ = rasters_rio.read(dst, resolution=[20, 20])
            raster_3, _ = rasters_rio.read(dst, size=(meta1["width"], meta1["height"]))
            with pytest.raises(ValueError):
                rasters_rio.read(dst, resolution=[20, 20, 20])

            assert raster.shape == (dst.count, dst.height, dst.width)
            assert meta["crs"] == dst.crs
            assert meta["transform"] == dst.transform
            np.testing.assert_array_equal(raster_1, raster_2)
            np.testing.assert_array_equal(raster_1, raster_3)

            # Write
            raster_out = os.path.join(tmp_dir, "test.tif")
            rasters_rio.write(raster, raster_out, meta)
            assert os.path.isfile(raster_out)

            # Mask
            raster_masked_out = os.path.join(tmp_dir, "test_mask.tif")
            mask = gpd.read_file(mask_path)
            mask_arr, mask_meta = rasters_rio.mask(dst, mask.geometry)
            rasters_rio.write(mask_arr, raster_masked_out, mask_meta)

            # Crop
            raster_cropped_out = os.path.join(tmp_dir, "test_crop.tif")
            crop = gpd.read_file(mask_path)
            crop_arr, crop_meta = rasters_rio.crop(dst, crop.geometry)
            rasters_rio.write(crop_arr, raster_cropped_out, crop_meta)

            # Sieve
            sieve_out = os.path.join(tmp_dir, "test_sieved.tif")
            sieve_arr, sieve_meta = rasters_rio.sieve(raster, meta, sieve_thresh=20, connectivity=4)
            rasters_rio.write(sieve_arr, sieve_out, sieve_meta, nodata=255)

            # Collocate
            coll_arr, coll_meta = rasters_rio.collocate(meta, raster, meta)  # Just hope that it doesnt crash
            assert coll_meta == meta

            # Merge GTiff
            raster_merged_gtiff_out = os.path.join(tmp_dir, "test_merged.tif")
            rasters_rio.merge_gtiff([raster_path, raster_to_merge_path], raster_merged_gtiff_out, method="max")

            # Merge VRT
            raster_merged_vrt_out = os.path.join(tmp_dir, "test_merged.vrt")
            rasters_rio.merge_vrt([raster_path, raster_to_merge_path], raster_merged_vrt_out)

            # Vectorize
            val = 2
            vect = rasters_rio.vectorize(raster_path)
            vect_val = rasters_rio.vectorize(raster_path, values=val)
            vect.to_file(os.path.join(tmp_dir, "test_vector.geojson"), driver="GeoJSON")
            vect_truth = gpd.read_file(vect_truth_path)
            ci.assert_geom_equal(vect, vect_truth)
            ci.assert_geom_equal(vect_val, vect_truth.loc[vect_truth.raster_val == val])

            # Get valid vec
            valid_vec = rasters_rio.get_valid_vector(raster_path)
            valid_truth = gpd.read_file(valid_truth_path)
            ci.assert_geom_equal(valid_vec, valid_truth)

            # Get nodata vec
            nodata_vec = rasters_rio.get_nodata_vector(raster_path)
            nodata_truth = gpd.read_file(nodata_truth_path)
            ci.assert_geom_equal(nodata_vec, nodata_truth)


        # Tests
        ci.assert_raster_equal(raster_path, raster_out)
        ci.assert_raster_equal(raster_masked_out, raster_masked_path)
        ci.assert_raster_equal(raster_cropped_out, raster_cropped_path)
        ci.assert_raster_equal(sieve_out, raster_sieved_path)
        ci.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)
        ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)


def test_dim():
    """ Test on BEAM-DIMAP function """
    dim_path = os.path.join(RASTER_DATA, "DIM.dim")
    assert (rasters_rio.get_dim_img_path(dim_path) == os.path.join(RASTER_DATA, "DIM.data", "dim.img"))


def test_bit():
    """ Test bit arrays """
    np_ones = np.ones((1, 2, 2), dtype=np.uint16)
    ones = rasters_rio.read_bit_array(np_ones, bit_id=0)
    zeros = rasters_rio.read_bit_array(np_ones, bit_id=list(np.arange(1, 15)))
    assert (np_ones == ones).all()
    for arr in zeros:
        assert (np_ones == 1 + arr).all()