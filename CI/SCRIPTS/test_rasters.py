""" Script testing the rasters """
import os
import tempfile

import pytest
import rasterio
import numpy as np
import xarray as xr
import geopandas as gpd
from CI.SCRIPTS.script_utils import RASTER_DATA, get_ci_data_path
from sertit import rasters, ci


def test_rasters():
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

    # Create tmp file
    # VRT needs to be build on te same disk
    with tempfile.TemporaryDirectory(prefix=get_ci_data_path()) as tmp_dir:
        # tmp_dir = os.path.join(RASTER_DATA, "OUTPUT_XARRAY")
        # os.makedirs(tmp_dir, exist_ok=True)

        # Get Extent
        extent = rasters.get_extent(raster_path)
        truth_extent = gpd.read_file(extent_path)
        ci.assert_geom_equal(extent, truth_extent)

        # Get Footprint
        footprint = rasters.get_footprint(raster_path)
        truth_footprint = gpd.read_file(footprint_path)
        ci.assert_geom_equal(footprint, truth_footprint)

        with rasterio.open(raster_path) as dst:
            # Read
            raster = rasters.read(raster_path)
            raster_1 = rasters.read(raster_path, resolution=dst.res[0])
            raster_2 = rasters.read(raster_path, resolution=[dst.res[0], dst.res[1]])
            raster_3 = rasters.read(raster_path, size=(raster_1.rio.width, raster_1.rio.height))
            with pytest.raises(ValueError):
                rasters.read(dst, resolution=[20, 20, 20])

            assert raster.shape == (dst.count, dst.height, dst.width)
            assert raster_1.rio.crs == dst.crs
            assert raster_1.rio.transform() == dst.transform
            np.testing.assert_array_equal(raster_1, raster_2)
            np.testing.assert_array_equal(raster_1, raster_3)

            # Write
            raster_out = os.path.join(tmp_dir, "test.tif")
            rasters.write(raster, raster_out)
            assert os.path.isfile(raster_out)

            # Mask
            raster_masked_out = os.path.join(tmp_dir, "test_mask.tif")
            mask = gpd.read_file(mask_path)
            mask_arr = rasters.mask(dst, mask.geometry)
            rasters.write(mask_arr, raster_masked_out)

            # Crop
            raster_cropped_out = os.path.join(tmp_dir, "test_crop.tif")
            crop = gpd.read_file(mask_path)
            crop_arr = rasters.crop(dst, crop.geometry)
            rasters.write(crop_arr, raster_cropped_out)

            # Sieve
            sieve_out = os.path.join(tmp_dir, "test_sieved.tif")
            sieve_arr = rasters.sieve(raster, sieve_thresh=20, connectivity=4)
            rasters.write(sieve_arr, sieve_out)

            # Collocate
            coll_arr = rasters.collocate(raster, raster)  # Just hope that it doesnt crash
            xr.testing.assert_equal(coll_arr, raster)

            # Merge GTiff
            raster_merged_gtiff_out = os.path.join(tmp_dir, "test_merged.tif")
            rasters.merge_gtiff([raster_path, raster_to_merge_path], raster_merged_gtiff_out, method="max")

            # Merge VRT
            raster_merged_vrt_out = os.path.join(tmp_dir, "test_merged.vrt")
            rasters.merge_vrt([raster_path, raster_to_merge_path], raster_merged_vrt_out)

            # Vectorize
            val = 2
            vect = rasters.vectorize(raster_path)
            vect_val = rasters.vectorize(raster_path, values=val)
            vect.to_file(os.path.join(tmp_dir, "test_vector.geojson"), driver="GeoJSON")
            vect_truth = gpd.read_file(vect_truth_path)
            ci.assert_geom_equal(vect, vect_truth)
            ci.assert_geom_equal(vect_val, vect_truth.loc[vect_truth.raster_val == val])

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
    assert (rasters.get_dim_img_path(dim_path) == os.path.join(RASTER_DATA, "DIM.data", "dim.img"))


def test_bit():
    """ Test bit arrays """
    np_ones = np.ones((1, 2, 2), dtype=np.uint16)
    ones = rasters.read_bit_array(np_ones, bit_id=0)
    zeros = rasters.read_bit_array(np_ones, bit_id=list(np.arange(1, 15)))
    assert (np_ones == ones).all()
    for arr in zeros:
        assert (np_ones == 1 + arr).all()
