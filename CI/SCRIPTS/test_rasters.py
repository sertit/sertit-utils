""" Script testing the geo_utils """
import os
import tempfile
import rasterio
import numpy as np
import geopandas as gpd
from CI.SCRIPTS import script_utils
from sertit import rasters

RASTER_DATA = os.path.join(script_utils.get_ci_data_path(), "raster_utils")


def test_raster():
    """ Test raster functions """
    raster_path = os.path.join(RASTER_DATA, "raster.tif")
    raster_masked_path = os.path.join(RASTER_DATA, "raster_masked.tif")
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
    with tempfile.TemporaryDirectory(prefix=script_utils.get_ci_data_path()) as tmp_dir:
        # Get Extent
        extent = rasters.get_extent(raster_path)
        truth_extent = gpd.read_file(extent_path)
        script_utils.assert_geom_equal(extent, truth_extent)

        # Get Footprint
        footprint = rasters.get_footprint(raster_path)
        truth_footprint = gpd.read_file(footprint_path)
        script_utils.assert_geom_equal(footprint, truth_footprint)

        with rasterio.open(raster_path) as dst:
            # Read
            raster, meta = rasters.read(dst)
            assert raster.shape == (dst.count, dst.height, dst.width)
            assert meta["crs"] == dst.crs
            assert meta["transform"] == dst.transform

            # Write
            raster_out = os.path.join(tmp_dir, "test.tif")
            rasters.write(raster, raster_out, meta)
            assert os.path.isfile(raster_out)

            # Mask
            raster_masked_out = os.path.join(tmp_dir, "test_mask.tif")
            mask = gpd.read_file(mask_path)
            mask_arr, mask_tr = rasters.ma_mask(dst, mask.geometry, crop=True)
            rasters.write(mask_arr, raster_masked_out, meta, transform=mask_tr)

            # Sieve
            sieve_out = os.path.join(tmp_dir, "test_sieved.tif")
            sieve_arr, sieve_meta = rasters.sieve(raster, meta, sieve_thresh=20, connectivity=4)
            rasters.write(sieve_arr, sieve_out, sieve_meta, nodata=255)

            # Collocate
            coll_arr, coll_meta = rasters.collocate(meta, raster, meta)  # Just hope that it doesnt crash
            assert coll_meta == meta

            # Merge GTiff
            raster_merged_gtiff_out = os.path.join(tmp_dir, "test_merged.tif")
            rasters.merge_gtiff([raster_path, raster_to_merge_path], raster_merged_gtiff_out)

            # Merge VRT
            raster_merged_vrt_out = os.path.join(tmp_dir, "test_merged.vrt")
            rasters.merge_vrt([raster_path, raster_to_merge_path], raster_merged_vrt_out)

            # Vectorize
            vect = rasters.vectorize(raster_path)
            vect.to_file(os.path.join(tmp_dir, "test_vector.geojson"), driver="GeoJSON")
            vect_truth = gpd.read_file(vect_truth_path)
            equality = script_utils.assert_geom_equal(vect, vect_truth)
            # assert equality if isinstance(equality, bool) else equality.all()

        # Tests
        script_utils.assert_raster_equal(raster_path, raster_out)
        script_utils.assert_raster_equal(raster_masked_out, raster_masked_path)
        script_utils.assert_raster_equal(sieve_out, raster_sieved_path)
        script_utils.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)
        script_utils.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)


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
