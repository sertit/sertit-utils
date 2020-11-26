""" Script testing the geo_utils """
import os
import tempfile
import rasterio
import geopandas as gpd
from CI.SCRIPTS import script_utils
from sertit_utils.eo import raster_utils

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
    tmp_dir = tempfile.TemporaryDirectory(prefix=script_utils.get_ci_data_path())

    # Get Extent
    extent = raster_utils.get_extent(raster_path)
    truth_extent = gpd.read_file(extent_path)
    assert extent.geometry.equals(truth_extent.geometry)

    # Get Footprint
    footprint = raster_utils.get_footprint(raster_path)
    truth_footprint = gpd.read_file(footprint_path)
    assert footprint.geometry.equals(truth_footprint.geometry)

    with rasterio.open(raster_path) as dst:
        # Read
        raster, meta = raster_utils.read(dst)
        assert raster.shape == (dst.count, dst.height, dst.width)
        assert meta["crs"] == dst.crs
        assert meta["transform"] == dst.transform

        # Write
        raster_out = os.path.join(tmp_dir.name, "test.tif")
        raster_utils.write(raster, raster_out, meta)
        assert os.path.isfile(raster_out)

        # Mask
        raster_masked_out = os.path.join(tmp_dir.name, "test_mask.tif")
        mask = gpd.read_file(mask_path)
        mask_arr, mask_tr = raster_utils.ma_mask(dst, mask.geometry, crop=True)
        raster_utils.write(mask_arr, raster_masked_out, meta, transform=mask_tr)

        # Sieve
        sieve_out = os.path.join(script_utils.get_ci_data_path(), "test_sieved.tif")
        sieve_arr, sieve_meta = raster_utils.sieve(raster, meta, sieve_thresh=20, connectivity=4)
        raster_utils.write(sieve_arr, sieve_out, sieve_meta, nodata=255)

        # Collocate
        coll_arr, coll_meta = raster_utils.collocate(meta, raster, meta)  # Just hope that it doesnt crash
        assert coll_meta == meta

        # Merge GTiff
        raster_merged_gtiff_out = os.path.join(tmp_dir.name, "test_merged.tif")
        raster_utils.merge_gtiff([raster_path, raster_to_merge_path], raster_merged_gtiff_out)

        # Merge VRT
        raster_merged_vrt_out = os.path.join(tmp_dir.name, "test_merged.vrt")
        raster_utils.merge_vrt([raster_path, raster_to_merge_path], raster_merged_vrt_out)

        # Vectorize
        vect = raster_utils.vectorize(raster_path)
        vect.to_file(os.path.join(tmp_dir.name, "test_vector.geojson"), driver="GeoJSON")
        vect_truth = gpd.read_file(vect_truth_path)
        equality = vect.geometry.equals(vect_truth.geometry)
        assert equality if isinstance(equality, bool) else equality.all()

    # Tests
    script_utils.assert_raster_equals(raster_path, raster_out)
    script_utils.assert_raster_equals(raster_masked_out, raster_masked_path)
    script_utils.assert_raster_equals(sieve_out, raster_sieved_path)
    script_utils.assert_raster_equals(raster_merged_gtiff_out, raster_merged_gtiff_path)
    script_utils.assert_raster_equals(raster_merged_vrt_out, raster_merged_vrt_path)

    # Cleanup
    tmp_dir.cleanup()


def test_dim():
    """ Test on BEAM-DIMAP function """
    dim_path = os.path.join(RASTER_DATA, "DIM.dim")
    assert (raster_utils.get_dim_img_path(dim_path) == os.path.join(RASTER_DATA, "DIM.data", "dim.img"))
