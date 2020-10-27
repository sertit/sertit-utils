""" Script testing the geo_utils """
import os
import tempfile
import rasterio
import geopandas as gpd
from CI.SCRIPTS import script_utils
from sertit_utils.eo import raster_utils


def test_raster():
    """ Test raster functions """
    raster_path = os.path.join(script_utils.get_ci_data_path(), "raster.tif")
    raster_masked_path = os.path.join(script_utils.get_ci_data_path(), "raster_masked.tif")
    raster_sieved_path = os.path.join(script_utils.get_ci_data_path(), "raster_sieved.tif")
    vect_truth_path = os.path.join(script_utils.get_ci_data_path(), "vector.geojson")
    tmp_dir = tempfile.TemporaryDirectory()

    # Get Extent
    extent = raster_utils.get_extent(raster_path)
    assert str(extent.envelope[0]) == 'POLYGON ((630000 4864020, 636000 4864020, ' \
                                      '636000 4870020, 630000 4870020, 630000 4864020))'

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
        mask_path = os.path.join(script_utils.get_ci_data_path(), "raster_mask.geojson")
        raster_masked_out = os.path.join(tmp_dir.name, "test_mask.tif")
        mask = gpd.read_file(mask_path)
        mask_arr, mask_tr = raster_utils.ma_mask(dst, mask.envelope, crop=False)
        raster_utils.write(mask_arr, raster_masked_out, meta, transform=mask_tr)

        # Sieve
        sieve_out = os.path.join(script_utils.get_ci_data_path(), "raster_sieved.tif")
        sieve_arr, sieve_meta = raster_utils.sieve(raster, meta, sieve_thresh=20, connectivity=4)
        raster_utils.write(sieve_arr, sieve_out, sieve_meta, nodata=255)

        # Collocate
        coll_arr, coll_meta = raster_utils.collocate(meta, raster, meta)  # Just hope that it doesnt crash
        assert coll_meta == meta

        # Vectorize
        vect = raster_utils.vectorize(raster_path)
        vect_truth = gpd.read_file(vect_truth_path)
        equality = vect.envelope.equals(vect_truth.envelope)
        assert equality if isinstance(equality, bool) else equality.all()

    # Tests
    script_utils.assert_raster_equals(raster_path, raster_out)
    script_utils.assert_raster_equals(raster_masked_out, raster_masked_path)
    script_utils.assert_raster_equals(sieve_out, raster_sieved_path)

    # Cleanup
    tmp_dir.cleanup()


def test_dim():
    """ Test on BEAM-DIMAP function """
    dim_path = os.path.join(script_utils.get_ci_data_path(), "DIM.dim")
    assert (raster_utils.get_dim_img_path(dim_path) ==
            os.path.join(script_utils.get_ci_data_path(), "DIM.data", "dim.img"))
