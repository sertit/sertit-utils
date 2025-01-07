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
"""Script testing raster functions (with XARRAY)"""

import logging
import os
import shutil

import numpy as np
import pytest
import rasterio
import shapely
import xarray as xr

from ci.script_utils import KAPUT_KWARGS, dask_env, rasters_path, s3_env
from sertit import ci, path, rasters, unistra, vectors
from sertit.rasters import (
    FLOAT_NODATA,
    INT8_NODATA,
    UINT8_NODATA,
    UINT16_NODATA,
    any_raster_to_xr_ds,
    get_nodata_value_from_dtype,
    get_nodata_value_from_xr,
)
from sertit.vectors import EPSG_4326

ci.reduce_verbosity()


def test_indexes(caplog):
    @s3_env
    @dask_env
    def test_core():
        l1_path = rasters_path().joinpath("19760712T093233_L1_215030_MSS_stack.tif")
        xda_raw = rasters.read(l1_path, **KAPUT_KWARGS)
        xda_idx = rasters.read(l1_path, indexes=1)
        np.testing.assert_array_equal(xda_raw[[0], :], xda_idx)

        xda_idx2 = rasters.read(l1_path, indexes=[3, 2])
        xda_raw2 = np.concatenate((xda_raw.data[[2], :], xda_raw.data[[1], :]))
        np.testing.assert_array_equal(xda_raw2, xda_idx2)

        with pytest.raises(ValueError):
            rasters.read(l1_path, indexes=0)

        with caplog.at_level(logging.WARNING):
            idx = [4, 5]
            rasters.read(l1_path, indexes=idx)
            assert f"Non available index: {idx}" in caplog.text

    test_core()


@pytest.fixture
def raster_path():
    return rasters_path().joinpath("raster.tif")


@pytest.fixture
def dem_path():
    return rasters_path().joinpath("dem.tif")


@pytest.fixture
def xda(raster_path):
    return rasters.read(raster_path)


@pytest.fixture
def xda_dask(raster_path):
    return rasters.read(raster_path, chunks=True)


@pytest.fixture
def mask_path():
    return rasters_path().joinpath("raster_mask.geojson")


@pytest.fixture
def mask(mask_path):
    return vectors.read(mask_path)


@pytest.fixture
def ds_name(raster_path):
    with rasterio.open(str(raster_path)) as ds:
        return path.get_filename(ds.name)


@pytest.fixture
def ds_dtype(raster_path):
    with rasterio.open(str(raster_path)) as ds:
        return getattr(np, ds.meta["dtype"])


@pytest.fixture
def xds(raster_path, xda, ds_name):
    return xr.Dataset({ds_name: xda})


@s3_env
@dask_env
def test_read(tmp_path, raster_path, xda, xds, xda_dask, ds_name, ds_dtype):
    """Test read function"""
    extent_path = rasters_path().joinpath("extent.geojson")
    footprint_path = rasters_path().joinpath("footprint.geojson")

    # Get Extent
    extent = rasters.get_extent(raster_path)
    truth_extent = vectors.read(extent_path)
    ci.assert_geom_equal(extent, truth_extent)

    # Get Footprint
    footprint = rasters.get_footprint(raster_path)
    truth_footprint = vectors.read(footprint_path)
    ci.assert_geom_equal(footprint, truth_footprint)

    with rasterio.open(str(raster_path)) as ds:
        # ----------------------------------------------------------------------------------------------
        # -- Read
        xda_1 = rasters.read(raster_path, resolution=ds.res[0])
        xda_2 = rasters.read(raster_path, resolution=[ds.res[0], ds.res[1]])
        xda_3 = rasters.read(raster_path, size=(xda_1.rio.width, xda_1.rio.height))
        xda_4 = rasters.read(raster_path, resolution=ds.res[0] / 2)
        xda_5 = rasters.read(xda)
        assert xda.chunks is not None

        assert xda_dask.chunks is not None

        # Test shape (link between resolution and size)
        assert xda_4.shape[-2] == xda.shape[-2] * 2
        assert xda_4.shape[-1] == xda.shape[-1] * 2
        with pytest.raises(ValueError):
            rasters.read(ds, resolution=[20, 20, 20])

        # Test dataset integrity
        assert xda.shape == (ds.count, ds.height, ds.width)
        assert xda.encoding["dtype"] == ds_dtype
        assert xds[ds_name].shape == xda.shape
        assert xda_1.rio.crs == ds.crs
        assert xda_1.rio.transform() == ds.transform
        np.testing.assert_array_equal(xda_1, xda_2)
        np.testing.assert_array_equal(xda_1, xda_3)
        np.testing.assert_array_equal(xda, xda_5)
        np.testing.assert_array_equal(xda, xda_dask)

        ci.assert_xr_encoding_attrs(xda, xda_1)
        ci.assert_xr_encoding_attrs(xda, xda_2)
        ci.assert_xr_encoding_attrs(xda, xda_3)
        ci.assert_xr_encoding_attrs(xda, xda_4)
        ci.assert_xr_encoding_attrs(xda, xda_5)
        ci.assert_xr_encoding_attrs(xda, xda_dask, unchecked_attr="preferred_chunks")


@s3_env
@dask_env
def test_read_with_window(tmp_path, raster_path, mask_path, mask):
    """Test read (with window) function"""
    raster_window_path = rasters_path().joinpath("window.tif")
    raster_window_20_path = rasters_path().joinpath("window_20.tif")

    xda_window_out = os.path.join(tmp_path, "test_xda_window.tif")
    xda_window = rasters.read(
        raster_path,
        window=mask_path,
    )
    assert xda_window.chunks is not None
    rasters.write(xda_window, xda_window_out, dtype=np.uint8)
    ci.assert_raster_equal(xda_window_out, raster_window_path)

    xda_window_20_out = os.path.join(tmp_path, "test_xda_20_window.tif")
    gdf = mask.to_crs(EPSG_4326)
    xda_window_20 = rasters.read(raster_path, window=gdf, resolution=20)
    rasters.write(xda_window_20, xda_window_20_out, dtype=np.uint8)
    ci.assert_raster_equal(xda_window_20_out, raster_window_20_path)

    with pytest.raises(FileNotFoundError):
        rasters.read(
            raster_path,
            window=rasters_path().joinpath("non_existing_window.kml"),
        )


@s3_env
@dask_env
@pytest.mark.timeout(2, func_only=True)
def test_very_big_file(tmp_path, mask_path):
    """Test read with big files function (should be fast, if not there is something going on...)"""
    dem_path = unistra.get_geodatastore() / "GLOBAL" / "EUDEM_v2" / "eudem_wgs84.tif"
    aoi_path = rasters_path() / "DAX.shp"
    with rasterio.open(dem_path) as ds:
        dem_res = ds.res[0]  # noqa
        dem_crs = ds.crs  # noqa

    aoi = vectors.read(aoi_path)
    rasters.crop(rasters.read(dem_path, window=aoi), aoi, nodata=0)


@s3_env
@dask_env
def test_write_basic(tmp_path, raster_path, xda, xds, xda_dask, ds_dtype):
    """Test write (basic) function"""
    # DataArray
    xda_out = os.path.join(tmp_path, "test_xda.tif")
    rasters.write(xda, xda_out, dtype=ds_dtype)
    assert os.path.isfile(xda_out)

    # Dataset
    xds_out = os.path.join(tmp_path, "test_xds.tif")
    rasters.write(xds, xds_out, dtype=ds_dtype)
    assert os.path.isfile(xds_out)

    # With dask
    xda_dask_out = os.path.join(tmp_path, "test_xda_dask.tif")
    rasters.write(xda_dask, xda_dask_out, dtype=ds_dtype)
    assert os.path.isfile(xda_dask_out)

    # Tests
    ci.assert_raster_equal(raster_path, xda_out)
    ci.assert_raster_equal(raster_path, xds_out)
    ci.assert_raster_equal(raster_path, xda_dask_out)


@s3_env
@dask_env
def test_mask(tmp_path, xda, xds, xda_dask, mask):
    """Test mask function"""
    # DataArray
    xda_masked = os.path.join(tmp_path, "test_mask_xda.tif")
    mask_xda = rasters.mask(xda, mask.geometry, **KAPUT_KWARGS)
    rasters.write(mask_xda, xda_masked, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, mask_xda)

    # Dataset
    xds_masked = os.path.join(tmp_path, "test_mask_xds.tif")
    mask_xds = rasters.mask(xds, mask)
    rasters.write(mask_xds, xds_masked, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, mask_xds)

    # With dask
    mask_xda_dask = rasters.mask(xda_dask, mask)
    assert mask_xda_dask.chunks is not None
    np.testing.assert_array_equal(mask_xda, mask_xda_dask)
    ci.assert_xr_encoding_attrs(xda_dask, mask_xda_dask)

    raster_masked_path = rasters_path().joinpath("raster_masked.tif")
    ci.assert_raster_equal(xda_masked, raster_masked_path)
    ci.assert_raster_equal(xds_masked, raster_masked_path)


@s3_env
@dask_env
def test_paint(tmp_path, xda, xds, xda_dask, mask):
    """Test paint function"""
    # DataArray
    xda_paint_true = os.path.join(tmp_path, "test_paint_true_xda.tif")
    xda_paint_false = os.path.join(tmp_path, "test_paint_false_xda.tif")
    paint_true_xda = rasters.paint(
        xda, mask.geometry, value=600, invert=True, **KAPUT_KWARGS
    )
    paint_false_xda = rasters.paint(xda, mask.geometry, value=600, invert=False)
    rasters.write(paint_true_xda, xda_paint_true, dtype=np.uint8)
    rasters.write(paint_false_xda, xda_paint_false, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, paint_true_xda)
    ci.assert_xr_encoding_attrs(xda, paint_false_xda)

    # Dataset
    xds_paint_true = os.path.join(tmp_path, "test_paint_true_xds.tif")
    xds_paint_false = os.path.join(tmp_path, "test_paint_false_xds.tif")
    paint_true_xds = rasters.paint(xds, mask, value=600, invert=True)
    paint_false_xds = rasters.paint(xds, mask, value=600, invert=False)
    rasters.write(paint_true_xds, xds_paint_true, dtype=np.uint8)
    rasters.write(paint_false_xds, xds_paint_false, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, paint_true_xds)
    ci.assert_xr_encoding_attrs(xds, paint_false_xds)

    # With dask
    paint_true_xda_dask = rasters.paint(xda_dask, mask.geometry, value=600, invert=True)
    paint_false_xda_dask = rasters.paint(
        xda_dask, mask.geometry, value=600, invert=False
    )
    assert paint_true_xda_dask.chunks is not None
    assert paint_false_xda_dask.chunks is not None

    np.testing.assert_array_equal(paint_true_xda, paint_true_xda_dask)
    np.testing.assert_array_equal(paint_false_xda, paint_false_xda_dask)
    ci.assert_xr_encoding_attrs(xda_dask, paint_true_xda_dask)
    ci.assert_xr_encoding_attrs(xda_dask, paint_false_xda_dask)


@s3_env
@dask_env
def test_crop(tmp_path, xda, xds, xda_dask, mask):
    """Test crop function"""
    # DataArray
    xda_cropped = os.path.join(tmp_path, "test_crop_xda.tif")
    crop_xda = rasters.crop(xda, mask.geometry, **KAPUT_KWARGS)
    rasters.write(crop_xda, xda_cropped, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, crop_xda)

    # Dataset
    xds_cropped = os.path.join(tmp_path, "test_crop_xds.tif")
    crop_xds = rasters.crop(xds, mask, nodata=get_nodata_value_from_xr(xds))
    rasters.write(crop_xds, xds_cropped, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, crop_xds)

    # With dask
    crop_xda_dask = rasters.crop(xda_dask, mask)
    assert crop_xda_dask.chunks is not None
    np.testing.assert_array_equal(crop_xda, crop_xda_dask)
    ci.assert_xr_encoding_attrs(xda_dask, crop_xda_dask)

    raster_cropped_xarray_path = rasters_path().joinpath("raster_cropped_xarray.tif")
    ci.assert_raster_equal(xda_cropped, raster_cropped_xarray_path)
    ci.assert_raster_equal(xds_cropped, raster_cropped_xarray_path)


@s3_env
@dask_env
def test_sieve(tmp_path, raster_path, xda, xds, xda_dask):
    """Test sieve function"""
    # DataArray
    xda_sieved = os.path.join(tmp_path, "test_sieved_xda.tif")
    sieve_xda = rasters.sieve(xda, sieve_thresh=20, connectivity=4)
    rasters.write(sieve_xda, xda_sieved, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, sieve_xda)

    # Test with different dtypes
    sieve_xda_float = rasters.sieve(
        xda.astype(np.uint8).astype(np.float32), sieve_thresh=20, connectivity=4
    )
    sieve_xda_uint = rasters.sieve(
        xda.astype(np.uint8), sieve_thresh=20, connectivity=4
    )
    np.testing.assert_array_equal(sieve_xda_uint, sieve_xda_float)

    # Dataset
    xds_sieved = os.path.join(tmp_path, "test_sieved_xds.tif")
    sieve_xds = rasters.sieve(xds, sieve_thresh=20, connectivity=4)
    rasters.write(sieve_xds, xds_sieved, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, sieve_xds)

    # With dask
    sieve_xda_dask = rasters.sieve(xda_dask, sieve_thresh=20, connectivity=4)
    # assert sieve_xda_dask.chunks is not None
    np.testing.assert_array_equal(sieve_xda, sieve_xda_dask)
    ci.assert_xr_encoding_attrs(xda_dask, sieve_xda_dask)

    # Tests
    raster_sieved_path = rasters_path().joinpath("raster_sieved.tif")
    ci.assert_raster_equal(xda_sieved, raster_sieved_path)
    ci.assert_raster_equal(xds_sieved, raster_sieved_path)

    # From path
    sieve_xda_path = rasters.sieve(raster_path, sieve_thresh=20, connectivity=4)
    np.testing.assert_array_equal(sieve_xda, sieve_xda_path)


@s3_env
@dask_env
def test_collocate_self(tmp_path, xda, xds, xda_dask):
    """Test collocate (with itself) functions"""
    # DataArray
    coll_xda = rasters.collocate(xda, xda, **KAPUT_KWARGS)
    xr.testing.assert_equal(coll_xda, xda)
    ci.assert_xr_encoding_attrs(xda, coll_xda)

    # Dataset
    coll_xds = rasters.collocate(xds, xds)
    xr.testing.assert_equal(coll_xds, xds)
    ci.assert_xr_encoding_attrs(xds, coll_xds)

    # Dataset with dataarray
    coll_xds = rasters.collocate(reference=xda, other=xds)
    xr.testing.assert_equal(coll_xds, xds)
    ci.assert_xr_encoding_attrs(xds, coll_xds)

    # With dask
    coll_xda_dask = rasters.collocate(xda_dask, xda_dask)
    assert coll_xda_dask.chunks is not None
    xr.testing.assert_equal(coll_xda_dask, xda_dask)
    ci.assert_xr_encoding_attrs(xda_dask, coll_xda_dask)


@s3_env
@dask_env
def test_collocate(tmp_path, xda, xds, xda_dask):
    """Test collocate functions"""

    def __test_collocate_output(ref, other, coll, dtype):
        """Test collocated outputs"""

        # Keeps the same attrs as the original array (attrs, encoding, dtype and name)
        ci.assert_xr_encoding_attrs(other, coll)
        ci.assert_val(coll.dtype, other.dtype, f"Collocated dtype ({dtype})")
        ci.assert_val(coll.name, other.name, f"Collocated name ({float})")

        # But located on the reference one
        assert ref.coords.identical(coll.coords)
        with pytest.raises(AssertionError):
            ci.assert_val(ref.coords, other.coords, f"Raw coordinates ({dtype})")

    # Inputs
    other_float = rasters.read(
        rasters_path().joinpath("20191115T233722_S3_SLSTR_RBT_CLOUDS_25000-00m.tif")
    )
    other_uint8 = rasters.read(
        rasters_path().joinpath(
            "20191115T233722_S3_SLSTR_RBT_CLOUDS_25000-00m_uint8.tif"
        )
    )
    ref = rasters.read(
        rasters_path().joinpath("20191115T233722_S3_SLSTR_RBT_HILLSHADE_MERIT_DEM.tif")
    )

    # Other in float
    coll_float = rasters.collocate(reference=ref, other=other_float)
    __test_collocate_output(ref, other_float, coll_float, "float")

    # Other in uint8
    coll_uint8 = rasters.collocate(reference=ref, other=other_uint8)
    __test_collocate_output(ref, other_uint8, coll_uint8, "uint8")


@s3_env
@dask_env
def test_merge_gtiff(tmp_path, raster_path):
    """Test merge_gtiff function"""
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_merged_gtiff_out = os.path.join(tmp_path, "test_merged.tif")
    rasters.merge_gtiff(
        [raster_path, raster_to_merge_path],
        raster_merged_gtiff_out,
        method="max",
    )

    raster_merged_gtiff_path = rasters_path().joinpath("raster_merged.tif")
    ci.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)


@s3_env
@dask_env
def test_vectorize(tmp_path, raster_path, ds_name, xda, xds, xda_dask):
    """Test vectorize function"""
    if shapely.__version__ >= "1.8a1":
        vect_truth_path = rasters_path().joinpath("vector.geojson")
        diss_truth_path = rasters_path().joinpath("dissolved.geojson")
    else:
        print("USING OLD VECTORS")
        vect_truth_path = rasters_path().joinpath("vector_old.geojson")
        diss_truth_path = rasters_path().joinpath("dissolved_old.geojson")

    val = 2
    vect_truth = vectors.read(vect_truth_path)

    # DataArray
    vect_xda = rasters.vectorize(xda)
    vect_val = rasters.vectorize(raster_path, values=val)
    vect_val_diss = rasters.vectorize(raster_path, values=val, dissolve=True)
    vect_val_disc = rasters.vectorize(raster_path, values=[1, 255], keep_values=False)
    ci.assert_geom_equal(vect_xda, vect_truth)
    ci.assert_geom_equal(vect_val_diss, diss_truth_path)
    ci.assert_geom_equal(vect_val, vect_truth.loc[vect_truth.raster_val == val])
    ci.assert_geom_equal(vect_val_disc, vect_truth.loc[vect_truth.raster_val == val])

    # Dataset
    vect_xds = rasters.vectorize(xds)
    ci.assert_geom_equal(vect_xds[ds_name], vect_truth)

    # With dask
    vect_xda_dask = rasters.vectorize(xda_dask)
    ci.assert_geom_equal(vect_xda_dask, vect_truth)


@s3_env
@dask_env
def test_get_valid_vec(tmp_path, xda, xds, xda_dask, ds_name):
    """Test get_valid_vector function"""
    valid_truth_path = rasters_path().joinpath("valid.geojson")

    # ----------------------------------------------------------------------------------------------
    # -- Get valid vec
    valid_truth = vectors.read(valid_truth_path)

    # DataArray
    valid_vec = rasters.get_valid_vector(xda)
    ci.assert_geom_equal(valid_vec, valid_truth)

    # Dataset
    valid_vec_xds = rasters.get_valid_vector(xds)
    ci.assert_geom_equal(valid_vec_xds[ds_name], valid_truth)

    # With dask
    valid_vec_xda_dask = rasters.get_valid_vector(xda_dask)
    ci.assert_geom_equal(valid_vec_xda_dask, valid_truth)


@s3_env
@dask_env
def test_nodata_vec(tmp_path, xda, xds, xda_dask, ds_name):
    """Test get_nodata_vector function"""
    nodata_truth_path = rasters_path().joinpath("nodata.geojson")

    # ----------------------------------------------------------------------------------------------
    # -- Get nodata vec
    nodata_truth = vectors.read(nodata_truth_path)

    # DataArray
    nodata_vec = rasters.get_nodata_vector(xda)
    ci.assert_geom_equal(nodata_vec, nodata_truth)

    # Dataset
    nodata_vec_xds = rasters.get_nodata_vector(xds)
    ci.assert_geom_equal(nodata_vec_xds[ds_name], nodata_truth)

    # With dask
    nodata_vec_dask = rasters.get_nodata_vector(xda_dask)
    ci.assert_geom_equal(nodata_vec_dask, nodata_truth)


@s3_env
@pytest.mark.skipif(
    shutil.which("gdalbuildvrt") is None,
    reason="Only works if gdalbuildvrt can be found.",
)
def test_vrt(tmp_path, raster_path):
    """Test merge_vrt function"""
    # SAME CRS
    raster_merged_vrt_path = rasters_path().joinpath("raster_merged.vrt")
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_path = rasters_path().joinpath("raster.tif")

    # Merge VRT
    raster_merged_vrt_out = os.path.join(tmp_path, "test_merged.vrt")
    rasters.merge_vrt(
        [raster_path, raster_to_merge_path], raster_merged_vrt_out, **KAPUT_KWARGS
    )
    ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)

    os.remove(raster_merged_vrt_out)

    rasters.merge_vrt(
        [raster_path, raster_to_merge_path], raster_merged_vrt_out, abs_path=True
    )
    ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)


@s3_env
@pytest.mark.skipif(
    shutil.which("gdalbuildvrt") is None,
    reason="Only works if gdalbuildvrt can be found.",
)
def test_merge_different_crs_rel(tmp_path):
    """Test merge_vrt (with different CRS) function"""
    # DIFFERENT CRS
    true_vrt_path = rasters_path().joinpath("merge_32-31.vrt")

    raster_1_path = rasters_path().joinpath(
        "20220228T102849_S2_T31TGN_L2A_134712_RED.tif"
    )
    raster_2_path = rasters_path().joinpath(
        "20220228T102849_S2_T32TLT_L2A_134712_RED.tif"
    )

    # Merge VRT
    raster_merged_vrt_out = os.path.join(tmp_path, "test_merged.vrt")
    rasters.merge_vrt([raster_1_path, raster_2_path], raster_merged_vrt_out)
    ci.assert_raster_equal(raster_merged_vrt_out, true_vrt_path)


@s3_env
@pytest.mark.skipif(
    shutil.which("gdalbuildvrt") is None,
    reason="Only works if gdalbuildvrt can be found.",
)
def test_merge_different_crs_abs(tmp_path):
    """Test merge_vrt (with different CRS) function"""
    # DIFFERENT CRS
    true_vrt_path = rasters_path().joinpath("merge_32-31.vrt")

    raster_1_path = rasters_path().joinpath(
        "20220228T102849_S2_T31TGN_L2A_134712_RED.tif"
    )
    raster_2_path = rasters_path().joinpath(
        "20220228T102849_S2_T32TLT_L2A_134712_RED.tif"
    )

    raster_merged_vrt_out = os.path.join(tmp_path, "test_merged.vrt")
    rasters.merge_vrt(
        [raster_1_path, raster_2_path], raster_merged_vrt_out, abs_path=True
    )
    ci.assert_raster_equal(raster_merged_vrt_out, true_vrt_path)


@s3_env
def test_merge_different_crs_gtiff(tmp_path):
    """Test merge_vrt (with different CRS) function"""
    # DIFFERENT CRS
    true_tif_path = rasters_path().joinpath("merge_32-31.tif")

    raster_1_path = rasters_path().joinpath(
        "20220228T102849_S2_T31TGN_L2A_134712_RED.tif"
    )
    raster_2_path = rasters_path().joinpath(
        "20220228T102849_S2_T32TLT_L2A_134712_RED.tif"
    )
    # Merge GTiff
    raster_merged_tif_out = os.path.join(tmp_path, "test_merged.tif")
    rasters.merge_gtiff([raster_1_path, raster_2_path], raster_merged_tif_out)
    ci.assert_raster_max_mismatch(
        raster_merged_tif_out, true_tif_path, max_mismatch_pct=1e-4
    )


def _test_raster_after_write(test_path, dtype, nodata_val):
    with rasterio.open(str(test_path)) as ds:
        assert ds.meta["dtype"] == dtype or ds.meta["dtype"] == dtype.__name__
        assert ds.meta["nodata"] == nodata_val
        assert ds.read()[:, 0, 0] == nodata_val  # Check value

        # Test negative value
        if "uint" not in dtype.__name__:
            assert ds.read()[:, 0, -1] == -3


@s3_env
@dask_env
@pytest.mark.parametrize(
    ("dtype", "nodata_val"),
    [
        pytest.param(np.uint8, UINT8_NODATA),
        pytest.param(np.int8, INT8_NODATA),
        pytest.param(np.uint16, UINT16_NODATA),
        pytest.param(np.int16, FLOAT_NODATA),
        pytest.param(np.uint32, UINT16_NODATA),
        pytest.param(np.int32, UINT16_NODATA),
        pytest.param(np.float32, FLOAT_NODATA),
        pytest.param(np.float64, FLOAT_NODATA),
    ],
)
def test_write(dtype, nodata_val, tmp_path, xda):
    """Test write (global check + cogs) function"""
    dtype_str = dtype.__name__

    test_path = os.path.join(tmp_path, f"test_nodata_{dtype_str}.tif")
    test_cog_path = os.path.join(tmp_path, f"test_cog_{dtype_str}.tif")
    test_cog_no_dask_path = os.path.join(tmp_path, f"test_cog_no_dask{dtype_str}.tif")

    # Force negative value if possible
    if "uint" not in dtype_str:
        xda.data[:, 0, -1] = -3

    # -------------------------------------------------------------------------------------------------
    # Test GeoTiffs
    rasters.write(xda, test_path, dtype=dtype, **KAPUT_KWARGS)
    _test_raster_after_write(test_path, dtype, nodata_val)

    # -------------------------------------------------------------------------------------------------
    # Test COGs
    if dtype not in [np.int8]:
        rasters.write(
            xda,
            test_cog_path,
            dtype=dtype,
            driver="COG",
            **KAPUT_KWARGS,
        )
        _test_raster_after_write(test_path, dtype, nodata_val)

    # -------------------------------------------------------------------------------------------------
    # COGs without dask
    if dtype not in [np.int8]:
        rasters.write(
            xda,
            test_cog_no_dask_path,
            dtype=dtype,
            driver="COG",
            write_cogs_with_dask=False,
            **KAPUT_KWARGS,
        )
        _test_raster_after_write(test_path, dtype, nodata_val)

    # test deprecation warning
    test_deprecated_path = os.path.join(tmp_path, "test_depr.tif")
    with pytest.deprecated_call():
        rasters.write(xda, path=test_deprecated_path, dtype=dtype)


def test_dim():
    """Test on BEAM-DIMAP function"""
    dim_path = rasters_path().joinpath("DIM.dim")
    dim_img_path = rasters.get_dim_img_path(dim_path)
    assert dim_img_path.is_file(), f"{dim_img_path} is not a file!"
    assert dim_img_path == rasters_path().joinpath("DIM.data", "dim.img")


@dask_env
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
@dask_env
def test_set_nodata():
    """Test xarray functions"""
    nodata_val = 0

    # Set nodata
    xda = xr.DataArray(
        dims=("x", "y"),
        data=[[1, nodata_val, nodata_val], [nodata_val, nodata_val, nodata_val]],
    )
    xda.rio.write_nodata(-9999, inplace=True, encoded=True)
    nodata = xr.DataArray(
        dims=("x", "y"), data=[[1, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )
    xda_nodata = rasters.set_nodata(xda, nodata_val)

    xr.testing.assert_equal(xda_nodata, nodata)
    ci.assert_val(xda_nodata.rio.encoded_nodata, nodata_val, "Encoded nodata")
    ci.assert_val(xda_nodata.rio.nodata, np.nan, "Array nodata")


@s3_env
@dask_env
def test_xarray_fct(xda):
    """Test xarray functions"""
    # Mtd
    xda_sum = xda + xda
    xda_sum = rasters.set_metadata(xda_sum, xda, "sum")

    ci.assert_val(xda_sum.rio.crs, xda.rio.crs, "CRS")
    assert np.isnan(xda_sum.rio.nodata)
    ci.assert_val(
        get_nodata_value_from_xr(xda_sum), get_nodata_value_from_xr(xda), "nodata"
    )
    ci.assert_val(xda_sum.attrs, xda.attrs, "attributes")
    ci.assert_val(xda_sum.encoding, xda.encoding, "encoding")
    ci.assert_val(xda_sum.rio.transform(), xda.rio.transform(), "transform")
    ci.assert_val(xda_sum.rio.width, xda.rio.width, "width")
    ci.assert_val(xda_sum.rio.height, xda.rio.height, "height")
    ci.assert_val(xda_sum.rio.count, xda.rio.count, "count")
    ci.assert_val(xda_sum.name, "sum", "name")


@dask_env
def test_where():
    """Test overloading of xr.where function"""
    A = xr.DataArray(dims=("x", "y"), data=[[1, 0, 5], [np.nan, 0, 0]])
    mask_A = rasters.where(A > 3, 0, 1, A, new_name="mask_A")

    np.testing.assert_equal(np.isnan(A.data), np.isnan(mask_A.data))
    assert A.attrs == mask_A.attrs
    np.testing.assert_equal(
        mask_A.data, np.array([[1.0, 1.0, 0.0], [np.nan, 1.0, 1.0]])
    )


@s3_env
@dask_env
def test_aspect(tmp_path, dem_path):
    """Test aspect function"""
    aspect_path = rasters_path().joinpath("aspect.tif")

    # Path OUT
    aspect_path_out = os.path.join(tmp_path, "aspect_out.tif")

    # Aspect
    aspect = rasters.aspect(dem_path)
    # assert aspect.chunks is not None
    rasters.write(aspect, aspect_path_out, dtype="float32")
    ci.assert_raster_almost_equal(aspect_path, aspect_path_out, decimal=4)


@s3_env
@dask_env
def test_hillshade(tmp_path, dem_path):
    """Test hillshade function"""
    hlsd_path = rasters_path().joinpath("hillshade.tif")
    hlsd_path_out = os.path.join(tmp_path, "hillshade_out.tif")

    # Hillshade
    hlsd = rasters.hillshade(dem_path, 34.0, 45.2)
    # assert hlsd.chunks is not None
    rasters.write(hlsd, hlsd_path_out, dtype="float32")
    ci.assert_raster_almost_equal(hlsd_path, hlsd_path_out, decimal=4)


@s3_env
@dask_env
def test_slope(tmp_path, dem_path):
    """Test slope function"""
    slope_path = rasters_path().joinpath("slope.tif")
    slope_path_out = os.path.join(tmp_path, "slope_out.tif")
    # Slope
    slp = rasters.slope(dem_path)
    # assert slp.chunks is not None
    rasters.write(slp, slope_path_out, dtype="float32")
    ci.assert_raster_almost_equal(slope_path, slope_path_out, decimal=4)


@s3_env
@dask_env
def test_slope_rad(tmp_path, dem_path):
    """Test slope (radian) function"""
    slope_r_path = rasters_path().joinpath("slope_r.tif")
    slope_r_path_out = os.path.join(tmp_path, "slope_r_out.tif")

    # Slope rad
    slp_r = rasters.slope(dem_path, in_pct=False, in_rad=True)
    # assert slp_r.chunks is not None
    rasters.write(slp_r, slope_r_path_out, dtype="float32")
    ci.assert_raster_almost_equal(slope_r_path, slope_r_path_out, decimal=4)


@s3_env
@dask_env
def test_slope_pct(tmp_path, dem_path):
    """Test slope (pct) function"""
    slope_p_path = rasters_path().joinpath("slope_p.tif")
    slope_p_path_out = os.path.join(tmp_path, "slope_p_out.tif")

    # Slope pct
    slp_p = rasters.slope(dem_path, in_pct=True)
    # assert slp_p.chunks is not None
    rasters.write(slp_p, slope_p_path_out, dtype="float32")
    ci.assert_raster_almost_equal(slope_p_path, slope_p_path_out, decimal=4)


@s3_env
@dask_env
def test_rasterize(tmp_path, raster_path):
    """Test rasterize fct"""
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_float_path = rasters_path().joinpath("raster_float.tif")
    raster_true_bin_path = rasters_path().joinpath("rasterized_bin.tif")
    raster_true_path = rasters_path().joinpath("rasterized.tif")

    chunks = {"band": 1, "x": 2048, "y": 2084}

    # Binary vector
    out_bin_path = os.path.join(tmp_path, "out_bin.tif")
    rast_bin = rasters.rasterize(
        rasters.read(raster_path, chunks=chunks), vec_path, **KAPUT_KWARGS
    )
    rasters.write(rast_bin, out_bin_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_path, decimal=4)

    # Binary vector with floating point raster
    out_bin_path = os.path.join(tmp_path, "out_bin_float.tif")
    rast_bin = rasters.rasterize(
        rasters.read(raster_float_path, chunks=chunks), vec_path
    )
    rasters.write(rast_bin, out_bin_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_path, decimal=4)

    # Vector
    out_path = os.path.join(tmp_path, "out.tif")
    rast = rasters.rasterize(
        raster_path, vec_path, value_field="raster_val", dtype=np.uint8
    )
    rasters.write(rast, out_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_path, out_path, decimal=4)


@s3_env
def test_decorator_deprecation(raster_path):
    from sertit.rasters import path_xarr_dst

    @any_raster_to_xr_ds
    def _ok_rasters(xds):
        assert isinstance(xds, xr.DataArray)
        return xds

    @path_xarr_dst
    def _depr_rasters(xds):
        assert isinstance(xds, xr.DataArray)
        return xds

    # Not able to warn deprecation from inside the decorator
    xr.testing.assert_equal(_ok_rasters(raster_path), _depr_rasters(raster_path))


def test_get_nodata_deprecation():
    """Test deprecation of get_nodata_value"""
    # Test deprecation
    for dtype in [
        np.uint8,
        np.int8,
        np.uint16,
        np.uint32,
        np.int32,
        np.int64,
        np.uint64,
        int,
        "int",
        np.int16,
        np.float32,
        np.float64,
        float,
        "float",
    ]:
        with pytest.deprecated_call():
            from sertit.rasters import get_nodata_value

            ci.assert_val(
                get_nodata_value_from_dtype(dtype), get_nodata_value(dtype), dtype
            )


@s3_env
@dask_env
def test_get_notata_from_xr(raster_path):
    """Test get_nodata_value_from_xr"""
    ci.assert_val(get_nodata_value_from_xr(rasters.read(raster_path)), 255, "nodata")

    raster_path = rasters_path().joinpath(
        "20220228T102849_S2_T31TGN_L2A_134712_RED.tif"
    )
    ci.assert_val(get_nodata_value_from_xr(rasters.read(raster_path)), 65535, "nodata")

    raster_path = rasters_path().joinpath(
        "Copernicus_DSM_10_N43_00_W003_00_DEM_resampled.tif"
    )
    ci.assert_val(get_nodata_value_from_xr(rasters.read(raster_path)), None, "nodata")

    raster_path = rasters_path().joinpath("dem.tif")
    ci.assert_val(get_nodata_value_from_xr(rasters.read(raster_path)), -9999, "nodata")
