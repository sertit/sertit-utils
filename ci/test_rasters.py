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

from ci.script_utils import (
    CI_SERTIT_USE_DASK,
    KAPUT_KWARGS,
    assert_chunked_computed,
    dask_env,
    get_output,
    is_not_lazy_yet,
    rasters_path,
    s3_env,
)
from sertit import ci, geometry, path, rasters, unistra, vectors
from sertit.rasters import (
    DEG_LAT_TO_M,
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

DEBUG = False


def test_indexes(caplog):
    @s3_env
    @dask_env(nof_computes=2)  # assert_equal x2
    def test_core():
        l1_path = rasters_path().joinpath("19760712T093233_L1_215030_MSS_stack.tif")
        xda_raw = rasters.read(l1_path, **KAPUT_KWARGS)
        xda_idx = rasters.read(l1_path, indexes=1)

        xr.testing.assert_equal(xda_raw[[0], :], xda_idx)  # Compute #1

        xda_idx2 = rasters.read(l1_path, indexes=[3, 2])
        xda_raw2 = np.concatenate((xda_raw.data[[2], :], xda_raw.data[[1], :]))

        with pytest.raises(ValueError):
            rasters.read(l1_path, indexes=0)

        with caplog.at_level(logging.WARNING):
            idx = [4, 5]
            rasters.read(l1_path, indexes=idx)
            assert f"Non available index: {idx}" in caplog.text

        # Test laziness
        assert_chunked_computed(xda_raw, "Read without index")
        assert_chunked_computed(xda_idx, "Read with one index")
        assert_chunked_computed(xda_idx2, "Read with indices")
        assert_chunked_computed(xda_raw2, "Concatenation")

        # Compute after checking the array is chunked (otherwise it fails!)
        import dask

        xda_raw2, xda_idx2 = dask.compute(xda_raw2, xda_idx2)  # Compute #2
        np.testing.assert_equal(xda_raw2, xda_idx2)

    test_core()


@pytest.fixture
def raster_path():
    return rasters_path().joinpath("raster.tif")


@pytest.fixture
def dem_path():
    return rasters_path().joinpath("dem.tif")


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


def get_xda(raster_path, **kwargs):
    return rasters.read(raster_path, **kwargs)


def get_xds(raster_path):
    with rasterio.open(str(raster_path)) as ds:
        ds_name = path.get_filename(ds.name)
    return xr.Dataset({ds_name: get_xda(raster_path)})


@s3_env
@dask_env()
def test_rasters_extent(tmp_path, raster_path, ds_name, ds_dtype):
    """"""
    extent_path = rasters_path().joinpath("extent.geojson")
    # Get Extent
    extent = rasters.get_extent(raster_path)
    truth_extent = vectors.read(extent_path)
    ci.assert_geom_equal(extent, truth_extent)


@s3_env
@dask_env(nof_computes=1)  # Vectorize is not lazy
def test_rasters_footprint(tmp_path, raster_path, ds_name, ds_dtype):
    # Get Footprint
    footprint_path = rasters_path().joinpath("footprint.geojson")
    footprint = rasters.get_footprint(raster_path)  # Compute #1
    truth_footprint = vectors.read(footprint_path)
    ci.assert_geom_equal(footprint, truth_footprint)


@s3_env
@dask_env(
    nof_computes=1
)  # TODO: Read with downsampling issue https://github.com/opendatacube/odc-geo/issues/236
def test_read(tmp_path, raster_path, ds_name, ds_dtype):
    """Test read function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    with rasterio.open(str(raster_path)) as ds:
        # -- Read and test laziness
        assert_chunked_computed(xda, "Native xda")

        # Native resolution
        xda_1 = rasters.read(ds, resolution=ds.res[0])
        assert_chunked_computed(xda_1, "Read with native resolution (x)")

        # Native resolution
        xda_2 = rasters.read(raster_path, resolution=[ds.res[0], ds.res[1]])
        assert_chunked_computed(xda_2, "Read with native resolution (x,y)")

        # Native size
        xda_3 = rasters.read(raster_path, size=(xda_1.rio.width, xda_1.rio.height))
        assert_chunked_computed(xda_3, "Read with native size")

        # Upsampling (reproject)
        xda_4 = rasters.read(raster_path, resolution=ds.res[0] / 2)
        assert_chunked_computed(xda_4, "Read with upsampling")

        # Read a xarray
        xda_5 = rasters.read(xda)
        assert_chunked_computed(xda_5, "Read an already existing xr.DataArray")

        # Downsampling (coarsen)
        xda_6 = rasters.read(raster_path, resolution=ds.res[0] * 2)
        assert_chunked_computed(xda_6, "Read with downsampling")

        # Downsampling (reproject)
        xda_7 = rasters.read(raster_path, resolution=ds.res[0] * 10.1)  # Compute #1
        # TODO: remove when https://github.com/opendatacube/odc-geo/issues/236 is fixed
        # assert_chunked_computed(xda_7, "Read with downsampling (reproject)")

        # Test shape (link between resolution and size)
        assert xda_4.shape[-2] == xda.shape[-2] * 2
        assert xda_4.shape[-1] == xda.shape[-1] * 2
        assert xda_6.shape[-2] == xda.shape[-2] / 2
        assert xda_6.shape[-1] == xda.shape[-1] / 2
        assert xda_7.shape[-1] == round(xda.shape[-1] / 10.1)
        assert xda_7.shape[-2] == round(xda.shape[-2] / 10.1)
        with pytest.raises(ValueError):
            rasters.read(ds, resolution=[20, 20, 20])

        # Test dataset integrity
        assert xda.shape == (ds.count, ds.height, ds.width)
        assert xda.encoding["dtype"] == ds_dtype
        assert xds[ds_name].shape == xda.shape
        assert xda_1.rio.crs == ds.crs
        assert xda_1.rio.transform() == ds.transform
        xr.testing.assert_equal(xda_1, xda_2)
        xr.testing.assert_equal(xda_1, xda_3)
        xr.testing.assert_equal(xda, xda_5)

        ci.assert_xr_encoding_attrs(xda, xda_1)
        ci.assert_xr_encoding_attrs(xda, xda_2)
        ci.assert_xr_encoding_attrs(xda, xda_3)
        ci.assert_xr_encoding_attrs(xda, xda_4)
        ci.assert_xr_encoding_attrs(xda, xda_5)
        ci.assert_xr_encoding_attrs(xda, xda_6)
        ci.assert_xr_encoding_attrs(xda, xda_7)
        ci.assert_val(xda_1.attrs["path"], str(raster_path), "raster path")


@s3_env
def test_read_np_vs_dask(raster_path):
    xda = get_xda(raster_path, chunks=None)

    # Should be always lazy
    xda_dask = rasters.read(raster_path, chunks=True)
    ci.assert_chunked(xda_dask), "Read with chunks = True"

    # Test arrays are the same
    xr.testing.assert_equal(xda, xda_dask)
    ci.assert_xr_encoding_attrs(xda, xda_dask, unchecked_attr="preferred_chunks")


@s3_env
@dask_env(nof_computes=2)  # Write x2
def test_read_with_window(tmp_path, raster_path, mask_path, mask):
    """Test read (with window) function"""
    raster_window_path = rasters_path().joinpath("window.tif")
    raster_window_20_path = rasters_path().joinpath("window_20.tif")

    # Window with native resolution
    xda_window_out = get_output(tmp_path, "test_xda_window.tif", DEBUG)
    xda_window = rasters.read(
        raster_path,
        window=mask_path,
    )
    assert_chunked_computed(xda_window, "Read with a window with native resolution")
    rasters.write(xda_window, xda_window_out, dtype=np.uint8)  # Compute #1
    ci.assert_raster_equal(xda_window_out, raster_window_path)

    # Window with updated resolution
    xda_window_20_out = get_output(tmp_path, "test_xda_20_window.tif", DEBUG)
    gdf = mask.to_crs(EPSG_4326)
    xda_window_20 = rasters.read(raster_path, window=gdf, resolution=20)
    ci.assert_val(round(xda_window_20.rio.resolution()[0]), 20, "resolution")

    # TODO: https://github.com/opendatacube/odc-geo/issues/236
    # assert_chunked_computed(xda_window_20, "Read with a window with updated resolution")

    rasters.write(xda_window_20, xda_window_20_out, dtype=np.uint8)  # Compute #2
    ci.assert_raster_equal(xda_window_20_out, raster_window_20_path)

    # Non existing window
    with pytest.raises(FileNotFoundError):
        rasters.read(
            raster_path,
            window=rasters_path().joinpath("non_existing_window.kml"),
        )


@s3_env
@dask_env()
@pytest.mark.timeout(4, func_only=True)
def test_very_big_file(tmp_path, mask_path):
    """Test read with big files function (should be fast, if not there is something going on...)"""
    dem_path = unistra.get_geodatastore() / "GLOBAL" / "EUDEM_v2" / "eudem_wgs84.tif"
    aoi_path = rasters_path() / "DAX.shp"
    with rasterio.open(dem_path) as ds:
        dem_res = ds.res[0]  # noqa
        dem_crs = ds.crs  # noqa

    aoi = vectors.read(aoi_path)
    cropped = rasters.crop(rasters.read(dem_path, window=aoi), aoi, nodata=0)
    assert_chunked_computed(cropped, "Crop (big file)")


@s3_env
@dask_env(max_computes=3)  # Write x2 or 3
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
def test_write(dtype, nodata_val, tmp_path, raster_path):
    """Test write (global check + cogs) function"""
    xda = get_xda(raster_path)
    dtype_str = dtype.__name__

    test_path = get_output(tmp_path, f"test_nodata_{dtype_str}.tif", DEBUG)
    test_cog_path = get_output(tmp_path, f"test_cog_{dtype_str}.tif", DEBUG)
    test_cog_no_dask_path = get_output(
        tmp_path, f"test_cog_no_dask{dtype_str}.tif", DEBUG
    )

    # Force negative value if possible
    if "uint" not in dtype_str:
        xda.data[:, 0, -1] = -3

    # -------------------------------------------------------------------------------------------------
    # Test GeoTiffs
    rasters.write(xda, test_path, dtype=dtype, **KAPUT_KWARGS)  # Compute #1
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
        )  # Compute #2
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
        )  # Compute #3
        _test_raster_after_write(test_path, dtype, nodata_val)

    # test deprecation warning
    test_deprecated_path = get_output(tmp_path, "test_depr.tif", DEBUG)
    with pytest.deprecated_call():
        rasters.write(
            xda, path=test_deprecated_path, dtype=dtype
        )  # This doesn't compute (because of context manager ?)


@s3_env
# @dask_env(nof_compute=1)  # Write x1: TODO: Right now, Zarr doesn't seem to work with dask and GDAL 3.6.
def test_write_zarr(tmp_path, raster_path):
    # test zarr
    xda = get_xda(raster_path, chunks=None)
    zarr_path = get_output(tmp_path, "z.zarr", DEBUG)
    rasters.write(xda, path=zarr_path, driver="Zarr")
    xr.testing.assert_equal(xda, rasters.read(zarr_path, driver="Zarr"))


@s3_env
@dask_env(nof_computes=2)  # Write x2
def test_write_basic(tmp_path, raster_path, ds_dtype):
    """Test write (basic) function"""

    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # DataArray
    xda_out = get_output(tmp_path, "test_xda.tif", DEBUG)
    rasters.write(xda, xda_out, dtype=ds_dtype)
    assert os.path.isfile(xda_out)

    # Dataset
    xds_out = get_output(tmp_path, "test_xds.tif", DEBUG)
    rasters.write(xds, xds_out, dtype=ds_dtype)
    assert os.path.isfile(xds_out)

    # Tests
    ci.assert_raster_equal(raster_path, xda_out)
    ci.assert_raster_equal(raster_path, xds_out)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_write_png(tmp_path, raster_path):
    """Test write (PNG) function"""

    # Just test if this doesn't fail
    xda_out = get_output(tmp_path, "test_xda.png", DEBUG)
    rasters.write(get_xda(raster_path), xda_out, dtype="uint8", driver="PNG")


@s3_env
@dask_env(nof_computes=1)  # Write x2 with dask, but only one compute!
def test_write_dask(tmp_path, raster_path, ds_dtype):
    """Test write (basic) function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # DataArray
    xda_out = get_output(tmp_path, "test_xda.tif", DEBUG)
    delayed_1 = rasters.write(xda, xda_out, dtype=ds_dtype, compute=False)

    # Dataset
    xds_out = get_output(tmp_path, "test_xds.tif", DEBUG)
    delayed_2 = rasters.write(xds, xds_out, dtype=ds_dtype, compute=False)

    # Compute
    import dask

    dask.compute(delayed_1, delayed_2)
    assert os.path.isfile(xda_out)
    assert os.path.isfile(xds_out)

    # Tests
    ci.assert_raster_equal(raster_path, xda_out)
    ci.assert_raster_equal(raster_path, xds_out)


@s3_env
@dask_env(nof_computes=2)  # Write x2
def test_mask(tmp_path, raster_path, mask):
    """Test mask function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # DataArray
    xda_masked = get_output(tmp_path, "test_mask_xda.tif", DEBUG)
    mask_xda = rasters.mask(xda, mask.geometry, **KAPUT_KWARGS)
    assert_chunked_computed(mask_xda, "Mask DataArray")
    rasters.write(mask_xda, xda_masked, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, mask_xda)

    # Dataset
    xds_masked = get_output(tmp_path, "test_mask_xds.tif", DEBUG)
    mask_xds = rasters.mask(xds, mask)
    assert_chunked_computed(mask_xds, "Mask Dataset")
    rasters.write(mask_xds, xds_masked, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, mask_xds)

    raster_masked_path = rasters_path().joinpath("raster_masked.tif")
    ci.assert_raster_equal(xda_masked, raster_masked_path)
    ci.assert_raster_equal(xds_masked, raster_masked_path)


@s3_env
@dask_env(nof_computes=4)  # Write x4
def test_paint(tmp_path, raster_path, mask):
    """Test paint function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # -- DataArray --
    # Invert = True
    xda_paint_true = get_output(tmp_path, "test_paint_true_xda.tif", DEBUG)
    paint_true_xda = rasters.paint(
        xda, mask.geometry, value=600, invert=True, **KAPUT_KWARGS
    )
    assert_chunked_computed(paint_true_xda, "Paint DataArray (invert = true)")
    rasters.write(paint_true_xda, xda_paint_true, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, paint_true_xda)

    # Invert = False
    xda_paint_false = get_output(tmp_path, "test_paint_false_xda.tif", DEBUG)
    paint_false_xda = rasters.paint(xda, mask.geometry, value=600, invert=False)
    assert_chunked_computed(paint_false_xda, "Paint DataArray (invert = false)")
    rasters.write(paint_false_xda, xda_paint_false, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, paint_false_xda)

    # -- Dataset --
    # Invert = True
    xds_paint_true = get_output(tmp_path, "test_paint_true_xds.tif", DEBUG)
    paint_true_xds = rasters.paint(xds, mask, value=600, invert=True)
    assert_chunked_computed(paint_true_xds, "Paint Dataset (invert = true)")
    rasters.write(paint_true_xds, xds_paint_true, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, paint_true_xds)

    # Invert = False
    xds_paint_false = get_output(tmp_path, "test_paint_false_xds.tif", DEBUG)
    paint_false_xds = rasters.paint(xds, mask, value=600, invert=False)
    assert_chunked_computed(paint_false_xds, "Paint Dataset (invert = false)")
    rasters.write(paint_false_xds, xds_paint_false, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, paint_false_xds)


@s3_env
@dask_env(nof_computes=2)  # Write x2
def test_crop(tmp_path, raster_path, mask):
    """Test crop function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # DataArray
    xda_cropped = get_output(tmp_path, "test_crop_xda.tif", DEBUG)
    crop_xda = rasters.crop(xda, mask.geometry, **KAPUT_KWARGS)
    assert_chunked_computed(crop_xda, "Crop DataArray")
    rasters.write(crop_xda, xda_cropped, dtype=np.uint8)  # Compute #1
    ci.assert_xr_encoding_attrs(xda, crop_xda)

    # Dataset
    xds_cropped = get_output(tmp_path, "test_crop_xds.tif", DEBUG)
    crop_xds = rasters.crop(xds, mask, nodata=get_nodata_value_from_xr(xds))
    assert_chunked_computed(crop_xds, "Crop Dataset")
    rasters.write(crop_xds, xds_cropped, dtype=np.uint8)  # Compute #2
    ci.assert_xr_encoding_attrs(xds, crop_xds)

    raster_cropped_xarray_path = rasters_path().joinpath("raster_cropped_xarray.tif")
    ci.assert_raster_equal(xda_cropped, raster_cropped_xarray_path)
    ci.assert_raster_equal(xds_cropped, raster_cropped_xarray_path)

    # Test with mask with Z
    mask_z = geometry.force_3d(mask)
    crop_z = rasters.crop(xda, mask_z)
    assert_chunked_computed(crop_z, "Crop 3D")
    xr.testing.assert_equal(crop_xda, crop_z)
    ci.assert_xr_encoding_attrs(crop_xda, crop_z)


@s3_env
@is_not_lazy_yet
@dask_env(
    nof_computes=2
)  # Sieve x2 (write a not-chunked array, so no additional 'compute')
def test_sieve(tmp_path, raster_path):
    """Test sieve function"""
    # TODO: not lazy

    # xda
    xda = get_xda(raster_path)

    # Sieve DataArray
    xda_sieved = get_output(tmp_path, "test_sieved_xda.tif", DEBUG)
    sieve_xda = rasters.sieve(xda, sieve_thresh=20, connectivity=4)
    assert_chunked_computed(sieve_xda, "Sieve DataArray")
    rasters.write(sieve_xda, xda_sieved, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xda, sieve_xda)

    # Test equality
    raster_sieved_path = rasters_path().joinpath("raster_sieved.tif")
    ci.assert_raster_equal(xda_sieved, raster_sieved_path)

    # Sieve From path
    sieve_xda_path = rasters.sieve(raster_path, sieve_thresh=20, connectivity=4)
    assert_chunked_computed(sieve_xda_path, "Sieve DataArray (directly from path)")
    xr.testing.assert_equal(sieve_xda, sieve_xda_path)


@s3_env
@is_not_lazy_yet
@dask_env(nof_computes=2)  # Sieve x2
def test_sieve_different_dtypes(tmp_path, raster_path):
    # Test with different dtypes

    # xda
    xda = get_xda(raster_path)

    sieve_xda_float = rasters.sieve(
        xda.astype(np.uint8).astype(np.float32), sieve_thresh=20, connectivity=4
    )
    assert_chunked_computed(sieve_xda_float, "Sieve DataArray (float)")
    sieve_xda_uint = rasters.sieve(
        xda.astype(np.uint8), sieve_thresh=20, connectivity=4
    )
    assert_chunked_computed(sieve_xda_uint, "Sieve DataArray (uint)")
    xr.testing.assert_equal(sieve_xda_uint, sieve_xda_float)


@s3_env
@is_not_lazy_yet
@dask_env(
    nof_computes=1
)  # Sieve x1 (write a not-chunked array, so no additional 'compute')
def test_sieve_dataset(tmp_path, raster_path):
    # xds
    xds = get_xds(raster_path)

    # Dataset
    xds_sieved = get_output(tmp_path, "test_sieved_xds.tif", DEBUG)
    sieve_xds = rasters.sieve(xds, sieve_thresh=20, connectivity=4)
    assert_chunked_computed(sieve_xds, "Sieve Dataset")
    rasters.write(sieve_xds, xds_sieved, dtype=np.uint8)
    ci.assert_xr_encoding_attrs(xds, sieve_xds)

    # Tests
    raster_sieved_path = rasters_path().joinpath("raster_sieved.tif")
    ci.assert_raster_equal(xds_sieved, raster_sieved_path)


@s3_env
@dask_env()
def test_collocate_self(tmp_path, raster_path):
    """Test collocate (with itself) functions"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # DataArray
    coll_xda = rasters.collocate(xda, xda, **KAPUT_KWARGS)
    assert_chunked_computed(coll_xda, "Collocate DataArray (self)")
    xr.testing.assert_equal(coll_xda, xda)
    ci.assert_xr_encoding_attrs(xda, coll_xda)

    # Dataset
    coll_xds = rasters.collocate(xds, xds)
    assert_chunked_computed(coll_xds, "Collocate Dataset (self)")
    xr.testing.assert_equal(coll_xds, xds)
    ci.assert_xr_encoding_attrs(xds, coll_xds)

    # Dataset with dataarray
    coll_xds = rasters.collocate(reference=xda, other=xds)
    assert_chunked_computed(coll_xds, "Collocate Dataset with DataArray")
    xr.testing.assert_equal(coll_xds, xds)
    ci.assert_xr_encoding_attrs(xds, coll_xds)


@s3_env
@dask_env()
def test_collocate(tmp_path):
    """Test collocate functions"""

    def __test_collocate_output(ref, other, coll, dtype):
        """Test collocated outputs"""
        assert_chunked_computed(coll, f"Collocate DataArray ({dtype})")

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
        rasters_path().joinpath("20191115T233722_S3_SLSTR_RBT_CLOUDS_25000-00m.tif"),
        masked=True,
        as_type=np.float32,
    )
    other_uint8 = rasters.read(
        rasters_path().joinpath(
            "20191115T233722_S3_SLSTR_RBT_CLOUDS_25000-00m_uint8.tif"
        ),
        masked=False,
        as_type=np.uint8,
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
def test_merge_gtiff(tmp_path, raster_path):
    """Test merge_gtiff function"""
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_merged_gtiff_out = get_output(tmp_path, "test_merged.tif", DEBUG)
    rasters.merge_gtiff(
        [raster_path, raster_to_merge_path],
        raster_merged_gtiff_out,
        method="max",
        driver="GTiff",
        dtype="uint8",
    )

    raster_merged_gtiff_path = rasters_path().joinpath("raster_merged.tif")
    ci.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)


@s3_env
@dask_env(nof_computes=5)  # vectorize x5 (vectorize not lazy yet)
def test_vectorize(tmp_path, raster_path, ds_name):
    """Test vectorize function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

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


@s3_env
@dask_env(nof_computes=2)  # vectorize x2 (vectorize not lazy yet)
def test_get_valid_vec(tmp_path, raster_path, ds_name):
    """Test get_valid_vector function"""
    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    valid_truth_path = rasters_path().joinpath("valid.geojson")

    # -- Get valid vec
    valid_truth = vectors.read(valid_truth_path)

    # DataArray
    valid_vec = rasters.get_valid_vector(xda)
    ci.assert_geom_equal(valid_vec, valid_truth)

    # Dataset
    valid_vec_xds = rasters.get_valid_vector(xds)
    ci.assert_geom_equal(valid_vec_xds[ds_name], valid_truth)


@s3_env
@dask_env(nof_computes=2)  # vectorize x2 (vectorize not lazy yet)
def test_nodata_vec(tmp_path, raster_path, ds_name):
    """Test get_nodata_vector function"""
    nodata_truth_path = rasters_path().joinpath("nodata.geojson")

    # xda, xds
    xda = get_xda(raster_path)
    xds = get_xds(raster_path)

    # -- Get nodata vec
    nodata_truth = vectors.read(nodata_truth_path)

    # DataArray
    nodata_vec = rasters.get_nodata_vector(xda)
    ci.assert_geom_equal(nodata_vec, nodata_truth)

    # Dataset
    nodata_vec_xds = rasters.get_nodata_vector(xds)
    ci.assert_geom_equal(nodata_vec_xds[ds_name], nodata_truth)


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
    raster_merged_vrt_out = get_output(tmp_path, "test_merged.vrt", DEBUG)
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
    raster_merged_vrt_out = get_output(tmp_path, "test_merged.vrt", DEBUG)
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

    raster_merged_vrt_out = get_output(tmp_path, "test_merged.vrt", DEBUG)
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
    raster_merged_tif_out = get_output(tmp_path, "test_merged.tif", DEBUG)
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


def test_dim():
    """Test on BEAM-DIMAP function"""
    dim_path = rasters_path().joinpath("DIM.dim")
    dim_img_path = rasters.get_dim_img_path(dim_path)
    assert dim_img_path.is_file(), f"{dim_img_path} is not a file!"
    assert dim_img_path == rasters_path().joinpath("DIM.data", "dim.img")


@s3_env
@dask_env(nof_computes=1)  # compute x1
@pytest.mark.parametrize(
    ("dtype", "bit_id"),
    [
        pytest.param(np.uint16, 15),
        pytest.param(np.uint8, 7),
        pytest.param(np.uint32, 31),
    ],
)
def test_read_bit_array(dtype, bit_id):
    """Test bit arrays"""
    data = np.ones((1, 2, 2), dtype=dtype)

    if os.environ[CI_SERTIT_USE_DASK] == "1":
        # If we want to use dask, convert this array to a dask array
        from dask.array import from_array

        data = from_array(data)
    # Bit
    np_ones = xr.DataArray(data)
    ones = rasters.read_bit_array(np_ones, bit_id=0)
    zeros = rasters.read_bit_array(np_ones, bit_id=list(np.arange(1, bit_id)))

    # Outputs an array (either dask or numpy)
    if os.environ[CI_SERTIT_USE_DASK] == "1":
        import dask

        delayed = []
        delayed.append(dask.delayed(np.testing.assert_array_equal)(np_ones.data, ones))
        for arr in zeros:
            delayed.append(
                dask.delayed(np.testing.assert_array_equal)(np_ones.data, 1 + arr)
            )

        dask.compute(delayed)
    else:
        np.testing.assert_array_equal(np_ones.data, ones)
        for arr in zeros:
            np.testing.assert_array_equal(np_ones.data, 1 + arr)


@s3_env
@dask_env(nof_computes=1)  # compute x1
@pytest.mark.parametrize(
    ("dtype", "bit_id"),
    [
        pytest.param(np.uint8, 7),
        pytest.param(float, 7),
    ],
)
def test_read_uint8_array(dtype, bit_id):
    # uint8
    data = np.ones((1, 2, 2), dtype=dtype)

    if os.environ[CI_SERTIT_USE_DASK] == "1":
        # If we want to use dask, convert this array to a dask array
        from dask.array import from_array

        data = from_array(data)

    np_ones = xr.DataArray(data)
    ones = rasters.read_uint8_array(np_ones, bit_id=0)
    zeros = rasters.read_uint8_array(np_ones, bit_id=list(np.arange(1, bit_id)))

    # Outputs an array (either dask or numpy)
    if os.environ[CI_SERTIT_USE_DASK] == "1":
        import dask

        delayed = []
        delayed.append(dask.delayed(np.testing.assert_array_equal)(np_ones.data, ones))
        for arr in zeros:
            delayed.append(
                dask.delayed(np.testing.assert_array_equal)(np_ones.data, 1 + arr)
            )

        dask.compute(delayed)
    else:
        np.testing.assert_array_equal(np_ones.data, ones)
        for arr in zeros:
            np.testing.assert_array_equal(np_ones.data, 1 + arr)


@s3_env
@dask_env(nof_computes=1)  # assert_equal x1
def test_set_nodata():
    """Test xarray functions"""
    nodata_val = 0
    data = [[1, nodata_val, nodata_val], [nodata_val, nodata_val, nodata_val]]
    if os.environ[CI_SERTIT_USE_DASK] == "1":
        # If we want to use dask, convert this array to a dask array
        from dask.array import from_array

        data = from_array(data)

    # Set nodata
    xda = xr.DataArray(
        dims=("x", "y"),
        data=data,
    )
    xda.rio.write_nodata(-9999, inplace=True, encoded=True)
    assert_chunked_computed(xda, "Write nodata (rioxarray)")

    xda_nodata = rasters.set_nodata(xda, nodata_val)
    assert_chunked_computed(xda_nodata, "Set nodata")

    nodata = xr.DataArray(
        dims=("x", "y"), data=[[1, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    )
    xr.testing.assert_equal(xda_nodata, nodata)
    ci.assert_val(xda_nodata.rio.encoded_nodata, nodata_val, "Encoded nodata")
    ci.assert_val(xda_nodata.rio.nodata, np.nan, "Array nodata")


@s3_env
@dask_env()
def test_xarray_fct(raster_path):
    """Test xarray functions"""
    xda = get_xda(raster_path)
    xda_sum = xda + xda
    assert_chunked_computed(xda_sum, "Sum between two DataArrays")

    # Mtd
    xda_sum = rasters.set_metadata(xda_sum, xda, "sum")
    assert_chunked_computed(xda_sum, "Set metadata")

    ci.assert_val(xda_sum.rio.crs, xda.rio.crs, "CRS")
    assert np.isnan(xda_sum.rio.nodata)
    ci.assert_val(
        get_nodata_value_from_xr(xda_sum), get_nodata_value_from_xr(xda), "nodata"
    )

    ci.assert_val(xda_sum.attrs.pop("long_name"), "sum", "long name")
    ci.assert_val(xda_sum.attrs, xda.attrs, "attributes")
    ci.assert_val(xda_sum.encoding, xda.encoding, "encoding")
    ci.assert_val(xda_sum.rio.transform(), xda.rio.transform(), "transform")
    ci.assert_val(xda_sum.rio.width, xda.rio.width, "width")
    ci.assert_val(xda_sum.rio.height, xda.rio.height, "height")
    ci.assert_val(xda_sum.rio.count, xda.rio.count, "count")
    ci.assert_val(xda_sum.name, "sum", "name")


@dask_env(nof_computes=2)  # assert_equal x2
def test_where():
    """Test overloading of xr.where function"""
    new_name = "mask_A"

    data = [[1, 0, 5], [np.nan, 0, 0]]
    if os.environ[CI_SERTIT_USE_DASK] == "1":
        # If we want to use dask, convert this array to a dask array
        from dask.array import from_array

        data = from_array(data)

    xarr = xr.DataArray(dims=("x", "y"), data=data)
    mask_xarr = rasters.where(xarr > 3, 0, 1, xarr, new_name=new_name)
    assert_chunked_computed(mask_xarr, "Where")

    xr.testing.assert_equal(np.isnan(xarr), np.isnan(mask_xarr))

    assert mask_xarr.attrs.pop("long_name") == new_name
    assert xarr.attrs == mask_xarr.attrs
    xr.testing.assert_equal(
        mask_xarr,
        xr.DataArray(
            dims=("x", "y"), data=np.array([[1.0, 1.0, 0.0], [np.nan, 1.0, 1.0]])
        ),
    )


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_aspect(tmp_path, dem_path):
    """Test aspect function"""
    aspect_path = rasters_path().joinpath("aspect.tif")

    # Path OUT
    aspect_path_out = get_output(tmp_path, "aspect_out.tif", DEBUG)

    # Aspect
    aspect = rasters.aspect(dem_path)
    assert_chunked_computed(aspect, "Aspect")
    rasters.write(aspect, aspect_path_out, dtype="float32")
    ci.assert_raster_almost_equal(aspect_path, aspect_path_out, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_hillshade(tmp_path, dem_path):
    """Test hillshade function"""
    hlsd_path = rasters_path().joinpath("hillshade.tif")
    hlsd_path_out = get_output(tmp_path, "hillshade_out.tif", DEBUG)

    # Hillshade
    hlsd = rasters.hillshade(dem_path, 34.0, 45.2)
    assert_chunked_computed(hlsd, "Hillshade")
    rasters.write(hlsd, hlsd_path_out, dtype="float32")
    ci.assert_raster_almost_equal(hlsd_path, hlsd_path_out, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_slope(tmp_path, dem_path):
    """Test slope function"""
    slope_path = rasters_path().joinpath("slope.tif")
    slope_path_out = get_output(tmp_path, "slope_out.tif", DEBUG)
    # Slope
    slp = rasters.slope(dem_path)
    assert_chunked_computed(slp, "Slope")
    rasters.write(slp, slope_path_out, dtype="float32")
    ci.assert_raster_almost_equal(slope_path, slope_path_out, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_slope_rad(tmp_path, dem_path):
    """Test slope (radian) function"""
    slope_r_path = rasters_path().joinpath("slope_r.tif")
    slope_r_path_out = get_output(tmp_path, "slope_r_out.tif", DEBUG)

    # Slope rad
    slp_r = rasters.slope(dem_path, in_pct=False, in_rad=True)
    assert_chunked_computed(slp_r, "Slope (radian)")
    rasters.write(slp_r, slope_r_path_out, dtype="float32")
    ci.assert_raster_almost_equal(slope_r_path, slope_r_path_out, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_slope_pct(tmp_path, dem_path):
    """Test slope (pct) function"""
    slope_p_path = rasters_path().joinpath("slope_p.tif")
    slope_p_path_out = get_output(tmp_path, "slope_p_out.tif", DEBUG)

    # Slope pct
    slp_p = rasters.slope(dem_path, in_pct=True)
    assert_chunked_computed(slp_p, "Slope (%)")
    rasters.write(slp_p, slope_p_path_out, dtype="float32")
    ci.assert_raster_almost_equal(slope_p_path, slope_p_path_out, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_rasterize_binary(tmp_path, raster_path):
    """Test rasterize fct"""
    xda = get_xda(raster_path)

    # Binary vector
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_true_bin_path = rasters_path().joinpath("rasterized_bin.tif")
    out_bin_path = get_output(tmp_path, "out_bin.tif", DEBUG)

    # Rasterize
    rast_bin = rasters.rasterize(xda, vec_path, **KAPUT_KWARGS)
    assert_chunked_computed(rast_bin, "Rasterize DataArray (binary vector)")
    rasters.write(rast_bin, out_bin_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_path, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_rasterize_float(tmp_path, raster_path):
    # Binary vector with floating point raster
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_float_path = rasters_path().joinpath("raster_float.tif")
    raster_true_bin_path = rasters_path().joinpath("rasterized_bin.tif")
    out_bin_path = get_output(tmp_path, "out_bin_float.tif", DEBUG)

    # Rasterize
    rast_bin = rasters.rasterize(rasters.read(raster_float_path), vec_path)
    assert_chunked_computed(
        rast_bin, "Rasterize DataArray (binary vector with floating point raster)"
    )
    rasters.write(rast_bin, out_bin_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_path, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_rasterize_value_field(tmp_path, raster_path):
    # Use value_field
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_true_path = rasters_path().joinpath("rasterized.tif")
    out_path = get_output(tmp_path, "out_rasterized.tif", DEBUG)

    # Rasterize
    rast = rasters.rasterize(
        raster_path, vec_path, value_field="raster_val", dtype=np.uint8
    )
    assert_chunked_computed(rast, "Rasterize DataArray (use value_field)")
    rasters.write(rast, out_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_path, out_path, decimal=4)


@s3_env
@dask_env(nof_computes=1)  # Write x1
def test_rasterize_multi_band_raster(tmp_path, raster_path):
    xda = get_xda(raster_path)
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_true_bin_path = rasters_path().joinpath("rasterized_bin.tif")

    # Multiband raster
    out_bin_mb_path = get_output(tmp_path, "out_bin_mb.tif", DEBUG)

    # Rasterize
    rast_bin = rasters.rasterize(
        xr.concat([xda, xda], dim="band"), vec_path, **KAPUT_KWARGS
    )
    assert_chunked_computed(rast_bin, "Rasterize DataArray (multiband raster)")
    rasters.write(rast_bin, out_bin_mb_path, dtype=np.uint8, nodata=255)

    ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_mb_path, decimal=4)


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
@dask_env()
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


def test_deg_meters_conversion():
    """
    Test conversions between deg and meters
    https://wiki.openstreetmap.org/wiki/Precision_of_coordinates#Precision_of_latitudes
    """
    # from_deg_to_meters
    ci.assert_val(
        rasters.from_deg_to_meters(1), DEG_LAT_TO_M, "1 degree of latitude in meters"
    )
    ci.assert_val(
        rasters.from_deg_to_meters(1, lat=45),
        78573.71,
        "1 degree of longitude at 45 degrees of latitude in meters",
    )
    ci.assert_val(
        rasters.from_deg_to_meters(1, lat=45, decimals=0),
        78574,
        "1 degree of longitude at 45 degrees of latitude in meters with 0 decimal",
    )
    ci.assert_val(
        rasters.from_deg_to_meters(1, lat=45, decimals=0, average_lat_lon=True),
        round((DEG_LAT_TO_M + 78574) / 2),
        "1 degree of longitude at 45 degrees of latitude in meters (averaged) with 0 decimal",
    )

    # from_deg_to_meters
    ci.assert_val(
        rasters.from_meters_to_deg(0.5), 4.5e-06, "0.5 meter in degrees of latitude"
    )
    ci.assert_val(
        rasters.from_meters_to_deg(0.5, lat=45),
        6.4e-06,
        "0.5 meter in degree of longitude at 45 degrees of latitude",
    )
    ci.assert_val(
        rasters.from_meters_to_deg(0.5, lat=45, decimals=9),
        6.363e-06,
        "0.5 meter in degree of longitude at 45 degrees of latitude with 9 decimal",
    )
    ci.assert_val(
        rasters.from_meters_to_deg(0.5, lat=45, decimals=9, average_lat_lon=True),
        5.432e-06,
        "0.5 meter in degree of longitude at 45 degrees of latitude (averaged) with 9 decimal",
    )


@s3_env
@dask_env(nof_computes=1)  # write x1
def test_classify(tmp_path):
    """Test classify"""
    d_ndvi_path = rasters_path() / "20200824_S2_20200908_S2_dNDVI.tif"
    sev_truth = rasters_path() / "fire_sev_ndvi_truth.tif"
    sev_out = get_output(tmp_path, "fire_sev_ndvi.tif", DEBUG, dask_folders=True)
    sev = rasters.classify(
        rasters.read(d_ndvi_path), bins=[0.2, 0.55], values=[2, 3, 4]
    )
    assert_chunked_computed(sev, "Classify DataArray")
    rasters.write(sev, sev_out, dtype=np.uint8, nodata=255)
    ci.assert_raster_equal(sev_truth, sev_out)


@s3_env
# @dask_env(nof_computes=1)  # write x1 RPC is not lazy yet, no need to test it multiple times
def test_reproject_rpc(tmp_path):
    """Test reproject with rpc"""
    spot_path = (
        rasters_path()
        / "IMG_SPOT7_MS_001_A"
        / "DIM_SPOT7_MS_201602150257025_SEN_1671661101.XML"
    )
    copdem_path = rasters_path() / "Copernicus_DSM_10_S07_00_E106_00_DEM.tif"
    with rasterio.open(str(spot_path)) as ds:
        rpc_reproj = rasters.reproject(
            spot_path, rpcs=ds.rpcs, dem_path=copdem_path, dst_crs="epsg:4326", nodata=0
        )

    ci.assert_val(rpc_reproj.rio.crs.to_epsg(), 4326, "Reprojected CRS")
    ci.assert_val(rpc_reproj.rio.count, 4, "Reprojected count")
    ci.assert_val(rpc_reproj.rio.encoded_nodata, 0, "Reprojected encoded nodata")
    ci.assert_val(rpc_reproj.rio.nodata, np.nan, "Reprojected nodata")


@s3_env
@dask_env(nof_computes=0)
def test_reproject(tmp_path, raster_path):
    """Test reproject"""
    # xda
    xda = get_xda(raster_path)

    # Reproject to a pixel_size
    size_arr = rasters.reproject(xda, pixel_size=60)
    ci.assert_val(round(size_arr.rio.resolution()[0]), 60, "Reprojected pixel size")
    ci.assert_val(size_arr.rio.count, 1, "Reprojected count")
    with pytest.raises(AssertionError):
        ci.assert_val(round(xda.rio.resolution()[0]), 60, "Native pixel pize")

    # Reproject to a shape
    shape_arr = rasters.reproject(xda, shape=(161, 232))
    ci.assert_val(shape_arr.shape, (1, 161, 232), "Reprojected shape")
    ci.assert_val(shape_arr.rio.shape, (161, 232), "Reprojected shape (rio)")
    ci.assert_val(shape_arr.rio.count, 1, "Reprojected count")
    with pytest.raises(AssertionError):
        ci.assert_val(xda.rio.shape, (1, 161, 232), "Native shape")

    # Reproject to espg:4326
    wgs84_arr = rasters.reproject(xda, dst_crs="EPSG:4326")
    ci.assert_val(wgs84_arr.rio.crs.to_epsg(), 4326, "Reprojected CRS")
    with pytest.raises(AssertionError):
        ci.assert_val(xda.rio.crs.to_epsg(), 4326, "Native CRS")
