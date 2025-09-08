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
"""Script testing raster function (with rasterio)"""

import os
import shutil

import numpy as np
import pytest
import rasterio
import shapely
from rasterio.windows import Window

from ci.script_utils import KAPUT_KWARGS, rasters_path, s3_env
from sertit import ci, rasters_rio, vectors
from sertit.rasters_rio import any_raster_to_rio_ds, get_nodata_value_from_dtype
from sertit.vectors import EPSG_4326

ci.reduce_verbosity()


@pytest.fixture
def raster_path():
    return rasters_path().joinpath("raster.tif")


@pytest.fixture
def raster_meta(raster_path):
    return rasters_rio.read(raster_path, **KAPUT_KWARGS)


@pytest.fixture
def mask_path():
    return rasters_path().joinpath("raster_mask.geojson")


@pytest.fixture
def mask(mask_path):
    return vectors.read(mask_path)


@pytest.fixture
def ds_dtype(raster_path):
    with rasterio.open(str(raster_path)) as ds:
        return ds.meta["dtype"]


@s3_env
def test_read(tmp_path, raster_path, mask_path, raster_meta):
    """Test read functions"""
    raster_window_path = rasters_path().joinpath("window.tif")
    raster_window_20_path = rasters_path().joinpath("window_20.tif")
    extent_path = rasters_path().joinpath("extent.geojson")
    footprint_path = rasters_path().joinpath("footprint.geojson")

    # Get Extent
    extent = rasters_rio.get_extent(raster_path)
    truth_extent = vectors.read(extent_path)
    ci.assert_geom_equal(extent, truth_extent)

    # Get Footprint
    footprint = rasters_rio.get_footprint(raster_path)
    truth_footprint = vectors.read(footprint_path)
    ci.assert_geom_equal(footprint, truth_footprint)

    # Read
    raster, meta = raster_meta
    with rasterio.open(str(raster_path)) as ds:
        raster_1, meta1 = rasters_rio.read(ds, resolution=20)
        raster_2, _ = rasters_rio.read(ds, resolution=[20, 20])
        raster_3, _ = rasters_rio.read(ds, size=(meta1["width"], meta1["height"]))
        raster_4, _ = rasters_rio.read(raster_path, indexes=[1])
        with pytest.raises(ValueError):
            rasters_rio.read(ds, resolution=[20, 20, 20])

        assert raster.shape == (ds.count, ds.height, ds.width)
        assert meta["crs"] == ds.crs
        assert meta["transform"] == ds.transform
        np.testing.assert_array_equal(raster_1, raster_2)
        np.testing.assert_array_equal(raster_1, raster_3)
        np.testing.assert_array_equal(raster, raster_4)  # 2D array

        # -- Read with window
        window_out = os.path.join(tmp_path, "test_xda_window.tif")
        window, w_mt = rasters_rio.read(
            raster_path,
            window=mask_path,
        )
        rasters_rio.write(window, w_mt, window_out, **KAPUT_KWARGS)
        ci.assert_raster_equal(window_out, raster_window_path)

        # Gdf
        window_20_out = os.path.join(tmp_path, "test_xda_20_window.tif")
        gdf = vectors.read(mask_path)
        bounds = gdf.bounds.values[0]
        window_20, w_mt_20 = rasters_rio.read(
            raster_path, window=gdf.to_crs(EPSG_4326), resolution=20
        )
        rasters_rio.write(window_20, w_mt_20, window_20_out)
        ci.assert_raster_equal(window_20_out, raster_window_20_path)

        # Bounds
        window_20_2, w_mt_20_2 = rasters_rio.read(
            raster_path, window=bounds, resolution=20
        )
        np.testing.assert_array_equal(window_20, window_20_2)
        ci.assert_meta(w_mt_20, w_mt_20_2)

        # Window
        window_20_3, w_mt_20_3 = rasters_rio.read(
            raster_path,
            window=Window(col_off=57, row_off=0, width=363, height=321),
            resolution=20,
        )
        np.testing.assert_array_equal(window_20, window_20_3)
        ci.assert_meta(w_mt_20, w_mt_20_3)

        with pytest.raises(FileNotFoundError):
            rasters_rio.read(
                raster_path,
                window=rasters_path().joinpath("non_existing_window.kml"),
            )


@s3_env
def test_write(tmp_path, raster_path, raster_meta):
    """Test write functions"""
    raster_out = os.path.join(tmp_path, "test.tif")

    rasters_rio.write(*raster_meta, raster_out)

    assert os.path.isfile(raster_out)
    ci.assert_raster_equal(raster_path, raster_out)


@s3_env
def test_mask(tmp_path, raster_path, mask):
    """Test mask function"""
    raster_masked_path = rasters_path().joinpath("raster_masked.tif")
    raster_masked_out = os.path.join(tmp_path, "test_mask.tif")

    with rasterio.open(str(raster_path)) as ds:
        mask_arr, mask_meta = rasters_rio.mask(ds, mask.geometry)

    rasters_rio.write(mask_arr, mask_meta, raster_masked_out)
    ci.assert_raster_equal(raster_masked_out, raster_masked_path)


@s3_env
def test_crop(tmp_path, raster_path, mask):
    """Test crop function"""
    # Crop
    raster_cropped_path = rasters_path().joinpath("raster_cropped.tif")
    raster_cropped_out = os.path.join(tmp_path, "test_crop.tif")

    with rasterio.open(str(raster_path)) as ds:
        crop_arr, crop_meta = rasters_rio.crop(ds, mask)

    rasters_rio.write(crop_arr, crop_meta, raster_cropped_out)
    ci.assert_raster_equal(raster_cropped_out, raster_cropped_path)


@s3_env
def test_sieve(tmp_path, raster_meta):
    """Test sieve function"""
    raster_sieved_path = rasters_path().joinpath("raster_sieved.tif")

    # Sieve
    raster, meta = raster_meta
    sieve_out = os.path.join(tmp_path, "test_sieved.tif")

    sieve_arr, sieve_meta = rasters_rio.sieve(
        raster, meta, sieve_thresh=20, connectivity=4
    )

    rasters_rio.write(sieve_arr, sieve_meta, sieve_out, nodata=255)
    ci.assert_raster_equal(sieve_out, raster_sieved_path)


@s3_env
def test_collocate(raster_meta):
    """Test collocate function"""
    raster, meta = raster_meta

    coll_arr, coll_meta = rasters_rio.collocate(
        meta, raster, meta
    )  # Just hope that it doesnt crash

    assert coll_meta == meta


@s3_env
def test_merge_gtiff(tmp_path, raster_path):
    """Test merge_gtiff function"""
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")
    raster_merged_gtiff_path = rasters_path().joinpath("raster_merged.tif")
    raster_merged_gtiff_out = os.path.join(tmp_path, "test_merged.tif")

    rasters_rio.merge_gtiff(
        [raster_path, raster_to_merge_path],
        raster_merged_gtiff_out,
        method="max",
        driver="GTiff",
        dtype="uint8",
    )

    ci.assert_raster_equal(raster_merged_gtiff_out, raster_merged_gtiff_path)


@s3_env
def test_vectorize(raster_path):
    """Test vectorize function"""
    if shapely.__version__ >= "1.8a1":
        vect_truth_path = rasters_path().joinpath("vector.geojson")
        diss_truth_path = rasters_path().joinpath("dissolved.geojson")
    else:
        print("USING OLD VECTORS")
        vect_truth_path = rasters_path().joinpath("vector_old.geojson")
        diss_truth_path = rasters_path().joinpath("dissolved_old.geojson")

    # Vectorize
    val = 2
    vect = rasters_rio.vectorize(raster_path)
    vect_val = rasters_rio.vectorize(raster_path, values=val)
    vect_val_diss = rasters_rio.vectorize(raster_path, values=val, dissolve=True)
    vect_val_disc = rasters_rio.vectorize(
        raster_path, values=[1, 255], keep_values=False
    )
    vect_truth = vectors.read(vect_truth_path)
    diss_truth = vectors.read(diss_truth_path)
    ci.assert_geom_equal(vect, vect_truth)
    ci.assert_geom_equal(vect_val, vect_truth.loc[vect_truth.raster_val == val])
    ci.assert_geom_equal(vect_val_diss, diss_truth)
    ci.assert_geom_equal(vect_val_disc, vect_truth.loc[vect_truth.raster_val == val])


@s3_env
def test_get_valid_vector(raster_path):
    """Test get_valid_vector function"""
    valid_truth_path = rasters_path().joinpath("valid.geojson")

    valid_vec = rasters_rio.get_valid_vector(raster_path)

    valid_truth = vectors.read(valid_truth_path)
    ci.assert_geom_equal(valid_vec, valid_truth)


@s3_env
def test_nodata_vec(raster_path):
    """Test get_nodata_vector function"""
    nodata_truth_path = rasters_path().joinpath("nodata.geojson")

    nodata_vec = rasters_rio.get_nodata_vector(raster_path)

    nodata_truth = vectors.read(nodata_truth_path)
    ci.assert_geom_equal(nodata_vec, nodata_truth)


@s3_env
@pytest.mark.skipif(
    shutil.which("gdalbuildvrt") is None,
    reason="Only works if gdalbuildvrt can be found.",
)
def test_vrt(tmp_path, raster_path):
    raster_merged_vrt_path = rasters_path().joinpath("raster_merged.vrt")
    raster_to_merge_path = rasters_path().joinpath("raster_to_merge.tif")

    # Merge VRT
    raster_merged_vrt_out = os.path.join(tmp_path, "test_merged.vrt")
    rasters_rio.merge_vrt([raster_path, raster_to_merge_path], raster_merged_vrt_out)
    ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)

    os.remove(raster_merged_vrt_out)

    rasters_rio.merge_vrt(
        [raster_path, raster_to_merge_path],
        raster_merged_vrt_out,
        abs_path=True,
        **KAPUT_KWARGS,
    )
    ci.assert_raster_equal(raster_merged_vrt_out, raster_merged_vrt_path)


def test_dim():
    """Test on BEAM-DIMAP function"""
    dim_path = rasters_path().joinpath("DIM.dim")
    dim_img_path = rasters_rio.get_dim_img_path(dim_path)
    assert dim_img_path.is_file(), f"{dim_img_path} is not a file!"
    assert dim_img_path == rasters_path().joinpath("DIM.data", "dim.img")


def test_bit():
    """Test bit arrays"""
    np_ones = np.ones((1, 2, 2), dtype=np.uint16)
    ones = rasters_rio.read_bit_array(np_ones, bit_id=0)
    zeros = rasters_rio.read_bit_array(np_ones, bit_id=list(np.arange(1, 15)))
    assert (np_ones == ones).all()
    for arr in zeros:
        assert (np_ones == 1 + arr).all()


@s3_env
def test_dem_fct(tmp_path):
    """Test DEM fct, i.e. slope and hillshade"""
    # Paths IN
    dem_path = rasters_path().joinpath("dem.tif")
    hlsd_path = rasters_path().joinpath("hillshade_rio.tif")
    slope_path = rasters_path().joinpath("slope_rio.tif")
    slope_r_path = rasters_path().joinpath("slope_r_rio.tif")
    slope_p_path = rasters_path().joinpath("slope_p_rio.tif")

    # Path OUT
    hlsd_path_out = os.path.join(tmp_path, "hillshade_out_rio.tif")
    slope_path_out = os.path.join(tmp_path, "slope_out_rio.tif")
    slope_r_path_out = os.path.join(tmp_path, "slope_out_r_rio.tif")
    slope_p_path_out = os.path.join(tmp_path, "slope_out_p_rio.tif")

    # Compute
    hlsd, meta = rasters_rio.hillshade(dem_path, 34.0, 45.2)
    rasters_rio.write(hlsd, meta, hlsd_path_out, dtype="float32")

    slp, meta = rasters_rio.slope(dem_path)
    rasters_rio.write(slp, meta, slope_path_out, dtype="float32")

    slp_r, meta = rasters_rio.slope(dem_path, in_pct=False, in_rad=True)
    rasters_rio.write(slp_r, meta, slope_r_path_out, dtype="float32")

    slp_p, meta = rasters_rio.slope(dem_path, in_pct=True)
    rasters_rio.write(slp_p, meta, slope_p_path_out, dtype="float32")

    # Test
    ci.assert_raster_almost_equal(hlsd_path, hlsd_path_out, decimal=4)
    ci.assert_raster_almost_equal(slope_path, slope_path_out, decimal=4)
    ci.assert_raster_almost_equal(slope_r_path, slope_r_path_out, decimal=4)
    ci.assert_raster_almost_equal(slope_p_path, slope_p_path_out, decimal=4)


@s3_env
def test_reproj(tmp_path, raster_path):
    """Test reproject fct"""
    dem_path = rasters_path().joinpath(
        "Copernicus_DSM_10_N43_00_W003_00_DEM_resampled.tif"
    )
    reproj_path = rasters_path().joinpath("reproj_out.tif")

    with rasterio.open(str(dem_path)) as src, rasterio.open(str(raster_path)) as ds:
        dst_arr, dst_meta = rasters_rio.reproject_match(
            ds.meta, src.read(), src.meta, **KAPUT_KWARGS
        )

        # from ds
        assert ds.meta["driver"] == dst_meta["driver"]
        assert ds.meta["width"] == dst_meta["width"]
        assert ds.meta["height"] == dst_meta["height"]
        assert ds.meta["transform"] == dst_meta["transform"]

        # from src
        assert src.meta["count"] == dst_meta["count"]
        assert src.meta["nodata"] == dst_meta["nodata"]
        assert src.meta["dtype"] == dst_meta["dtype"]

        path_out = os.path.join(tmp_path, "out.tif")
        rasters_rio.write(dst_arr, dst_meta, path_out)

        ci.assert_raster_almost_equal(path_out, reproj_path, decimal=4)


@s3_env
def test_rasterize(tmp_path, raster_path):
    """Test rasterize fct"""
    vec_path = rasters_path().joinpath("vector.geojson")
    raster_true_bin_path = rasters_path().joinpath("rasterized_bin.tif")
    raster_true_path = rasters_path().joinpath("rasterized.tif")

    # Binary vector
    out_bin_path = os.path.join(tmp_path, "out_bin.tif")
    rast_bin, bin_meta = rasters_rio.rasterize(raster_path, vec_path)
    rasters_rio.write(rast_bin, bin_meta, out_bin_path)

    ci.assert_raster_almost_equal(raster_true_bin_path, out_bin_path, decimal=4)

    # Vector
    out_path = os.path.join(tmp_path, "out.tif")
    rast, meta = rasters_rio.rasterize(
        raster_path, vec_path, "raster_val", dtype=np.uint8
    )
    rasters_rio.write(rast, meta, out_path)

    ci.assert_raster_almost_equal(raster_true_path, out_path, decimal=4)


def test_read_idx():
    """Test mtd after reading with index"""
    l1_path = rasters_path().joinpath("19760712T093233_L1_215030_MSS_stack.tif")

    def _test_idx(idx_list):
        raster, meta = rasters_rio.read(l1_path, indexes=idx_list)

        if isinstance(idx_list, list):
            nof_idx = len(idx_list)
            shape = (meta["count"], meta["height"], meta["width"])
        else:
            nof_idx = 1
            shape = (meta["height"], meta["width"])

        ci.assert_val(meta["count"], nof_idx, "Count")
        ci.assert_val(raster.shape, shape, "Shape")

    # Tests
    _test_idx([1])
    _test_idx([1, 2])
    _test_idx(1)


@s3_env
def test_decorator_deprecation(raster_path):
    from sertit.rasters_rio import path_arr_dst

    @any_raster_to_rio_ds
    def _ok_rasters(ds):
        return ds.read()

    @path_arr_dst
    def _depr_rasters(ds):
        return ds.read()

    # Not able to warn deprecation from inside the decorator
    np.testing.assert_equal(_ok_rasters(raster_path), _depr_rasters(raster_path))


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
            from sertit.rasters_rio import get_nodata_value

            ci.assert_val(
                get_nodata_value_from_dtype(dtype), get_nodata_value(dtype), dtype
            )


@s3_env
def test_write_deprecated(tmp_path, raster_path):
    test_deprecated_path = os.path.join(tmp_path, "test_depr.tif")
    raster, mtd = rasters_rio.read(raster_path)

    # test deprecation warning
    with pytest.deprecated_call():
        rasters_rio.write(raster, mtd, path=test_deprecated_path)
