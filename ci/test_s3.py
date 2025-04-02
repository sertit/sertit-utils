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
"""Script testing the CI"""

import logging
import os

import geopandas
import pytest
import rasterio
from cloudpathlib import AnyPath, S3Client
from tempenv import tempenv

from ci.script_utils import CI_SERTIT_S3
from sertit import ci, rasters, vectors
from sertit.ci import AWS_S3_ENDPOINT
from sertit.logs import SU_NAME
from sertit.s3 import USE_S3_STORAGE, s3_env, temp_s3

LOGGER = logging.getLogger(SU_NAME)


def base_fct(value):
    raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
        "DATA", "rasters", "raster.tif"
    )
    assert raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
    assert raster_path.is_file()
    assert rasters.read(raster_path).rio.count == 1
    return 1


@s3_env(use_s3_env_var=CI_SERTIT_S3)
def with_s3(variable_1, variable_2):
    return base_fct(1)


def without_s3():
    S3Client().set_as_default_client()
    return base_fct(None)


def test_s3_raster():
    with tempenv.TemporaryEnvironment(
        {USE_S3_STORAGE: "1", AWS_S3_ENDPOINT: "s3.unistra.fr", CI_SERTIT_S3: "1"}
    ):
        # There is a mistake in endpoint but AWS_S3_ENDPOINT should override it
        with temp_s3(endpoint="mistake.s3.unistra.fr"):
            raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
                "DATA", "rasters", "raster.tif"
            )
            assert (
                raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
            )
            assert raster_path.is_file()

            # Test only rasterio since rasters module may interfere
            assert rasterio.open(str(raster_path)).count == 1
            assert rasters.read(raster_path).rio.count == 1

        with pytest.raises(AssertionError):
            without_s3()
        assert with_s3(1, 2) == 1


def test_s3_vector():
    # There is a mistake in endpoint but AWS_S3_ENDPOINT should override it
    with (
        tempenv.TemporaryEnvironment({USE_S3_STORAGE: "1", CI_SERTIT_S3: "1"}),
        temp_s3(endpoint="s3.unistra.fr"),
    ):
        vector_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
            "DATA", "vectors", "aoi.shp"
        )
        assert vector_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
        assert vector_path.is_file()
        # Test geopandas since vectors module could interfere with s3 configuration
        assert geopandas.read_file(str(vector_path)).shape[0] == 1
        assert vectors.read(vector_path).shape[0] == 1


def test_no_sign_request():
    with (
        tempenv.TemporaryEnvironment(
            {
                "AWS_S3_ENDPOINT": "s3.us-west-2.amazonaws.com",
            }
        ),
        temp_s3(no_sign_request=True),
    ):
        path = AnyPath(
            "s3://sentinel-cogs/sentinel-s2-l2a-cogs/40/V/DR/2023/11/S2A_40VDR_20231114_0_L2A"
        )
        assert path.exists()
        with rasterio.open(str(path / "B12.tif")) as ds:
            assert ds.meta["dtype"] == "uint16"


def test_requester_pays():
    with (
        tempenv.TemporaryEnvironment(
            {
                "AWS_S3_ENDPOINT": "s3.eu-central-1.amazonaws.com",
                "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_S3_AWS_SECRET_ACCESS_KEY"),
                "AWS_ACCESS_KEY_ID": os.getenv("AWS_S3_AWS_ACCESS_KEY_ID"),
            }
        ),
        temp_s3(requester_pays=True),
    ):
        path = AnyPath("s3://sentinel-s2-l1c/tiles/29/H/NA/2023/11/14/0/")
        assert path.exists()
        with rasterio.open(str(path / "B12.jp2")) as ds:
            assert ds.meta["dtype"] == "uint16"


def ko_to_bytes(value):
    return int(value * 1e3)


def mo_to_bytes(value):
    return int(value * 1e6)


def _update_s3_env(function):
    def s3_env_wrapper(*args, **kwargs):
        from sertit import unistra

        LOGGER.debug("In decorator!")
        with rasterio.Env(
            # Set to TRUE or EMPTY_DIR to avoid listing all files in the directory once a single file is opened (this is highly recommended).
            GDAL_DISABLE_READDIR_ON_OPEN=True,
            # Size of the default block cache, can be set in byte, MB, or as a percentage of available main, memory.
            GDAL_CACHEMAX=mo_to_bytes(200),
            # Global cache size for downloads in bytes, defaults to 16 MB.
            CPL_VSIL_CURL_CACHE_SIZE=mo_to_bytes(200),
            # Enable / disable per-file caching by setting to TRUE or FALSE.
            VSI_CACHE=True,
            # Per-file cache size in bytes
            VSI_CACHE_SIZE=mo_to_bytes(5),
            # When set to YES, this attempts to download multiple range requests in parallel, reusing the same TCP connection
            GDAL_HTTP_MULTIPLEX=True,
            # Gives the number of initial bytes GDAL should read when opening a file and inspecting its metadata.
            GDAL_INGESTED_BYTES_AT_OPEN=ko_to_bytes(32),
            GDAL_HTTP_VERSION=2,
            # Tells GDAL to merge consecutive range GET requests.
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            # Number of threads GDAL can use for block reads and (de)compression, set to ALL_CPUS to use all available cores.
            GDAL_NUM_THREADS="ALL_CPUS",
        ):
            return unistra.s3_env(function(*args, **kwargs), *args, **kwargs)

    return s3_env_wrapper


def _fct():
    LOGGER.debug("In _fct!")
    assert rasterio.env.hasenv()
    rasterio_env = rasterio.env.getenv()
    LOGGER.info(rasterio_env)
    print(rasterio_env)
    assert "GDAL_CACHEMAX" in rasterio_env
    assert "VSI_CACHE" in rasterio_env
    assert "GDAL_HTTP_VERSION" in rasterio_env
    assert "GDAL_DISABLE_READDIR_ON_OPEN" in rasterio_env
    ci.assert_val(rasterio_env["GDAL_CACHEMAX"], mo_to_bytes(200), "GDAL_CACHEMAX")
    ci.assert_val(rasterio_env["VSI_CACHE"], True, "VSI_CACHE")
    ci.assert_val(rasterio_env["GDAL_HTTP_VERSION"], 2, "GDAL_HTTP_VERSION")
    ci.assert_val(rasterio_env["GDAL_DISABLE_READDIR_ON_OPEN"], True, "VSI_CACHE")


@_update_s3_env
def _decorated_fct():
    return _fct()


def test_rasterio_env():
    # See https://developmentseed.org/titiler/advanced/performance_tuning/#recommended-configuration-for-dynamic-tiling
    # And https://gdalcubes.github.io/source/concepts/config.html#recommended-settings-for-cloud-access

    _decorated_fct()

    with pytest.raises(AssertionError):
        _fct()
