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

import os
import tempfile

import pytest
import rasterio
from cloudpathlib import AnyPath, S3Client
from tempenv import tempenv

from ci.script_utils import CI_SERTIT_S3
from sertit import ci, misc, rasters, s3
from sertit.ci import AWS_ACCESS_KEY_ID, AWS_S3_ENDPOINT, AWS_SECRET_ACCESS_KEY
from sertit.unistra import (
    _get_db_path,
    get_db2_path,
    get_db3_path,
    get_db4_path,
    get_geodatastore,
    s3_env,
)

ci.reduce_verbosity()


def base_fct(value):
    raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
        "DATA", "rasters", "raster.tif"
    )
    assert raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
    assert raster_path.is_file()
    assert rasters.read(raster_path).rio.count == 1


@s3_env
def with_s3():
    base_fct(1)
    return 1


def without_s3():
    S3Client().set_as_default_client()
    base_fct(None)


@s3_env
def test_unistra_s3():
    with tempenv.TemporaryEnvironment(
        {s3.USE_S3_STORAGE: "1", AWS_S3_ENDPOINT: None, CI_SERTIT_S3: "1"}
    ):
        # Test s3_env and define_s3_client (called inside)
        raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
            "DATA", "rasters", "raster.tif"
        )
        assert raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
        assert raster_path.is_file()
        assert rasterio.open(str(raster_path)).count == 1
        assert rasters.read(raster_path).rio.count == 1

    # Test profile
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        credentials = (
            f"[unistra]\n"
            f"aws_access_key_id={os.getenv(AWS_ACCESS_KEY_ID)}\n"
            f"aws_secret_access_key={os.getenv(AWS_SECRET_ACCESS_KEY)}\n"
        )
        f.write(credentials)
        filename = f.name
    with tempenv.TemporaryEnvironment(
        {
            s3.USE_S3_STORAGE: "1",
            AWS_ACCESS_KEY_ID: None,
            AWS_SECRET_ACCESS_KEY: None,
            AWS_S3_ENDPOINT: None,
            "AWS_SHARED_CREDENTIALS_FILE": filename,
            CI_SERTIT_S3: "1",
        }
    ):
        # Test s3_env and define_s3_client (called inside)
        raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
            "DATA", "rasters", "raster.tif"
        )
        assert raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
        assert raster_path.is_file()
        assert rasterio.open(str(raster_path)).count == 1
        assert rasters.read(raster_path).rio.count == 1
    # Test get_geodatastore without s3
    with tempenv.TemporaryEnvironment({s3.USE_S3_STORAGE: "0"}):
        assert str(get_geodatastore()).endswith("BASES_DE_DONNEES")


@pytest.mark.skipif(not misc.in_docker(), reason="Only works in docker")
def test_mnt():
    """Test mounted directories"""
    try:
        assert get_db2_path() == "/mnt/ds2_db2"
        assert get_db3_path() == "/mnt/ds2_db3"
        assert get_db4_path() == "/mnt/ds2_db4"
    except NotADirectoryError:
        pass

    with pytest.raises(NotADirectoryError):
        _get_db_path(5)
