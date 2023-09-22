# -*- coding: utf-8 -*-
# Copyright 2023, SERTIT-ICube - France, https://sertit.unistra.fr/
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
""" Script testing the CI """
import pytest
from cloudpathlib import AnyPath, S3Client
from tempenv import tempenv

from CI.SCRIPTS.script_utils import CI_SERTIT_S3, s3_env
from sertit import misc
from sertit.unistra import (
    SU_USE_S3,
    _get_db_path,
    get_db2_path,
    get_db3_path,
    get_db4_path,
    get_geodatastore,
)


def test_unistra_s3():
    with tempenv.TemporaryEnvironment({SU_USE_S3: "1", CI_SERTIT_S3: "1"}):
        # Test s3_env and define_s3_client (called inside)
        def base_fct():
            raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
                "DATA", "rasters", "raster.tif"
            )
            assert (
                raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
            )
            assert raster_path.is_file()

        @s3_env
        def with_s3():
            base_fct()

        def without_s3():
            S3Client().set_as_default_client()
            base_fct()

        with pytest.raises(AssertionError):
            without_s3()

        with_s3()

        # Test get_geodatastore with s3
        assert str(get_geodatastore()) == "s3://sertit-geodatastore"

    # Test get_geodatastore without s3
    with tempenv.TemporaryEnvironment({SU_USE_S3: "0"}):
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
