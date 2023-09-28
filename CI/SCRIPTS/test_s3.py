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

from CI.SCRIPTS.script_utils import CI_SERTIT_S3
from sertit.s3 import USE_S3_STORAGE, s3_env
from sertit.unistra import UNISTRA_S3_ENPOINT


def test_s3():
    with tempenv.TemporaryEnvironment({USE_S3_STORAGE: "1", CI_SERTIT_S3: "1"}):
        # Test s3_env and define_s3_client (called inside)
        def base_fct(value):
            raster_path = AnyPath("s3://sertit-sertit-utils-ci").joinpath(
                "DATA", "rasters", "raster.tif"
            )
            assert (
                raster_path.client.client.meta.endpoint_url == "https://s3.unistra.fr"
            )
            assert raster_path.is_file()

        @s3_env(default_endpoint=UNISTRA_S3_ENPOINT, use_s3_env_var=CI_SERTIT_S3)
        def with_s3():
            base_fct(1)

        def without_s3():
            S3Client().set_as_default_client()
            base_fct(None)

        with pytest.raises(AssertionError):
            without_s3()

        with_s3()
