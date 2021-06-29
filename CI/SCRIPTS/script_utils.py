# -*- coding: utf-8 -*-
# Copyright 2021, SERTIT-ICube - France, https://sertit.unistra.fr/
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
import os
from collections import Callable
from enum import unique
from functools import wraps

import rasterio
from cloudpathlib import AnyPath, S3Client

from sertit.misc import ListEnum

AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_S3_ENDPOINT = "s3.unistra.fr"
CI_SERTIT_S3 = "CI_SERTIT_USE_S3"


@unique
class Polarization(ListEnum):
    """SAR Polarizations"""

    hh = "HH"
    vv = "VV"
    vh = "VH"
    hv = "HV"


def get_proj_path():
    """Get project path"""
    if int(os.getenv(CI_SERTIT_S3, 0)):
        # ON S3
        client = S3Client(
            endpoint_url=f"https://{AWS_S3_ENDPOINT}",
            aws_access_key_id=os.getenv(AWS_ACCESS_KEY_ID),
            aws_secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY),
        )
        client.set_as_default_client()
        return AnyPath("s3://sertit-sertit-utils-ci")
    else:
        # ON DISK
        return AnyPath(__file__).parent.parent.parent


def get_ci_data_path():
    """Get CI DATA path"""
    if int(os.getenv(CI_SERTIT_S3, 0)):
        return get_proj_path().joinpath("DATA")
    else:
        return get_proj_path().joinpath("CI", "DATA")


def s3_env(function: Callable):
    """
    Create S3 compatible storage environment
    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
    """

    @wraps(function)
    def s3_env_wrapper():
        """ S3 environment wrapper """
        os.environ[CI_SERTIT_S3] = "0"
        print("Using on disk files")
        function()

        if os.getenv(AWS_SECRET_ACCESS_KEY):
            os.environ[CI_SERTIT_S3] = "1"
            print("Using S3 files")
            with rasterio.Env(
                CPL_CURL_VERBOSE=True,
                AWS_VIRTUAL_HOSTING=False,
                AWS_S3_ENDPOINT=AWS_S3_ENDPOINT,
                GDAL_DISABLE_READDIR_ON_OPEN=False,
            ):
                function()

            os.environ[CI_SERTIT_S3] = "0"

    return s3_env_wrapper


def rasters_path():
    return get_ci_data_path().joinpath("rasters")


def vectors_path():
    return get_ci_data_path().joinpath("vectors")


def files_path():
    return get_ci_data_path().joinpath("files")


def display_path():
    return get_ci_data_path().joinpath("display")
