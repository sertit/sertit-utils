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
import os
import sys
from collections.abc import Callable
from enum import unique
from functools import wraps

from cloudpathlib import AnyPath

from sertit import ci
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


def get_s3_ci_path():
    """Get S3 CI path"""
    ci.define_s3_client()
    return AnyPath("s3://sertit-sertit-utils-ci")


def get_proj_path():
    """Get project path"""
    if int(os.getenv(CI_SERTIT_S3, 1)) and sys.platform != "win32":
        return get_s3_ci_path()
    else:
        # ON DISK
        return AnyPath(ci.get_db3_path())


def get_ci_data_path():
    """Get CI DATA path"""
    if int(os.getenv(CI_SERTIT_S3, 1)) and sys.platform != "win32":
        return get_proj_path().joinpath("DATA")
    else:
        return get_proj_path().joinpath("CI", "sertit_utils", "DATA")


def dask_env(function: Callable):
    """
    Create dask-using environment
    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
    """

    @wraps(function)
    def dask_env_wrapper():
        """S3 environment wrapper"""
        try:
            from dask.distributed import Client, LocalCluster

            with LocalCluster() as cluster, Client(cluster):
                print("Using DASK")
                function()
        except ImportError:
            pass

        print("Using NUMPY")
        function()

    return dask_env_wrapper


def rasters_path():
    return get_ci_data_path().joinpath("rasters")


def vectors_path():
    return get_ci_data_path().joinpath("vectors")


def files_path():
    return get_ci_data_path().joinpath("files")


def display_path():
    return get_ci_data_path().joinpath("display")


def xml_path():
    return get_ci_data_path().joinpath("xml")


def s3_env(function):
    return ci.s3_env(function, use_s3_env_var=CI_SERTIT_S3)
