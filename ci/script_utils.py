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
import os
import sys
from enum import unique
from functools import wraps

from sertit import AnyPath, dask, unistra
from sertit.misc import ListEnum

CI_SERTIT_S3 = "CI_SERTIT_USE_S3"

KAPUT_KWARGS = {"fdezf": 0}


@unique
class Polarization(ListEnum):
    """SAR Polarizations"""

    hh = "HH"
    vv = "VV"
    vh = "VH"
    hv = "HV"


def get_s3_ci_path():
    """Get S3 CI path"""
    unistra.define_s3_client()
    return AnyPath("s3://sertit-sertit-utils-ci")


def get_ci_data_path():
    """Get CI DATA path"""
    if int(os.getenv(CI_SERTIT_S3, 1)) and sys.platform != "win32":
        return get_s3_ci_path() / "DATA"
    else:
        return AnyPath(unistra.get_db3_path()) / "CI" / "sertit_utils" / "DATA"


def dask_env(function):
    """
    Create dask-using environment

    Returns:
        Callable: decorated function
    """

    @wraps(function)
    def dask_env_wrapper(*_args, **_kwargs):
        """S3 environment wrapper"""
        try:
            with dask.get_or_create_dask_client():
                print("Using DASK multithreaded.")
                function(*_args, **_kwargs)

            with dask.get_or_create_dask_client():
                print("Using DASK with local cluster")
                function(*_args, **_kwargs)
        except ImportError:
            pass

        print("Using NUMPY")
        function(*_args, **_kwargs)

    return dask_env_wrapper


def rasters_path():
    return get_ci_data_path().joinpath("rasters")


def vectors_path():
    return get_ci_data_path().joinpath("vectors")


def geometry_path():
    return get_ci_data_path().joinpath("geometry")


def files_path():
    return get_ci_data_path().joinpath("files")


def display_path():
    return get_ci_data_path().joinpath("display")


def xml_path():
    return get_ci_data_path().joinpath("xml")


def s3_env(*args, **kwargs):
    return unistra.s3_env(use_s3_env_var=CI_SERTIT_S3, *args, **kwargs)  # noqa: B026
