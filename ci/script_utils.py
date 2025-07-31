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
import logging
import os
import sys
from enum import unique
from functools import wraps

import tempenv

from sertit import AnyPath, ci, dask, unistra
from sertit.logs import SU_NAME
from sertit.misc import ListEnum

LOGGER = logging.getLogger(SU_NAME)

CI_SERTIT_S3 = "CI_SERTIT_USE_S3"
""" Use files stored in S3 in CI. Else uses on disk files. """

CI_SERTIT_USE_DASK = "CI_SERTIT_USE_DASK"
""" Use Dask in CI (rasters only, chunks set to auto). If not, set chunks to None. """

CI_SERTIT_TEST_LAZY = "CI_SERTIT_TEST_LAZY"
"""
Test laziness if set to 1.  
This exists because of the difficulty of creating lazy raster functions. 

WARNING: 

For now we assume function is lazy == output is still chunked.  
However, this IS abusive as some functions may compute internally and rechunk the data in the output:  
=> If a process is lazy, then its output is chunked, but the other way around is false.

How to check this behavior is not clear yet.
"""

KAPUT_KWARGS = {"fdezf": 0}

TEST_LAZY = True


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


def set_dask_env_var():
    if os.getenv(CI_SERTIT_USE_DASK, "1") == "1":
        os.environ[dask.SERTIT_DEFAULT_CHUNKS] = "auto"
    else:
        os.environ[dask.SERTIT_DEFAULT_CHUNKS] = "none"


def dask_env(nof_computes=None, max_computes=0):
    """
    Create dask-using environment

    Returns:
        Callable: decorated function
    """

    def dask_env_decorator(function):
        # Test laziness
        os.environ[CI_SERTIT_TEST_LAZY] = "1" if TEST_LAZY else "0"

        @wraps(function)
        def dask_env_wrapper(*_args, **_kwargs):
            """S3 environment wrapper"""
            # Not dask
            LOGGER.info("Using NUMPY")
            os.environ[CI_SERTIT_USE_DASK] = "0"
            set_dask_env_var()
            function(*_args, **_kwargs)

            # Dask
            os.environ[CI_SERTIT_USE_DASK] = "1"
            set_dask_env_var()

            with (
                dask.get_or_create_dask_client(),
                dask.raise_if_dask_computes(
                    nof_computes=nof_computes,
                    max_computes=max_computes,
                    dont_raise=False,  # os.environ[CI_SERTIT_TEST_LAZY] == "0", -> putting the right 'compute' number in nof_computes does the trick
                ),
            ):
                LOGGER.info("Using DASK multithreaded.")
                function(*_args, **_kwargs)

            with (
                dask.get_or_create_dask_client(processes=True),
                dask.raise_if_dask_computes(
                    nof_computes=nof_computes,
                    max_computes=max_computes,
                    dont_raise=False,  # os.environ[CI_SERTIT_TEST_LAZY] == "0", -> putting the right 'compute' number in nof_computes does the trick
                ),
            ):
                LOGGER.info("Using DASK with local cluster")
                function(*_args, **_kwargs)

        os.environ.pop(CI_SERTIT_USE_DASK, None)
        return dask_env_wrapper

    return dask_env_decorator


def is_not_lazy_yet(function):
    """
    The tested fiunction is not lazy yet : don't evaluate it

    Returns:
        Callable: decorated function
    """

    @wraps(function)
    def is_not_lazy_yet_wrapper(*_args, **_kwargs):
        """S3 environment wrapper"""
        with tempenv.TemporaryEnvironment({CI_SERTIT_TEST_LAZY: "0"}):
            LOGGER.warning(
                "This function is not lazy yet. Laziness won't be tested here."
            )
            function(*_args, **_kwargs)

    return is_not_lazy_yet_wrapper


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


def get_output(tmp, file, debug=False, dask_folders=False):
    if debug:
        out_path = AnyPath(__file__).resolve().parent / "ci_output"

        if dask_folders:
            client = dask.get_client()
            if client:
                multithreading = not client.cluster.processes
                if multithreading:
                    out_path /= "dask_multithreading"
                else:
                    out_path /= "dask_multiple_clusters"
            else:
                out_path /= "numpy"

        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / file
    else:
        return AnyPath(tmp, file)


def assert_chunked_computed(result, text):
    """ """

    if os.environ[CI_SERTIT_TEST_LAZY] == "1":
        if os.environ[CI_SERTIT_USE_DASK] == "1":
            ci.assert_chunked(result), f"{text}: Your data should be chunked!"
        else:
            ci.assert_computed(result), f"{text}: Your data should be computed!"
    else:
        if os.environ[CI_SERTIT_USE_DASK] == "1" and not dask.is_chunked(result):
            LOGGER.error(
                f"{text}: You are currently using dask and therefore your function should be chunked. However, your output has been computed."
            )
        elif os.environ[CI_SERTIT_USE_DASK] == "0" and dask.is_chunked(result):
            LOGGER.error(
                f"{text}: You are currently using numpy and therefore your function should load data into memory. However, your output is lazy."
            )
