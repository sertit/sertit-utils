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
"""Script testing dask functions"""

import logging
import os

import pytest

from ci.script_utils import rasters_path, s3_env
from sertit import ci, dask, rasters
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)

ci.reduce_verbosity()


@s3_env
def test_computed_lazy():
    """Test read function"""
    raster_path = rasters_path().joinpath("raster.tif")

    # Read already computed
    np_arr = rasters.read(raster_path, chunks=None)
    assert np_arr is not None
    assert not dask.is_chunked(np_arr)
    ci.assert_computed(np_arr)

    # Read lazy
    da_arr = rasters.read(raster_path)
    assert da_arr is not None
    assert dask.is_chunked(da_arr)
    ci.assert_chunked(da_arr)

    # Assert computed after compute
    ci.assert_computed(da_arr.compute())


def test_raise_compute():
    raster_path = rasters_path().joinpath("raster.tif")

    # Read lazily
    with dask.raise_if_dask_computes():
        da = rasters.read(raster_path)

    # Ensure computing raises
    with (
        pytest.raises(RuntimeError),
        dask.raise_if_dask_computes(force_synchronous=True),
    ):
        da.compute()

    # Ensure computing raises
    with (
        pytest.raises(RuntimeError),
        dask.raise_if_dask_computes(force_synchronous=True, max_computes=1),
    ):
        da.compute()
        da.compute()

    # Ensure computing raises
    with (
        pytest.raises(RuntimeError),
        dask.raise_if_dask_computes(force_synchronous=True, nof_computes=1),
    ):
        da.compute()
        da.compute()

    # Ensure computing raises
    with (
        pytest.raises(RuntimeError),
        dask.raise_if_dask_computes(force_synchronous=True, nof_computes=3),
    ):
        da.compute()

    # Ensure computing don't raise
    with dask.raise_if_dask_computes(force_synchronous=True, nof_computes=1):
        da.compute()

    # Ensure computing don't raise
    with dask.raise_if_dask_computes(force_synchronous=True, max_computes=1):
        da.compute()

    # Ensure computing don't raise id specified
    with dask.raise_if_dask_computes(force_synchronous=True, dont_raise=True):
        da.compute()

    # Check if da is still chunked
    ci.assert_chunked(da)

    # First gotcha
    with (
        pytest.raises(RuntimeError),
        dask.raise_if_dask_computes(force_synchronous=True),
    ):
        if da.max() > 5:
            print("Gotcha!")

    # Check if da is still chunked
    ci.assert_chunked(da)

    # Second gotchas
    # with pytest.raises(RuntimeError), dask.raise_if_dask_computes():
    #     pass

    # Check if da is still chunked
    # ci.assert_chunked(da)

    # Other gotchas from digitize: TODO?
    # import dask.array as da
    # import xarray as xr
    # import numpy as np
    # with pytest.raises(RuntimeError), dask.raise_if_dask_computes():
    #     values = da.from_array([1, 2, 3], chunks="auto")
    #     arr = values.vindex[
    #         xr.apply_ufunc(np.digitize, a, [0.1, 0.2], dask="allowed").data
    #     ]


def test_env_var(capfd):
    with dask.get_or_create_dask_client(
        processes=True, env_vars={"PROCESSES": "True"}
    ) as client:
        client.run(lambda: print("1. processes=" + os.environ["PROCESSES"]))
    out, err = capfd.readouterr()
    assert "1. processes=True" in out
    assert "1. processes=False" not in out

    with dask.get_or_create_dask_client(
        processes=False, env_vars={"PROCESSES": "False"}
    ) as client:
        client.run(lambda: print("2. processes=" + os.environ["PROCESSES"]))
    out, err = capfd.readouterr()
    assert "2. processes=False" in out
    assert "2. processes=True" not in out
