# -*- coding: utf-8 -*-
# Copyright 2022, SERTIT-ICube - France, https://sertit.unistra.fr/
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
""" Script testing SNAP functions """

import logging

import numpy as np

from CI.SCRIPTS.script_utils import dask_env, display_path, s3_env
from sertit import ci, display, rasters_rio

ci.reduce_verbosity()

LOGGER = logging.getLogger("Test_logger")


@s3_env
@dask_env
def test_display():
    """Testing Display functions"""
    path_2d = display_path().joinpath("2d.tif")
    path_3d = display_path().joinpath("3d.tif")
    path_3d_minmax = display_path().joinpath("3d_minmax.tif")
    path_stack = display_path().joinpath("stack.tif")

    # Open data
    arr_2d, _ = rasters_rio.read(path_2d)
    arr_3d, _ = rasters_rio.read(path_3d)
    arr_3d_m, _ = rasters_rio.read(path_3d_minmax)
    stack, _ = rasters_rio.read(path_stack)

    # Scale data
    scaled_2d = display.scale(stack[0, ...])
    scaled_3d = display.scale(stack)
    scaled_3d_m = display.scale(stack, perc=0)

    # Test
    np.testing.assert_array_equal(scaled_2d, arr_2d[0, ...])
    np.testing.assert_array_equal(scaled_3d, arr_3d)
    np.testing.assert_almost_equal(scaled_3d_m, arr_3d_m)  # Almost here

    # Scale to uint8
    scaled_uint8_2d = display.scale_to_uint8(stack[0, ...])
    scaled_uint8_3d = display.scale_to_uint8(stack)

    assert scaled_uint8_2d.dtype == scaled_uint8_3d.dtype == np.uint8
    np.testing.assert_array_equal(
        scaled_uint8_2d, (arr_2d[0, ...] * 254 + 1).astype(np.uint8)
    )
    np.testing.assert_array_equal(scaled_uint8_3d, (arr_3d * 254 + 1).astype(np.uint8))
