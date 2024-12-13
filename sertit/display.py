# Copyright 2024, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""
Display tools
"""

import numpy as np

from sertit.types import AnyNumpyArray


def scale(array: AnyNumpyArray, perc: int = 2) -> AnyNumpyArray:
    """
    Scale a raster given as a np.ndarray between 0 and 1.

    The min max are computed with percentiles (2 by default), but can be true min/max if :code:`perc=0`.

    .. WARNING::
        If 3D, the raster should be in rasterio's convention: :code:`(count, height, width)`

    Args:
        array (AnyNumpyArray): Matrix to be scaled
        perc (int): Percentile to cut. 0 = min/max, 2 by default

    Returns:
        numpy array: Scaled matrix
    """
    # Convert to float
    f_arr = array.astype(np.float32)

    # Manage NaN values
    masked_idx = None
    if isinstance(array, np.ma.masked_array):
        masked_idx = np.where(array.mask == 1)
        f_arr[masked_idx] = np.nan

    true_shape = f_arr.shape
    if len(true_shape) == 2:
        # Get min max through percentiles
        mins = np.nanpercentile(f_arr, perc)
        maxs = np.nanpercentile(f_arr, 100 - perc)

    elif len(true_shape) == 3:
        count, height, width = true_shape
        f_arr = np.reshape(f_arr, [count, height * width])

        # Get min max through percentiles
        mins = np.nanpercentile(f_arr, perc, axis=1, keepdims=True)
        maxs = np.nanpercentile(f_arr, 100 - perc, axis=1, keepdims=True)
    else:
        raise ValueError("Only 2D or 3D arrays can be rescaled.")

    # Scale
    f_arr = ((f_arr - mins) / (maxs - mins)).astype(
        np.float32
    )  # By default, div returns float64...

    # Clip just in case
    f_arr = f_arr.clip(0, 1)

    # Reshape if 3D arrays
    if len(true_shape) == 3:
        f_arr = np.reshape(f_arr, true_shape)

    # Set back masked values
    if masked_idx:
        f_arr[masked_idx] = array.data[masked_idx]
        f_arr = np.ma.masked_array(f_arr, mask=array.mask, fill_value=array.fill_value)

    return f_arr


def scale_to_uint8(array: AnyNumpyArray, perc: int = 2) -> AnyNumpyArray:
    """
    Rescale array (read as rasterio arrays, which means the bands are the first dimension) to uint8.
    0 will be the nodata.

    Args:
        arr (numpy array): Array to rescale

    Returns:
        numpy array: Rescaled array from 0 to 255 saved in uint8
    """
    # Convert it to uint8
    scaled_arr = (scale(array) * 254 + 1).astype(np.uint8)

    if isinstance(array, np.ma.masked_array):
        scaled_arr = np.where(array.mask, 0, scaled_arr)
        scaled_arr = np.ma.masked_array(scaled_arr, mask=array.mask, fill_value=0)

    return scaled_arr
