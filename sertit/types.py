# Copyright 2026, SERTIT-ICube - France, https://sertit.unistra.fr/
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
"""Types used throughout the library."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from cloudpathlib import CloudPath

try:
    import geopandas as gpd
    from geopandas import GeoDataFrame
    from shapely import MultiPolygon, Polygon
except ModuleNotFoundError:  # pragma: no cover
    gpd = None
    GeoDataFrame = Any
    Polygon = Any
    MultiPolygon = Any

try:
    from rasterio.io import DatasetReader, DatasetWriter
except ModuleNotFoundError:  # pragma: no cover
    DatasetReader = Any
    DatasetWriter = Any

AnyPathType = CloudPath | Path
"""Any Path Type (derived from Pathlib and CloudpathLib)"""

AnyPathStrType = str | CloudPath | Path
"""Same as :code:`AnyPathType` but appended with :code:`str`"""

AnyXrDataStructure = xr.DataArray | xr.Dataset
"""Xarray's DataArray or Dataset"""

AnyNumpyArray = np.ndarray | np.ma.masked_array
"""Numpy array or masked array"""

AnyPolygonType = Polygon | MultiPolygon
"""Shapely Polygon or MultiPolygon"""

AnyRioDatasetType = DatasetReader | DatasetWriter
""" Any Rasterio Dataset Type (both Reader | Writer) """

AnyRasterType = (
    AnyPathStrType | tuple[AnyNumpyArray, dict] | AnyXrDataStructure | AnyRioDatasetType
)

"""
Any object potentially describing a raster:

- its path,
- its ``xarray`` representation (:class:`xarray:xarray.Dataset` or :class:`xarray:xarray.DataArray`),
- its ``rasterio`` representation (``DatasetReader`` or ``DatasetWriter``)
- or its array + metadata (``np.ndarray`` + dict)
"""

AnyVectorType = AnyPathStrType | GeoDataFrame
"""
Any object potentially describing a vector:

- its path,
- its :class:`geopandas:geopandas.GeoDataFrame`
"""


def is_iterable(obj: Any, str_allowed: bool = False):
    """
    Is the object an iterable?

    Useful to replace this kind of code:

        if isinstance(my_items, (list, tuple)):
            do...

    by:

        if is_iterable(my_items):
            do...

    It allows not to forget checking for some iterables you aren't aware of.

    Args:
        obj (Any): Object to check
        str_allowed (bool): If set, strings are considered as iterable. If not, they are not considered as iterable.

    Returns:
        bool: True if the object is iterable

    Example:

        >>> is_iterable((1, 2, 3))
        True
        >>> is_iterable([1, 2, 3])
        True
        >>> is_iterable({1, 2, 3})
        True
        >>> is_iterable(np.array([1, 2, 3]))
        True
        >>> is_iterable("1, 2, 3")
        False
        >>> is_iterable("1, 2, 3", str_allowed=True)
        True
        >>> is_iterable(1)
        False
        >>> is_iterable(Path("path/to/file.tif"))
        False
    """
    if isinstance(obj, str) and not str_allowed:
        return False
    else:
        return isinstance(obj, Iterable)


def make_iterable(
    obj: Any, str_allowed: bool = False, convert_none: bool = False
) -> list | None:
    """
    Convert the object to a list if this object is not iterable

    Useful to replace this kind of code:

        if to_convert is not None and not isinstance(to_convert, (list, tuple)):
            to_convert = [to_convert]

    by:

        to_convert = make_iterable(to_convert)

    or:

        if isinstance(to_convert, (list, tuple)):
            first_item = to_convert[0]
        else:
            first_item = to_convert

    by:

        first_item = make_iterable(to_convert)[0]

    Args:
        obj (Any): Object to check
        str_allowed (bool): If set, strings are considered as iterable. If not, they are not considered as iterable.
        convert_none (bool): If True and :code:`obj` is :code:`None`, keep :code:`None` instead of wrapping it in a list

    Returns:
        list | None: Object as an iterable

    Example:

        >>> make_iterable((1, 2, 3))
        (1, 2, 3)
        >>> make_iterable([1, 2, 3])
        [1, 2, 3]
        >>> make_iterable({1, 2, 3})
        {1, 2, 3}
        >>> make_iterable(np.array([1, 2, 3]))
        array([1, 2, 3])
        >>> make_iterable("1, 2, 3", str_allowed=True)
        '1, 2, 3'
        >>> make_iterable("1, 2, 3", str_allowed=False)
        ['1, 2, 3']
        >>> make_iterable(1)
        [1]
        >>> make_iterable(Path("path/to/file.tif"))
        [PosixPath('path/to/file.tif')]
        >>> make_iterable(None)
        [None]
        >>> make_iterable(None, convert_none=True)
        [None]

    """
    if obj is None and not convert_none:
        return None

    if (convert_none and obj is None) or not is_iterable(obj, str_allowed):
        obj = [obj]

    return obj
