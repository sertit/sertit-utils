from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from cloudpathlib import CloudPath
from rasterio.io import DatasetReader, DatasetWriter
from shapely import MultiPolygon, Polygon

AnyPathType = Union[CloudPath, Path]
"""Any Path Type (derived from Pathlib and CloudpathLib)"""

AnyPathStrType = Union[str, CloudPath, Path]
"""Same as :code:`AnyPathType` but appened with :code:`str`"""

AnyXrDataStructure = Union[xr.DataArray, xr.Dataset]
"""Xarray's DataArray or Dataset"""

AnyNumpyArray = Union[np.ndarray, np.ma.masked_array]
"""Numpy array or masked array"""

AnyPolygonType = Union[Polygon, MultiPolygon]
"""Shapely Polygon or MultiPolygon"""

AnyRioDatasetType = Union[DatasetReader, DatasetWriter]
""" Any Rasterio Dataset Type (both Reader, Writer) """

AnyRasterType = Union[
    AnyPathStrType, tuple[AnyNumpyArray, dict], AnyXrDataStructure, AnyRioDatasetType
]
"""
Any object potentially describing a raster:

- its path,
- its ``xarray`` representation (:class:`xarray:xarray.Dataset` or :class:`xarray:xarray.DataArray`),
- its ``rasterio`` representation (``DatasetReader`` or ``DatasetWriter``)
- or its array + metadata (``np.ndarray`` + dict)
"""

AnyVectorType = Union[AnyPathStrType, gpd.GeoDataFrame]
"""
Any object potentially describing a vector:

- its path,
- its :class:`geopandas:geopandas.GeoDataFrame`
"""


def is_iterable(obj: Any, str_allowed: bool = False):
    """
    Is the object an iterable?

    Useful to replace this kind of code:

        >>> if isinstance(my_items, (list, tuple)):
        >>>     do...

    by:

        >>> if is_iterable(my_items):
        >>>     do...

    It allows not to forget checking for some iterables you aren't aware of.

    Args:
        obj (Any): Object to check
        str_allowed (bool): If set, strings are considered as iterable. If not, they are not considered as iterable.

    Returns:
        bool: True if the iobject is iterable

    Examples:

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
        >>> is_iterable(AnyPath("1, 2, 3"))
        False
    """
    if isinstance(obj, str) and not str_allowed:
        return False
    else:
        return isinstance(obj, Iterable)


def make_iterable(
    obj: Any, str_allowed: bool = False, convert_none: bool = False
) -> list:
    """
    Convert the object to a list if this object is not iterable

    Useful to replace this kind of code:

        >>> if to_convert is not None and not isinstance(to_convert, (list, tuple)):
        >>>    to_convert = [to_convert]

    by:

        >>> to_convert = make_iterable(to_convert)

    or:

        >>> if isinstance(my_items, (list, tuple)):
        >>>    first_item = my_items[0]
        >>> else:
        >>>    first_item = my_items

    by:

        >>> first_item = make_iterable(to_convert)[0]

    Args:
        obj (Any): Object to check
        str_allowed (bool): If set, strings are considered as iterable. If not, they are not considered as iterable.
        convert_none (bool): If true, if obj is None, then it won't be converted into a list. By default, Nones are not converted to list.

    Returns:
        list: Object as an iterable

    Examples:

        >>> make_interable((1, 2, 3))
        (1, 2, 3)
        >>> make_interable([1, 2, 3])
        [1, 2, 3]
        >>> make_interable({1, 2, 3})
        {1, 2, 3}
        >>> make_interable(np.array([1, 2, 3]))
        np.array([1, 2, 3])
        >>> make_interable("1, 2, 3", str_allowed=True)
        "1, 2, 3"
        >>> make_interable("1, 2, 3", str_allowed=False)
        ["1, 2, 3"]
        >>> make_interable(1)
        [1]
        >>> make_interable(AnyPath("1, 2, 3"))
        [AnyPath("1, 2, 3")]
        >>> make_interable(None)
        None
        >>> make_interable(None, convert_none=True)
        [None]

    """
    if (convert_none and obj is None) or not is_iterable(obj, str_allowed):
        obj = [obj]

    return obj
