from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr
from cloudpathlib import CloudPath
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


def is_iterable(obj: Any):
    """
    Is the object an iterable?
    Args:
        obj (Any): Object to check

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
        True
        >>> is_iterable(1)
        False
        >>> is_iterable(AnyPath("1, 2, 3"))
        False
    """
    return isinstance(obj, Iterable)


def make_iterable(obj: Any) -> list:
    """
    Convert the object to a list if this object is not iterable

    Args:
        obj (Any): Object to check

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
        >>> make_interable("1, 2, 3")
        "1, 2, 3"
        >>> make_interable(1)
        [1]
        >>> make_interable(AnyPath("1, 2, 3"))
        [AnyPath("1, 2, 3")] si

    """
    if not is_iterable(obj):
        obj = [obj]

    return obj
