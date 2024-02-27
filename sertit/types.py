from pathlib import Path
from typing import Union

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
