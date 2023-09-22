from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from cloudpathlib import CloudPath
from shapely import MultiPolygon, Polygon

AnyPathType = Union[CloudPath, Path]
AnyPathStrType = Union[str, CloudPath, Path]
AnyXrDataStructure = Union[xr.DataArray, xr.Dataset]
AnyNumpyArray = Union[np.ndarray, np.ma.masked_array]
AnyPolygonType = Union[Polygon, MultiPolygon]
