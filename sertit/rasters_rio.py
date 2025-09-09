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
"""
Raster tools

You can use this only if you have installed sertit[full] or sertit[rasters_rio]
"""

import contextlib
import logging
import os
import tempfile
from functools import wraps
from typing import Any, Callable, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

try:
    import rasterio
    from rasterio import MemoryFile, features, merge, warp
    from rasterio import mask as rio_mask
    from rasterio import shutil as rio_shutil
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window, from_bounds
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "Please install 'rasterio' to use the 'rasters_rio' package."
    ) from ex

from sertit import (
    AnyPath,
    geometry,
    logs,
    misc,
    os_utils,
    path,
    perf,
    strings,
    vectors,
    xml,
)
from sertit.logs import SU_NAME
from sertit.types import AnyNumpyArray, AnyPathStrType, AnyPathType, AnyRasterType

np.seterr(divide="ignore", invalid="ignore")

MAX_CORES = os.cpu_count() - 2
PATH_ARR_DS = AnyRasterType
LOGGER = logging.getLogger(SU_NAME)

DEG_2_RAD = np.pi / 180


# NODATAs
UINT8_NODATA = 255
""" :code:`uint8` nodata """

INT8_NODATA = -128
""" :code:`int8` nodata """

UINT16_NODATA = 65535
""" :code:`uint16` nodata """

FLOAT_NODATA = -9999
""" :code:`float` nodata """


def get_nodata_value_from_dtype(dtype) -> float:
    """
    Get default nodata value from any given dtype.

    Args:
        dtype: Dtype for the wanted nodata. Best if numpy's dtype.

    Returns:
        float: Nodata value

    Examples:
        >>> rasters_rio.get_nodata_value_from_dtype("uint8")
        255

        >>> rasters_rio.get_nodata_value_from_dtype("uint16")
        65535

        >>> rasters_rio.get_nodata_value_from_dtype("int8")
        -128

        >>> rasters_rio.get_nodata_value_from_dtype("float")
        -9999
    """
    # Convert type to numpy if needed
    with contextlib.suppress(AttributeError, TypeError):
        dtype = getattr(np, dtype)

    if dtype == np.uint8:
        nodata = UINT8_NODATA
    elif dtype == np.int8:
        nodata = INT8_NODATA
    elif dtype in [np.uint16, np.uint32, np.int32, np.int64, np.uint64, int, "int"]:
        nodata = UINT16_NODATA
    elif dtype in [np.int16, np.float32, np.float64, float, "float"]:
        nodata = FLOAT_NODATA
    else:
        LOGGER.warning(f"Not recognized dtype: {dtype}. Setting -9999 by default.")
        nodata = FLOAT_NODATA

    return nodata


def get_nodata_value(dtype) -> float:
    """
    .. deprecated:: 1.41.0
       Use :code:`get_nodata_value_from_dtype` instead.

    Get default nodata value:

    Args:
        dtype: Dtype for the wanted nodata. Best if numpy's dtype.

    Returns:
        float: Nodata value
    """
    logs.deprecation_warning(
        "This function is deprecated. Use 'get_nodata_value_from_dtype' instead."
    )
    return get_nodata_value_from_dtype(dtype)


def bigtiff_value(arr: Any) -> str:
    """
    Returns :code:`YES` if array is larger than 4 GB, :code:`IF_NEEDED` otherwise.

    Args:
        arr (Any): Numpy array or xarray

    Returns:
        str: YES or IF_NEEDED

    """
    try:
        itemsize = arr.itemsize
    except AttributeError:
        itemsize = arr.data.itemsize

    # Check size
    bigtiff = "YES" if arr.size * itemsize / 1024 / 1024 / 1024 > 4 else "IF_NEEDED"

    return bigtiff


def any_raster_to_rio_ds(function: Callable) -> Callable:
    """
    Allows a function to ingest AnyRasterType and convert it into a rasterio.DatasetReader:

    - a path (:code:`Path`, :code:`CloudPath` or :code:`str`)
    - a :code:`xarray.Dataset` or a :code:`xarray.DataArray`
    - a :code:`rasterio.DatasetWriter` or :code:`rasterio.DatasetReader`
    - :code:`rasterio` dataset after reading, its array and metadata: (np.ndarray, dict)

    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function

    Example:
        >>> # Create mock function
        >>> @any_raster_to_rio_ds
        >>> def fct(ds):
        >>>     read(ds)
        >>>
        >>> # Test the two ways
        >>> read1 = fct("path/to/raster.tif")
        >>> with rasterio.open("path/to/raster.tif") as ds:
        >>>     read2 = fct(ds)
        >>>
        >>> # Test
        >>> read1 == read2
        True
    """

    @wraps(function)
    def wrapper(any_raster_type: AnyRasterType, *args, **kwargs) -> Any:
        """
        any_raster_to_rio_ds wrapper
        Args:
            any_raster_type (AnyRasterType): Raster path or its dataset
            *args: args
            **kwargs: kwargs

        Returns:
            Any: regular output
        """
        # Input is a path: open it with rasterio
        if path.is_path(any_raster_type):
            # GOTCHA: rasterio and cloudpathlib are not really compatible, so passing a CloudPath directly to rasterio (without turning it into a string) with cache the file!
            # This is really not ideal, so use the string conversion instead
            with rasterio.open(str(any_raster_type)) as ds:
                out = function(ds, *args, **kwargs)
        # Input is a tuple: we consider it's composed of an output of rasterio.read function, a numpy array and a metadata dict
        elif isinstance(any_raster_type, tuple):
            try:
                arr, meta = any_raster_type
                assert isinstance(arr, np.ndarray)
                assert isinstance(meta, dict)
            except (ValueError, AssertionError) as exc:
                raise TypeError(
                    "Input tuple should be composed of a numpy array of your data and the corresponding metadata dictionary, is rasterio's sense."
                ) from exc

            with (
                MemoryFile() as memfile,
                memfile.open(**meta, BIGTIFF=bigtiff_value(arr)) as ds,
            ):
                ds.write(arr)
                out = function(ds, *args, **kwargs)

        # Return given xarray object as is
        elif isinstance(any_raster_type, (xr.DataArray, xr.Dataset)):
            from sertit.rasters import get_nodata_value_from_xr

            nodata = get_nodata_value_from_xr(any_raster_type)

            meta = {
                "driver": "GTiff",
                "dtype": any_raster_type.dtype,
                "nodata": nodata,
                "width": any_raster_type.rio.width,
                "height": any_raster_type.rio.height,
                "count": any_raster_type.rio.count,
                "crs": any_raster_type.rio.crs,
                "transform": any_raster_type.rio.transform(),
            }
            with (
                MemoryFile() as memfile,
                memfile.open(**meta, BIGTIFF=bigtiff_value(any_raster_type)) as ds,
            ):
                if nodata is not None:
                    arr = any_raster_type.fillna(nodata)
                else:
                    arr = any_raster_type
                # /!\ Warning: this triggers a dask compute!
                ds.write(arr.data)
                out = function(ds, *args, **kwargs)

        # Run the fct directly on the input (which should be a rasterio Dataset). If not, this will fail and it's expected.
        else:
            out = function(any_raster_type, *args, **kwargs)

        return out

    return wrapper


def path_arr_dst(function: Callable) -> Callable:
    """
    .. deprecated:: 1.40.0
       Use :py:func:`rasters.any_raster_to_rio_ds` instead.
    """
    logs.deprecation_warning(
        "Deprecated 'path_arr_dst' decorator. Please use 'any_raster_to_rio_ds' instead."
    )
    return any_raster_to_rio_ds(function)


@any_raster_to_rio_ds
def get_new_shape(
    ds: AnyRasterType,
    resolution: Union[tuple, list, float],
    size: Union[tuple, list],
    window: Window = None,
) -> (int, int, bool):
    """
    Get the new shape (height, width) of a resampled raster.

    Size overrides resolution

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided.
        window (Window): Window to be read

    Returns:
        (int, int, bool): Height, width, do resampling

    """
    # By default, do a resampling
    do_resampling = True

    def _get_new_dim(dim: int, res_old: float, res_new: float) -> (int, bool):
        """
        Get the new dimension in pixels
        Args:
            dim (int): Old dimension
            res_old (float): Old resolution
            res_new (float): New resolution

        Returns:
            (int, bool): New dimension, do resampling
        """
        new_dim = int(np.round(dim * res_old / res_new))
        do_res = dim != new_dim
        return new_dim, do_res

    # By default keep original shape
    if window is None:
        new_height = ds.height
        new_width = ds.width
    else:
        new_height = window.height
        new_width = window.width

    # Compute new shape
    if size is not None:
        try:
            if new_height == size[1] and new_width == size[0]:
                do_resampling = False
            else:
                new_height = size[1]
                new_width = size[0]
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"Size should be None or a castable to a list: {size}"
            ) from exc
    elif resolution is not None:
        if isinstance(resolution, (int, float)):
            new_height, do_resampling = _get_new_dim(new_height, ds.res[1], resolution)
            new_width, do_resampling = _get_new_dim(new_width, ds.res[0], resolution)
        else:
            try:
                if len(resolution) != 2:
                    raise ValueError(
                        "We should have a resolution for X and Y dimensions"
                    )

                if resolution[0] is not None:
                    new_width, do_resampling = _get_new_dim(
                        new_width, ds.res[0], resolution[0]
                    )

                if resolution[1] is not None:
                    new_height, do_resampling = _get_new_dim(
                        new_height, ds.res[1], resolution[1]
                    )
            except (TypeError, KeyError) as exc:
                raise ValueError(
                    f"Resolution should be None, 2 floats or a castable to a list: {resolution}"
                ) from exc
    else:
        do_resampling = False

    return new_height, new_width, do_resampling


def update_meta(arr: AnyNumpyArray, meta: dict) -> dict:
    """
    Basic metadata update from a numpy array. Updates everything that we can find in the array:

    - :code:`dtype`: array dtype,
    - :code:`count`: first dimension of the array if the array is in 3D, else 1
    - :code:height`: second dimension of the array
    - :code:`width`: third dimension of the array
    - :code:`nodata`: if a masked array is given, nodata is its fill_value

    .. WARNING::
        The array's shape is interpreted in rasterio's way (count, height, width) !

    Args:
        arr (AnyNumpyArray): Array from which to update the metadata
        meta (dict): Metadata to update

    Returns:
        dict: Update metadata

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> with rasterio.open(raster_path) as ds:
        >>>      meta = ds.meta
        >>>      arr = ds.read()
        >>> meta
        {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': None,
            'width': 300,
            'height': 300,
            'count': 4,
            'crs': CRS.from_epsg(32630),
            'transform': Affine(20.0, 0.0, 630000.0,0.0, -20.0, 4870020.0)
        }
        >>> new_arr = np.ma.masked_array(arr[:, ::2, ::2].astype(np.uint8), fill_value=0)
        >>> new_arr.shape
        (4, 150, 150)
        >>> new_arr.dtype
        dtype('uint8')
        >>> new_arr.fill_value
        0
        >>> update_meta(new_arr, meta)
        {
            'driver': 'GTiff',
            'dtype': dtype('uint8'),
            'nodata': 0,
            'width': 150,
            'height': 150,
            'count': 4,
            'crs': CRS.from_epsg(32630),
            'transform': Affine(20.0, 0.0, 630000.0, 0.0, -20.0, 4870020.0)
        }

    """
    # Manage raster shape (Stored in rasterio's way)
    shape = arr.shape
    count = 1 if len(shape) == 2 else shape[0]
    width = shape[-1]
    height = shape[-2]

    # Update metadata that can be derived from raster
    out_meta = meta.copy()
    out_meta.update(
        {"dtype": arr.dtype, "count": count, "height": height, "width": width}
    )

    # Nodata
    if isinstance(arr, np.ma.masked_array):
        out_meta["nodata"] = arr.fill_value

    return out_meta


def get_nodata_mask(
    array: AnyNumpyArray,
    has_nodata: bool,
    default_nodata: int = 0,
) -> np.ndarray:
    """
    .. deprecated:: 1.36.0
       Use :py:func:`rasters_rio.get_data_mask` instead.
    """
    logs.deprecation_warning("This function is deprecated. Use 'get_data_mask' instead")
    return get_data_mask(array, has_nodata, default_nodata)


def get_data_mask(
    array: AnyNumpyArray,
    has_nodata: bool,
    default_nodata: int = 0,
) -> np.ndarray:
    """
    Get nodata mask from a masked array.

    The nodata may not be set before, then pass a nodata value that will be evaluated on the array.

    .. WARNING::
        Sets 1 where the data is valid and 0 where it is not!

    Example:
        >>> diag_arr = np.diag([1,2,3])
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])
        >>>
        >>> get_data_mask(diag_arr, has_nodata=False)
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]], dtype=uint8)
        >>>
        >>> get_data_mask(diag_arr, has_nodata=False, default_nodata=1)
        array([[0, 1, 1],
               [1, 1, 1],
               [1, 1, 1]], dtype=uint8)

    Args:
        array (np.ma.masked_array): Array to evaluate
        has_nodata (bool): If the array as its nodata specified. If not, using default_nodata.
        default_nodata (int): Default nodata used if the array's nodata is not set

    Returns:
        np.ndarray: Pixelwise nodata array

    """
    # Nodata mask
    if not has_nodata or not isinstance(
        array, np.ma.masked_array
    ):  # Unspecified nodata is set to None by rasterio
        if np.isnan(default_nodata):
            nodata_mask = np.where(np.isnan(array), 0, 1).astype(np.uint8)
        else:
            nodata_mask = np.where(array != default_nodata, 1, 0).astype(np.uint8)
    else:
        nodata_mask = np.where(array.mask, 0, 1).astype(np.uint8)

    return nodata_mask


@any_raster_to_rio_ds
def rasterize(
    ds: AnyRasterType,
    vector: Union[gpd.GeoDataFrame, AnyPathStrType],
    value_field: str = None,
    default_nodata: int = 0,
    default_value: int = 1,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Rasterize a vector into raster format.

    Note that passing `merge_alg = MergeAlg.add` will add the vector values to the given a raster

    See: https://pygis.io/docs/e_raster_rasterize.html

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        vector (Union[gpd.GeoDataFrame, AnyPathStrType]): Vector to be rasterized
        value_field (str): Field of the vector with the values to be burnt on the raster (should be scalars). If let to None, the raster will be binary (`default_nodata`, `default_value`).
        default_nodata (int): Default nodata of the raster (outside the vector in the raster extent)
        default_value (int): Used as value for all geometries, if `value_field` not provided

    Returns:
        np.ma.masked_array, dict: Rasterized vector and its metadata
    """
    if not isinstance(vector, gpd.GeoDataFrame):
        vector = vectors.read(vector, crs=ds.crs)
    else:
        vector = vector.to_crs(crs=ds.crs)

    # Manage vector values
    if value_field:
        geom_value = (
            (geom, value) for geom, value in zip(vector.geometry, vector[value_field])
        )
        dtype = kwargs.pop("dtype", vector[value_field].dtype)
    else:
        geom_value = vector.geometry
        dtype = kwargs.pop("dtype", np.uint8)

    # Manage nodata
    if "nodata" in kwargs:
        nodata = kwargs.pop("nodata")
    elif ds.nodata:
        nodata = ds.nodata
    else:
        nodata = default_nodata

    # Check if the nodata value can be cast into the new dtype
    is_castable = np.can_cast(np.array(nodata, dtype=ds.dtypes[0]), dtype)

    # Floating point values that can be converted to integers are allowed (if they respect min / max for the specific dtype)
    min_max_dtype = np.iinfo(dtype)
    is_valid_int = (
        abs(nodata - int(nodata)) == 0
        and min_max_dtype.min < nodata < min_max_dtype.max
    )

    if not is_castable and not is_valid_int:
        old_nodata = nodata
        nodata = get_nodata_value_from_dtype(dtype)

        # Only throw a warning if the value is really different  (we don't care about 255.0 being replaced by 255)
        if old_nodata - nodata != 0.0:
            LOGGER.warning(
                f"Impossible to cast nodata value ({old_nodata}) into the wanted dtype ({str(dtype)}). "
                f"Default nodata value for this current dtype will be used ({nodata})."
            )

    # Rasterize vector
    mask = features.rasterize(
        geom_value,
        out_shape=(ds.height, ds.width),
        fill=nodata,  # Outside vector
        default_value=default_value,  # Inside vector
        transform=ds.transform,
        dtype=dtype,
        all_touched=kwargs.get("all_touched", True),
        **misc.select_dict(kwargs, ["merge_alg"]),
    )

    meta = ds.meta.copy()
    meta["dtype"] = dtype
    meta["nodata"] = nodata

    return mask, meta


@any_raster_to_rio_ds
def _vectorize(
    ds: AnyRasterType,
    values: Union[None, int, list] = None,
    keep_values: bool = True,
    dissolve: bool = False,
    get_nodata: bool = False,
    default_nodata: int = 0,
) -> gpd.GeoDataFrame:
    """
    Vectorize a raster, both to get classes or nodata.

    If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value,
    with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.

    .. WARNING::
        If :code:`get_nodata` is set to False:
            - Please only use this function on a classified raster.
            - This could take a while as the computing time directly depends on the number of polygons to vectorize.
                Please be careful.
        Else:
            - You will get a classified polygon with data (value=0)/nodata pixels.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        get_nodata (bool): Get nodata vector (raster values are set to 0, nodata values are the other ones)
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Vector
    """
    # Get the shapes
    array = ds.read(masked=True)

    # Manage nodata value
    has_nodata = ds.nodata is not None
    nodata = ds.nodata if has_nodata else default_nodata

    # Manage values
    if values is not None:
        if not isinstance(values, list):
            values = [values]

        # If we want a dissolved vector, just set 1instead of real values
        arr_vals = 1 if dissolve else array
        if keep_values:
            true = arr_vals
            false = nodata
        else:
            true = nodata
            false = arr_vals

        data = np.where(np.isin(array, values), true, false).astype(array.dtype)
    else:
        data = array.data

    # Get nodata mask
    nodata_mask = get_data_mask(data, has_nodata=False, default_nodata=nodata)

    # Get shapes (on array or on mask to get nodata vector)
    shapes = features.shapes(
        nodata_mask if get_nodata else data,
        mask=None if get_nodata else nodata_mask,
        transform=ds.transform,
    )

    # Convert to geodataframe (with valid geometries)
    gdf = vectors.shapes_to_gdf(shapes, ds.crs)

    # Dissolve if needed
    if dissolve:
        gdf = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs).dissolve()

    return gdf


@any_raster_to_rio_ds
def vectorize(
    ds: AnyRasterType,
    values: Union[None, int, list] = None,
    keep_values: bool = True,
    dissolve: bool = False,
    default_nodata: int = 0,
) -> gpd.GeoDataFrame:
    """
    Vectorize a raster to get the class vectors.

    If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value,
    with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.

    .. WARNING::
        - Please only use this function on a classified raster.
        - This could take a while as the computing time directly depends on the number of polygons to vectorize.
            Please be careful.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        default_nodata (int): Default values for nodata in case of non-existing in file

    Returns:
        gpd.GeoDataFrame: Classes Vector

    Example:
        >>> raster_path = "path/to/raster.tif"  # Classified raster, with no data set to 255
        >>> vec1 = vectorize(raster_path)
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     vec2 = vectorize(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> vec1 == vec2
        True
    """
    return _vectorize(
        ds,
        values=values,
        get_nodata=False,
        keep_values=keep_values,
        dissolve=dissolve,
        default_nodata=default_nodata,
    )


@any_raster_to_rio_ds
def get_valid_vector(ds: AnyRasterType, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get the valid data of a raster as a vector.

    Pay attention that every nodata pixel will appear too.
    If you want only the footprint of the raster, please use :code:`get_footprint`.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        default_nodata (int): Default values for nodata in case of non-existing in file

    Returns:
        gpd.GeoDataFrame: Nodata Vector

    Example:
        >>> raster_path = "path/to/raster.tif"  # Classified raster, with no data set to 255
        >>> nodata1 = get_nodata_vec(raster_path)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     nodata2 = get_nodata_vec(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> nodata1 == nodata2
        True

    """
    nodata = _vectorize(ds, values=None, get_nodata=True, default_nodata=default_nodata)
    return nodata[
        nodata.raster_val != 0
    ]  # 0 is the values of not nodata put there by rasterio


@any_raster_to_rio_ds
def get_nodata_vector(ds: AnyRasterType, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get the nodata vector of a raster as a vector.

    Pay attention that every nodata pixel will appear too.
    If you want only the footprint of the raster, please use :code:`get_footprint`.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        default_nodata (int): Default values for nodata in case of non-existing in file

    Returns:
        gpd.GeoDataFrame: Nodata Vector

    Example:
        >>> raster_path = "path/to/raster.tif"  # Classified raster, with no data set to 255
        >>> nodata1 = get_nodata_vec(raster_path)
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     nodata2 = get_nodata_vec(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> nodata1 == nodata2
        True

    """
    nodata = _vectorize(ds, values=None, get_nodata=True, default_nodata=default_nodata)
    return nodata[
        nodata.raster_val == 0
    ]  # 0 is the values of not nodata put there by rasterio


@any_raster_to_rio_ds
def _mask(
    ds: AnyRasterType,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    do_crop: bool = False,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Overload of rasterio mask function in order to create a masked_array.

    The :code:`mask` function docs can be seen `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html>`_.

    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted.
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        do_crop (bool): Whether to crop the raster to the extent of the shapes. Default is False.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, dict): Cropped array as a masked array and its metadata
    """
    if isinstance(shapes, (gpd.GeoDataFrame, gpd.GeoSeries)):
        shapes = shapes.to_crs(ds.crs).geometry
    elif not isinstance(shapes, list):
        shapes = [shapes]

    # Set nodata
    if not nodata:
        nodata = ds.nodata if ds.nodata else 0

    # Crop dataset
    possible_kwargs = ["all_touched", "invert", "filled", "pad", "pad_width", "indexes"]
    msk, trf = rio_mask.mask(
        ds,
        shapes,
        nodata=nodata,
        crop=do_crop,
        **misc.select_dict(kwargs, possible_kwargs),
    )

    # Create masked array
    nodata_mask = np.where(msk == nodata, 1, 0).astype(np.uint8)
    mask_array = np.ma.masked_array(msk, nodata_mask, fill_value=nodata)

    # Update meta
    out_meta = update_meta(mask_array, ds.meta)
    out_meta["transform"] = trf

    return mask_array, out_meta


@any_raster_to_rio_ds
def mask(
    ds: AnyRasterType,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Masking a dataset:
    setting nodata outside of the given shapes, but without cropping the raster to the shapes extent.

    Overload of rasterio mask function in order to create a masked_array.

    The :code:`mask` function docs can be seen `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html>`_.
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted.
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, dict): Masked array as a masked array and its metadata

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> shape_path = "path/to/shapes.geojson"  # Any vector that geopandas can read
        >>> shapes = gpd.read_file(shape_path)
        >>> masked_raster1, meta1 = mask(raster_path, shapes)
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     masked_raster2, meta2 = mask(ds, shapes)
        >>>
        >>> # Assert those two approaches give the same result
        >>> masked_raster1 == masked_raster2
        True
        >>> meta1 == meta2
        True
    """
    return _mask(ds, shapes=shapes, nodata=nodata, do_crop=False, **kwargs)


@any_raster_to_rio_ds
def crop(
    ds: AnyRasterType,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Cropping a dataset:
    setting nodata outside of the given shapes AND cropping the raster to the shapes extent.

    **HOW:**

    Overload of rasterio mask function in order to create a masked_array.

    The :code:`mask` function docs can be seen `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html>`_.
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted.
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, dict): Cropped array as a masked array and its metadata

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> shape_path = "path/to/shapes.geojson"  # Any vector that geopandas can read
        >>> shapes = gpd.read_file(shape_path)
        >>> cropped_raster1, meta1 = crop(raster_path, shapes)
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     cropped_raster2, meta2 = crop(ds, shapes)
        >>>
        >>> # Assert those two approaches give the same result
        >>> cropped_raster1 == cropped_raster2
        True
        >>> meta1 == meta2
        True
    """
    return _mask(ds, shapes=shapes, nodata=nodata, do_crop=True, **kwargs)


@any_raster_to_rio_ds
def get_window(ds: AnyRasterType, window: Any):
    """
    Get a window from any type of input

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        window (Any): Anything that can be returned as a window. In case of iterable, assumption is made it's geographic bounds. For pixel, please provide a Window directly.

    Returns:
        Window: Rasterio window

    """
    if window is not None and not isinstance(window, Window):
        if isinstance(window, gpd.GeoDataFrame):
            bounds = window.to_crs(ds.crs).total_bounds
        elif path.is_path(window):
            bounds = vectors.read(window).to_crs(ds.crs).total_bounds
        else:
            bounds = window

        try:
            window = from_bounds(*bounds, ds.transform)
        except Exception as exc:
            raise TypeError(
                "Window should either be a GeoDataFrame, tuple, list, Window, readable as a vector or set to None"
            ) from exc

    # Use rioxarray way to convert window to integer
    (row_start, row_stop), (col_start, col_stop) = window.toranges()
    row_start = 0 if row_start < 0 else np.floor(row_start)
    row_stop = 0 if row_stop < 0 else np.ceil(row_stop)
    col_start = 0 if col_start < 0 else np.floor(col_start)
    col_stop = 0 if col_stop < 0 else np.ceil(col_stop)
    row_slice = slice(int(row_start), int(row_stop))
    col_slice = slice(int(col_start), int(col_stop))

    window = Window.from_slices(
        rows=row_slice,
        cols=col_slice,
        width=int(window.width),
        height=int(window.height),
    )
    return window


@any_raster_to_rio_ds
def read(
    ds: AnyRasterType,
    resolution: Union[tuple, list, float] = None,
    size: Union[tuple, list] = None,
    window: Any = None,
    resampling: Resampling = Resampling.nearest,
    masked: bool = True,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Read a raster dataset from a :code:`rasterio.Dataset` or a path.

    The resolution can be provided (in dataset unit) as:

    - a tuple or a list of (X, Y) resolutions
    - a float, in which case X resolution = Y resolution
    - None, in which case the dataset resolution will be used

    Tip:
    Use index with a list of one element to keep a 3D array

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided.
        window (Any): Anything that can be returned as a window (i.e. path, gpd.GeoPandas, Iterable, rasterio.Window...).
            In case of an iterable, assumption is made it corresponds to geographic bounds.
            For pixel, please provide a rasterio.Window directly.
        resampling (Resampling): Resampling method (nearest by default)
        masked (bool): Get a masked array, :code:`True` by default (whereas it is False by default in rasterio)
        **kwargs: Other ds.read() arguments such as indexes.

    Returns:
        np.ma.masked_array, dict: Masked array corresponding to the raster data and its metadata

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> raster1, meta1 = read(raster_path)
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>    raster2, meta2 = read(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> raster1 == raster2
        True
        >>> meta1 == meta2
        True

    """
    dst_transform = None
    if window is not None:
        window = get_window(ds, window)
        dst_transform = ds.window_transform(window)

    # Get new height and width
    new_height, new_width, _ = get_new_shape(ds, resolution, size, window)

    # Manage out_shape
    if "indexes" in kwargs:
        if isinstance(kwargs["indexes"], int):
            out_shape = (new_height, new_width)
            new_count = 1
        else:
            new_count = len(kwargs["indexes"])
            out_shape = (new_count, new_height, new_width)
    else:
        new_count = ds.count
        out_shape = (new_count, new_height, new_width)

    # Read data
    array = ds.read(
        out_shape=out_shape,
        resampling=resampling,
        masked=masked,
        window=window,
        **misc.select_dict(kwargs, ["indexes", "out_dtype", "boundless", "fill_value"]),
    )

    # Get destination transform
    if dst_transform is None:
        dst_transform = ds.transform * ds.transform.scale(
            (ds.width / new_width), (ds.height / new_height)
        )
    if window is not None and resolution:
        dst_transform = dst_transform * dst_transform.scale(
            (window.width / new_width), (window.height / new_height)
        )

    # Update meta
    dst_meta = ds.meta.copy()
    dst_meta.update(
        {
            "height": new_height,
            "width": new_width,
            "count": new_count,
            "transform": dst_transform,
            "dtype": array.dtype,
            "nodata": ds.nodata,
        }
    )

    return array, dst_meta


def write(
    raster: AnyRasterType,
    meta: dict,
    output_path: AnyPathStrType = None,
    tags: dict = None,
    **kwargs,
) -> None:
    """
    Write raster to disk (encapsulation of rasterio's function)

    Metadata will be copied and updated with raster's information (i.e. width, height, count, type...)
    The driver is GTiff by default, and no nodata value is provided.
    The file will be compressed if the raster is a mask (saved as uint8)

    Args:
        raster (AnyRasterType): Raster to save on disk
        output_path (AnyPathStrType): Path where to save it (directories should be existing)
        tags (dict): Tags to write to the GeoTiff
        **kwargs: Overloading metadata, ie :code:`nodata=255`

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> raster_out = "path/to/out.tif"
        >>>
        >>> # Read raster
        >>> raster, meta = read(raster_path)
        >>>
        >>> # Rewrite it on disk
        >>> write(raster, meta, raster_out)
    """
    if output_path is None:
        logs.deprecation_warning(
            "'path' is deprecated in 'rasters_rio.write'. Use 'output_path' instead."
        )
        output_path = kwargs.pop("path")

    raster_out = raster.copy()

    # Prune empty kwargs to avoid throwing GDAL warnings/errors
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Manage raster type (impossible to write boolean arrays)
    if raster_out.dtype == bool:
        raster_out = raster_out.astype(np.uint8)

    # Update metadata
    out_meta = meta.copy()

    # Update raster to be sure to write down correct nodata pixels
    nodata = kwargs.get("nodata")
    if nodata is None:
        nodata = meta.get("nodata")
        if nodata is None and isinstance(raster_out, np.ma.masked_array):
            nodata = raster_out.fill_value

    # TODO: change this with rasterio 1.3.0 (masked option in write)
    if isinstance(raster_out, np.ma.masked_array):
        raster_out[raster_out.mask] = nodata

    out_meta["nodata"] = nodata

    # Force compression and driver (but can be overwritten by kwargs)
    out_meta["driver"] = kwargs.get("driver", "GTiff")

    # Compress to LZW by default
    out_meta["compress"] = kwargs.get("compress", "lzw")

    if (
        out_meta["compress"].lower() in ["lzw", "deflate", "zstd"]
        and "predictor" not in out_meta  # noqa: W503
    ):
        if out_meta["dtype"] in [np.float32, np.float64, float]:
            out_meta["predictor"] = "3"
        else:
            out_meta["predictor"] = "2"

    # Bigtiff if needed (more than 4Go)
    out_meta["BIGTIFF"] = bigtiff_value(raster_out)

    # Set more threads
    out_meta["NUM_THREADS"] = perf.get_max_cores()

    # Update metadata with array data
    out_meta = update_meta(raster_out, out_meta)

    # Update metadata with additional params
    for key, val in kwargs.items():
        out_meta[key] = val

    # Manage raster shape
    if len(raster_out.shape) == 2:
        raster_out = np.expand_dims(raster_out, axis=0)

    # Write product
    with rasterio.open(str(output_path), "w", **out_meta) as ds:
        ds.write(raster_out)
        if tags is not None:
            ds.update_tags(**tags)


def collocate(
    reference_meta: dict,
    other_arr: AnyNumpyArray,
    other_meta: dict,
    resampling: Resampling = Resampling.nearest,
) -> (AnyNumpyArray, dict):
    """
    Collocate two georeferenced arrays:
    forces the *other* raster to be exactly georeferenced onto the *reference* raster by reprojection.

    Args:
        reference_meta (dict): Reference metadata
        other_arr (np.ma.masked_array): Other array to be collocated
        other_meta (dict): Other metadata
        resampling (Resampling): Resampling method

    Returns:
        np.ma.masked_array, dict: Collocated array and its metadata

    Example:
        >>> reference_path = "path/to/reference.tif"
        >>> other_path = "path/to/other.tif"
        >>> col_path = "path/to/collocated.tif"
        >>>
        >>> # Just open the master data
        >>> with rasterio.open(reference_path) as reference_dst:
        >>>     # Read other
        >>>     other, other_meta = read(other_path)
        >>>
        >>>     # Collocate the other to the reference
        >>>     col_arr, col_meta = collocate(reference_dst.meta,
        >>>                                   other,
        >>>                                   other_meta,
        >>>                                   Resampling.bilinear)
        >>>
        >>> # Write it
        >>> write(col_arr, col_path, col_meta)

    """
    collocated_arr = np.zeros(
        (reference_meta["count"], reference_meta["height"], reference_meta["width"]),
        dtype=reference_meta["dtype"],
    )
    warp.reproject(
        source=other_arr,
        destination=collocated_arr,
        src_transform=other_meta["transform"],
        src_crs=other_meta["crs"],
        dst_transform=reference_meta["transform"],
        dst_crs=reference_meta["crs"],
        src_nodata=other_meta["nodata"],
        dst_nodata=other_meta["nodata"],
        resampling=resampling,
        num_threads=perf.get_max_cores(),
    )

    meta = reference_meta.copy()
    meta.update(
        {
            "dtype": other_meta["dtype"],
            "driver": other_meta["driver"],
            "nodata": other_meta["nodata"],
        }
    )

    if isinstance(other_arr, np.ma.masked_array):
        collocated_arr = np.ma.masked_array(
            collocated_arr, other_arr.mask, fill_value=other_meta["nodata"]
        )

    return collocated_arr, meta


def sieve(
    array: AnyNumpyArray,
    out_meta: dict,
    sieve_thresh: int,
    connectivity: int = 4,
) -> (AnyNumpyArray, dict):
    """
    Sieving, overloads rasterio function with raster shaped like (1, h, w).

    Forces the output to :code:`np.uint8` (as only classified rasters should be sieved)

    Args:
        array (AnyNumpyArray): Array to sieve
        out_meta (dict): Metadata to update
        sieve_thresh (int): Sieving threshold in pixels
        connectivity (int): Connectivity, either 4 or 8

    Returns:
        (AnyNumpyArray, dict): Sieved array and updated meta

    Example:
        >>> raster_path = "path/to/raster.tif"  # classified raster
        >>>
        >>> # Read raster
        >>> raster, meta = read(raster_path)
        >>>
        >>> # Rewrite it
        >>> sieved, sieved_meta = sieve(raster, meta, sieve_thresh=20)
        >>>
        >>> # Write it
        >>> raster_out = "path/to/raster_sieved.tif"
        >>> write(sieved, raster_out, sieved_meta)
    """
    assert connectivity in [4, 8]

    # Read extraction array
    expand = False
    if len(array.shape) == 3 and array.shape[0] == 1:
        array = np.squeeze(array)  # Use this trick to make the sieve work
        expand = True

    # Get nodata mask
    msk = ~array.mask if isinstance(array, np.ma.masked_array) else np.ones_like(array)

    if expand:
        msk = np.squeeze(msk)

    # Convert to np.uint8 if needed
    dtype = np.uint8
    meta = out_meta.copy()
    if meta["dtype"] != dtype:
        array = array.astype(dtype)
        meta["dtype"] = dtype

    # Sieve
    result_array = np.empty(array.shape, dtype=array.dtype)
    features.sieve(
        array, size=sieve_thresh, out=result_array, connectivity=connectivity, mask=msk
    )

    # Use this trick to get the array back to 'normal'
    if expand:
        result_array = np.expand_dims(result_array, axis=0)

    return result_array, meta


def get_dim_img_path(
    dim_path: AnyPathStrType, img_name: str = "*", get_list: bool = False
) -> Union[list, AnyPathType]:
    """
    Get the image path (:code:`.img`) from a :code:`BEAM-DIMAP` data.

    A :code:`BEAM-DIMAP` file cannot be opened by rasterio, although its :code:`.img` file can.

    Args:
        dim_path (AnyPathStrType): DIM path (.dim or .data)
        img_name (str): .img file name (or regex), in case there are multiple .img files (i.e. for S3 data)

    Returns:
        AnyPathType: .img file

    Example:
        >>> dim_path = "path/to/dimap.dim"  # BEAM-DIMAP image
        >>> img_path = get_dim_img_path(dim_path)
        img_path = "path/to/dimap.dim/dimap.img"
        >>>
        >>> # Read raster
        >>> raster, meta = read(img_path)
    """
    dim_path = AnyPath(dim_path)
    if dim_path.suffix == ".dim":
        dim_path = dim_path.with_suffix(".data")

    assert dim_path.suffix == ".data" and dim_path.is_dir()

    return path.get_file_in_dir(
        dim_path, img_name, extension="img", exact_name=True, get_list=get_list
    )


@any_raster_to_rio_ds
def get_extent(ds: AnyRasterType) -> gpd.GeoDataFrame:
    """
    Get the extent of a raster as a :code:`geopandas.Geodataframe`.

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata

    Returns:
        gpd.GeoDataFrame: Extent as a  :code:`geopandas.Geodataframe`

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>>
        >>> extent1 = get_extent(raster_path)
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     extent2 = get_extent(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> extent1 == extent2
        True
    """
    return vectors.get_geodf(geom=[*ds.bounds], crs=ds.crs)


@any_raster_to_rio_ds
def get_footprint(ds: AnyRasterType) -> gpd.GeoDataFrame:
    """
    Get real footprint of the product (without nodata, in *french == emprise utile*)

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata

    Returns:
        gpd.GeoDataFrame: Footprint as a GeoDataFrame

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>>
        >>> footprint1 = get_footprint(raster_path)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     footprint2 = get_footprint(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> footprint1 == footprint2
    """
    footprint = get_valid_vector(ds)

    return geometry.get_wider_exterior(footprint)


def merge_vrt(
    paths: list,
    merged_path: AnyPathStrType,
    abs_path: bool = False,
    **kwargs,
) -> None:
    """
    Merge rasters as a VRT. Uses :code:`gdalbuildvrt`.

    See here: https://gdal.org/programs/gdalbuildvrt.html

    Creates VRT with relative paths!

    This function handles files of different projection by creating intermediate VRT used for warping.
    All VRTs will be written with relative paths.

    Args:
        paths (list): Path of the rasters to be merged with the same CRS)
        merged_path (AnyPathStrType): Path to the merged raster
        abs_path (bool): VRT with absolute paths. If not, VRT with relative paths (default)
        kwargs: Other :code:`gdalbuildvrt` arguments

    Example:
        >>> paths_utm32630 = ["path/to/raster1.tif", "path/to/raster2.tif", "path/to/raster3.tif"]
        >>> paths_utm32631 = ["path/to/raster4.tif", "path/to/raster5.tif"]
        >>>
        >>> mosaic_32630 = "path/to/mosaic_32630.vrt"
        >>> mosaic_32631 = "path/to/mosaic_32631.vrt"
        >>>
        >>> # Create mosaic, one by CRS !
        >>> merge_vrt(paths_utm32630, mosaic_32630)
        >>> merge_vrt(paths_utm32631, mosaic_32631, {"-srcnodata":255, "-vrtnodata":0})
    """
    # Copy crs_paths in order not to modify it in place (replacing str by Paths for example)
    crs_paths_cp = paths.copy()

    # Manage cloud paths (gdalbuildvrt needs url or true filepaths)
    merged_path = AnyPath(merged_path)

    first_crs = kwargs.get("crs")
    for i, crs_path in enumerate(crs_paths_cp):
        crs_path = AnyPath(crs_path)
        # Download file if VRT is needed
        if path.is_cloud_path(crs_path):
            crs_path = crs_path.download_to(merged_path.parent)

        with rasterio.open(str(crs_path)) as src:
            if first_crs is None:
                first_crs = src.crs
            else:
                # Reproject bands if needed
                if first_crs != src.crs:
                    crs_epsg = first_crs.to_epsg()
                    with WarpedVRT(src, **{"crs": first_crs.to_epsg()}) as vrt:
                        # At this point 'vrt' is a full dataset with dimensions,
                        # CRS, and spatial extent matching 'vrt_options'.
                        new_crs_name = path.get_filename(crs_path) + f"_{crs_epsg}.vrt"
                        new_crs_path = os.path.join(merged_path.parent, new_crs_name)
                        rio_shutil.copy(vrt, new_crs_path, driver="vrt")

                        try:
                            # Set to relative
                            # This is clearly a hack, but GDAL doesn't handle any copy with relative path
                            # See https://github.com/rasterio/rasterio/discussions/2720
                            vrt_root = xml.read(new_crs_path)
                            xml.update_attrib(
                                vrt_root, "SourceDataset", "relativeToVRT", "1"
                            )
                            xml.update_txt(
                                vrt_root,
                                "SourceDataset",
                                crs_path.relative_to(merged_path.parent),
                            )
                            xml.write(vrt_root, new_crs_path)
                        except ValueError as ex:
                            LOGGER.warning(f"Your VRT will be absolute as {str(ex)}")

                        # Add in place
                        crs_path = new_crs_path

        crs_paths_cp[i] = crs_path

    # Create relative paths
    vrt_root = os.path.dirname(merged_path)
    try:
        if abs_path:
            paths = [strings.to_cmd_string(path.to_abspath(p)) for p in crs_paths_cp]
            merged_path = strings.to_cmd_string(merged_path.resolve())
        else:
            paths = [
                strings.to_cmd_string(path.real_rel_path(p, vrt_root))
                for p in crs_paths_cp
            ]
            merged_path = strings.to_cmd_string(
                path.real_rel_path(merged_path, vrt_root)
            )

    except ValueError:
        # ValueError when crs_merged_path and crs_paths are not on the same disk
        paths = [strings.to_cmd_string(str(p)) for p in crs_paths_cp]
        merged_path = strings.to_cmd_string(str(merged_path))

    # Run cmd
    gdal_build_vrt_exe = os_utils.gdalbuildvrt_exe()
    arg_list = [val for item in kwargs.items() for val in item]
    try:
        vrt_cmd = [gdal_build_vrt_exe, merged_path, *paths, *arg_list]
        misc.run_cli(vrt_cmd, cwd=vrt_root)

    except RuntimeError:
        # Manage too long command line
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = os.path.join(tmp_dir, "list.txt")
            with open(tmp_file, "w+") as f:
                for p in paths:
                    p = p.replace('"', "")
                    f.write(f"{p}\n")

            vrt_cmd = [
                gdal_build_vrt_exe,
                "-input_file_list",
                tmp_file,
                merged_path,
                *arg_list,
            ]

            misc.run_cli(vrt_cmd, cwd=vrt_root)


def merge_gtiff(paths: list, merged_path: AnyPathStrType, **kwargs) -> None:
    """
    Merge rasters as a GeoTiff.

    This function handles files of different projection by creating intermediate VRT used for warping.

    Using :code:`rasterio.merge` with default arguments behind the hood so you can provide any valid argument into the kwargs.
    For example, to modify the merging method, you can pass :code:`method="max"`

    Args:
        paths (list): Path of the rasters to be merged with the same CRS)
        merged_path (AnyPathStrType): Path to the merged raster
        kwargs: Other rasterio.merge arguments
            More info `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html#rasterio.merge.merge>`_

    Example:
        >>> paths_utm32630 = ["path/to/raster1.tif", "path/to/raster2.tif", "path/to/raster3.tif"]
        >>> paths_utm32631 = ["path/to/raster4.tif", "path/to/raster5.tif"]
        >>>
        >>> mosaic_32630 = "path/to/mosaic_32630.tif"
        >>> mosaic_32631 = "path/to/mosaic_32631.tif"
        >>>
        >>> # Create mosaic, one by CRS !
        >>> merge_gtiff(paths_utm32630, mosaic_32630)
        >>> merge_gtiff(paths_utm32631, mosaic_32631)
    """
    # Open datasets for merging
    tmp_dir = None
    crs_datasets = []
    try:
        first_crs = kwargs.get("crs")
        for tile_path in paths:
            src = rasterio.open(str(tile_path))
            if first_crs is None:
                first_crs = src.crs
            else:
                # Reproject bands if needed
                if first_crs != src.crs:
                    tmp_dir = tempfile.TemporaryDirectory()
                    with WarpedVRT(src, **{"crs": first_crs.to_epsg()}) as vrt:
                        # At this point 'vrt' is a full dataset with dimensions,
                        # CRS, and spatial extent matching 'vrt_options'.
                        tile_path = os.path.join(
                            tmp_dir.name, path.get_filename(tile_path) + ".vrt"
                        )
                        rio_shutil.copy(vrt, tile_path, driver="vrt")

                    src.close()
                    src = rasterio.open(str(tile_path))

            crs_datasets.append(src)

        # Merge all datasets
        merge_args = [
            "bounds",
            "res",
            "precision",
            "indexes",
            "output_count",
            "resampling",
            "method",
            "target_aligned_pixels",
            "mem_limit",
            "use_highest_res",
            "masked",
            "dst_path",
            "dst_kwds",
        ]
        merge_kwargs = misc.select_dict(kwargs, merge_args + ["nodata", "dtype"])
        write_kwargs = misc.prune_dict(kwargs, merge_args)

        merged_array, merged_transform = merge.merge(crs_datasets, **merge_kwargs)
        merged_meta = crs_datasets[0].meta.copy()
        merged_meta.update(
            {
                "driver": "GTiff",
                "height": merged_array.shape[1],
                "width": merged_array.shape[2],
                "transform": merged_transform,
            }
        )
    finally:
        # Close all datasets
        src = None
        for dataset in crs_datasets:
            dataset.close()

        if tmp_dir is not None:
            tmp_dir.cleanup()

    # Save merge datasets
    write(merged_array, merged_meta, merged_path, **write_kwargs)


def unpackbits(array: np.ndarray, nof_bits: int) -> np.ndarray:
    """
    Function found
    `here <https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types>`_

    Args:
        array (np.ndarray): Array to unpack
        nof_bits (int): Number of bits to unpack

    Returns:
        np.ndarray: Unpacked array

    Example:
        >>> bit_array = np.random.randint(5, size=[3,3])
        array([[1, 1, 3],
               [4, 2, 0],
               [4, 3, 2]], dtype=uint8)
        >>>
        >>> # Unpack 8 bits (8*1, as itemsize of uint8 is 1)
        >>> unpackbits(bit_array, 8)
        array([[[1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0]]], dtype=uint8)
    """
    xshape = list(array.shape)
    array = array.reshape([-1, 1])
    msk = 2 ** np.arange(nof_bits, dtype=array.dtype).reshape([1, nof_bits])
    uint8_packed = (array & msk).astype(bool).astype(np.uint8)
    try:
        unpacked = uint8_packed.reshape(xshape + [nof_bits])
    except IndexError:
        # Workaround for weird bug in reshape with dask
        unpacked = uint8_packed.compute().reshape(xshape + [nof_bits])

    return unpacked


def read_bit_array(
    bit_mask: np.ndarray, bit_id: Union[list, int]
) -> Union[np.ndarray, list]:
    """
    Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number bit_id)

    Args:
        bit_mask (np.ndarray): Bit array to read
        bit_id (int): Bit ID of the slice to be read
          Example: read the bit 0 of the mask as a cloud mask (Theia)

    Returns:
        Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given

    Example:
        >>> bit_array = np.random.randint(5, size=[3,3])
        array([[1, 1, 3],
               [4, 2, 0],
               [4, 3, 2]], dtype=uint8)

        >>> # Get the 2nd bit array
        >>> read_bit_array(bit_array, 2)
        array([[0, 0, 0],
               [1, 0, 0],
               [1, 0, 0]], dtype=uint8)
    """
    # Get the number of bits
    nof_bits = 8 * bit_mask.dtype.itemsize

    # Read cloud mask as bits
    msk = unpackbits(bit_mask, nof_bits)

    # Only keep the bit number bit_id and reshape the vector
    if isinstance(bit_id, list):
        bit_arr = [msk[..., bid] for bid in bit_id]
    else:
        bit_arr = msk[..., bit_id]

    return bit_arr


@any_raster_to_rio_ds
def hillshade(
    ds: AnyRasterType, azimuth: float = 315, zenith: float = 45
) -> (np.ma.masked_array, dict):
    """
     Compute the hillshade of a DEM from an azimuth and zenith angle (in degrees).

     Goal: replace `gdaldem CLI <https://gdal.org/programs/gdaldem.html>`_

     NB: altitude = 90 - zenith

     .. WARNING::

         - It uses a 2nd order gradient instead of Horn's or Zevenbergen & Thorne's formula
         - z_factor is fixed to 1.0
         - scale managed by ds resolution

    `Reference <https://github.com/nasa/World-Wind-Java/blob/7c9886ab67ac03d53bdb04f161b9605d3f3dd810/GDAL/GDAL-1.7.2/apps/gdaldem.cpp#L349>`_

     Args:
         ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
         azimuth (float): Azimuth of the light, in degrees. 0 if it comes from the top of the raster, 90 from the east, ...
         zenith (float): Zenith angle in degrees

     Returns:
         (np.ma.masked_array, dict): Hillshade and its metadata
    """
    array = ds.read(masked=True)

    # Squeeze if needed
    expand = False
    if len(array.shape) == 3 and array.shape[0] == 1:
        array = np.squeeze(array)  # Use this trick to make the sieve work
        expand = True

    # Compute angles
    az_rad = azimuth * DEG_2_RAD
    alt_rad = (90 - zenith) * DEG_2_RAD

    # Compute slope and aspect
    dx, dy = np.gradient(np.where(array.mask, 0.0, array.data), *ds.res)
    x2_y2 = dx**2 + dy**2
    aspect = np.arctan2(dx, dy)

    # Compute hillshade (GDAL algo)
    hshade = (
        np.sin(alt_rad) + np.cos(alt_rad) * np.sqrt(x2_y2) * np.sin(aspect - az_rad)
    ) / np.sqrt(1 + x2_y2)
    hshade = np.where(hshade <= 0, 1.0, 254.0 * hshade + 1)

    # Use this trick to get the array back to 'normal'
    if expand:
        hshade = np.expand_dims(hshade, axis=0)

    # Convert to masked array
    hillshade_msk = np.ma.masked_array(hshade, array.mask, fill_value=ds.nodata)

    # Meta
    meta = update_meta(hillshade_msk, ds.meta)

    return hillshade_msk, meta


@any_raster_to_rio_ds
def slope(
    ds: AnyRasterType,
    in_pct: bool = False,
    in_rad: bool = False,
) -> (np.ma.masked_array, dict):
    """
     Compute the slope of a DEM (in degrees).

     Goal: replace `gdaldem CLI <https://gdal.org/programs/gdaldem.html>`_

     .. WARNING::

         - It uses a 2nd order gradient instead of Horn's or Zevenbergen & Thorne's formula
         - z_factor is fixed to 1.0
         - scale managed by ds resolution

    `Reference <https://git.earthdata.nasa.gov/projects/GEE/repos/gdal-enhancements-for-esdis/browse/gdal-1.10.0/apps/gdaldem.cpp>`_

     Args:
         ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
         in_pct (bool): Outputs slope in percents
         in_rad (bool): Outputs slope in radians. Not taken into account if :code:`in_pct == True`

     Returns:
         (np.ma.masked_array, dict): Slope and its metadata
    """
    array = ds.read(masked=True)

    # Squeeze if needed
    expand = False
    if len(array.shape) == 3 and array.shape[0] == 1:
        array = np.squeeze(array)  # Use this trick to make the sieve work
        expand = True

    # Compute slope (on unmasked data)
    dx, dy = np.gradient(np.where(array.mask, 0.0, array.data), *ds.res)
    x2_y2 = dx**2 + dy**2

    if in_pct:
        slp = 100 * (np.sqrt(x2_y2))
    else:
        slp = np.arctan(np.sqrt(x2_y2))

        # Convert into degrees
        if not in_rad:
            slp = slp / DEG_2_RAD

    # Use this trick to get the array back to 'normal'
    if expand:
        slp = np.expand_dims(slp, axis=0)

    # Convert to masked array
    slp_msk = np.ma.masked_array(slp, array.mask, fill_value=ds.nodata)

    # Meta
    meta = update_meta(slp_msk, ds.meta)

    return slp_msk, meta


def reproject_match(
    dst_meta: dict,
    src_arr: AnyNumpyArray,
    src_meta: dict,
    resampling: Resampling = Resampling.nearest,
    **kwargs,
) -> (AnyNumpyArray, dict):
    """
    Reproject a raster to match the resolution, projection, and region of another raster.

    Matching rioxarray reproject_match.

    Args:
        dst_meta (dict): Destination metadata
        src_arr (AnyNumpyArray): Source raster's array
        src_meta (dict): Source metadata
        resampling (Resampling): Resampling method
        **kwargs: Passing other kwargs to `calculate_default_transform` and `reproject`

    Returns:
        AnyNumpyArray, dict: Reprojected array and its metadata
    """

    # Source metadata
    src_crs = src_meta["crs"]
    src_count = src_meta["count"]
    src_dtype = src_meta["dtype"]
    src_nodata = src_meta["nodata"]

    # Destination metadata
    dst_crs = dst_meta["crs"]
    meta = dst_meta.copy()

    # Reproject data
    dst_arr = np.empty(
        (src_count, dst_meta["height"], dst_meta["width"]), dtype=src_dtype
    )
    dst_arr, dst_tr = warp.reproject(
        source=src_arr,
        destination=dst_arr,
        src_crs=src_crs,
        dst_crs=dst_crs,
        src_transform=src_meta["transform"],
        dst_transform=dst_meta["transform"],
        src_nodata=src_nodata,
        dst_nodata=dst_meta["nodata"],  # input data should be in integer
        num_threads=perf.get_max_cores(),
        resampling=resampling,
        **kwargs,
    )

    # Update metadata
    meta["count"] = src_count
    meta["nodata"] = src_nodata
    meta["dtype"] = src_dtype
    meta["transform"] = dst_tr
    meta["crs"] = dst_crs
    meta["driver"] = "GTiff"  # Force GTiff

    return dst_arr, meta
