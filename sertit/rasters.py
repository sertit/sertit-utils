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
"""
Raster tools

You can use this only if you have installed sertit[full] or sertit[rasters]

Functions that are not lazy yet:
- vectorize -> get_valid_vector -> get_footprint
- sieve
"""

import contextlib
import logging
from collections.abc import Callable
from functools import wraps
from tempfile import TemporaryDirectory
from typing import Any

import geopandas as gpd
import geoutils as gu
import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry

from sertit.vectors import EPSG_4326

try:
    from rasterio import CRS
    from rasterio.enums import Resampling
    from rasterio.windows import Window
except ModuleNotFoundError as ex:  # pragma: no cover
    raise ModuleNotFoundError(
        "Please install 'geoutils' to use the 'rasters' package."
    ) from ex


from sertit import AnyPath, dask, geometry, logs, misc, path, perf, rasters_rio, vectors
from sertit.types import (
    AnyPathStrType,
    AnyPathType,
    AnyRasterType,
    AnyVectorType,
    AnyXrDataStructure,
    is_iterable,
)

MAX_CORES = perf.MAX_CORES
PATH_XARR_DS = AnyRasterType
LOGGER = logging.getLogger(logs.SU_NAME)

# NODATAs
UINT8_NODATA = rasters_rio.UINT8_NODATA
""" :code:`uint8` nodata """

INT8_NODATA = rasters_rio.INT8_NODATA
""" :code:`int8` nodata """

UINT16_NODATA = rasters_rio.UINT16_NODATA
""" :code:`uint16` nodata """

FLOAT_NODATA = rasters_rio.FLOAT_NODATA
""" :code:`float` nodata """

DEG_LAT_TO_M = 111_120.0
DEG_2_RAD = rasters_rio.DEG_2_RAD

# https://wiki.openstreetmap.org/wiki/Precision_of_coordinates#Precision_of_latitudes


def get_nodata_value_from_dtype(dtype) -> float:
    """
    Get default nodata value from any given dtype.

    Args:
        dtype: Dtype for the wanted nodata. Best if numpy's dtype.

    Returns:
        int: Nodata value

    Examples:
        >>> rasters.get_nodata_value_from_dtype("uint8")
        255

        >>> rasters.get_nodata_value_from_dtype("uint16")
        65535

        >>> rasters.get_nodata_value_from_dtype("int8")
        -128

        >>> rasters.get_nodata_value_from_dtype("float")
        -9999
    """
    return rasters_rio.get_nodata_value_from_dtype(dtype)


def any_raster_to_xr_ds(function: Callable) -> Callable:
    """
    Allows a function to ingest AnyRasterType and convert it into a xr.DataArray:

    - a path (:code:`Path`, :code:`CloudPath` or :code:`str`)
    - a :code:`xarray.Dataset` or a :code:`xarray.DataArray`
    - a :code:`rasterio.DatasetWriter` or :code:`rasterio.DatasetReader`
    - :code:`rasterio` dataset after reading, its array and metadata: (np.ndarray, dict)

    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function

    Examples:
        >>> # Create mock function
        >>> @path_or_dst
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
        Path or dataset wrapper
        Args:
            any_raster_type (AnyRasterType): Raster path or its dataset
            *args: args
            **kwargs: kwargs

        Returns:
            Any: regular output
        """
        if any_raster_type is None:
            raise ValueError("'any_raster_type' shouldn't be None!")

        chunks = kwargs.get("chunks", dask.get_default_chunks())

        # By default, try with the read fct: this fct returns the xr data structure as is and manages other input types such as tuple, rasterio datasets, paths...
        try:
            out = function(
                read(any_raster_type, chunks=chunks),
                *args,
                **kwargs,
            )
        except Exception as exc:
            # Try on every DataArray of the Dataset
            # TODO: handle DataTrees?
            if isinstance(any_raster_type, xr.Dataset):
                try:
                    xds_dict = {}
                    convert_to_xdataset = False
                    for var in any_raster_type.data_vars:
                        xds_dict[var] = function(any_raster_type[var], *args, **kwargs)
                        if isinstance(xds_dict[var], xr.DataArray):
                            convert_to_xdataset = True

                    # Convert in dataset if we have DataArrays, else keep the dict
                    out = xr.Dataset(xds_dict) if convert_to_xdataset else xds_dict
                except Exception as ex:  # pragma: no cover
                    raise TypeError("Function not available for xarray.Dataset") from ex
            else:  # pragma: no cover
                raise exc

        return out

    return wrapper


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def rasterize(
    xda: xr.DataArray,
    vector: AnyVectorType,
    value_field: str = None,
    default_nodata: int = 0,
    default_value: int = 1,
    name: str = None,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Rasterize a vector into raster format.

    Note that passing :code:`merge_alg = MergeAlg.add` will add the vector values to the given raster

    See: https://pygis.io/docs/e_raster_rasterize.html

    Use :code:`value_field` to create a raster with multiple values. If set to :code:`None`, the raster will be binary.

    .. WARNING::
        With dask usage, this function is not lazy (yet)!

    Args:
        xda (xr.DataArray): Path to the raster or a rasterio dataset or a xarray.DataArray, used as base for the vector's rasterization data (shape, etc.)
        vector (AnyVectorType): Vector to be rasterized
        value_field (str): Field of the vector with the values to be burnt on the raster (should be scalars). If let to None, the raster will be binary.
        default_nodata (int): Default nodata of the raster (outside the vector in the raster extent)
        default_value (int): Used as value for all geometries, if `value_field` not provided
        name (str): Name to give to the raster
    Returns:
        AnyXrDataStructure: Rasterized vector

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> vec_path = "path/to/vector.shp"
        >>> rasterize(raster_path, vec_path, value_field="classes")
    """
    vector = vectors.read(vector, crs=xda.rio.crs)

    # Manage vector values
    value = vector[value_field] if value_field else default_value

    # Manage nodata
    nodata = kwargs.pop("nodata") if "nodata" in kwargs else xda.rst.nodata

    if nodata is None:
        nodata = default_nodata

    if not name:
        try:
            name = f"Rasterized {vector.name}"
        except AttributeError:
            name = "Rasterized vector"

    # Rasterize vector
    rasterized_xds = vector.vct.rasterize(
        xda,
        in_value=value,
        out_value=nodata,
        out_dtype=np.uint8,
        all_touched=kwargs.get("all_touched", True),
        dask=dask.is_chunked(xda),
        **kwargs,
    )
    rasterized_xds.attrs["long_name"] = name

    return rasterized_xds


def _works_on_xds(function):
    """"""

    @wraps(function)
    def wrapper(xds: AnyRasterType, *_args, **_kwargs):
        """S3 environment wrapper"""
        if isinstance(xds, xr.Dataset):
            vect = {}
            for key, xda in xds.items():
                vect[key] = function(xda, *_args, **_kwargs)

            return vect
        else:
            return function(xds, *_args, **_kwargs)

    return wrapper


# Is not lazy yet in GeoUtils
@dask.guard_compute(nof_computes=1)
@any_raster_to_xr_ds
@_works_on_xds
def vectorize(
    xda: AnyRasterType,
    values: None | int | tuple = None,
    keep_values: bool = True,
    dissolve: bool = False,
    default_nodata: int = 0,
) -> gpd.GeoDataFrame:
    """
    Vectorize a xarray, both to get classes or nodata.

    If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value,
    with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.

    .. WARNING::
        - If :code:`get_nodata` is set to False:
            - Your data is casted by force into np.uint8, so be sure that your data is classified.
            - This could take a while as the computing time directly depends on the number of polygons to vectorize.
                Please be careful.
        Else:
            - You will get a classified polygon with data (value=0)/nodata pixels.

    Args:
        xda (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
        values (None | int | list): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        default_nodata (int): Default values for nodata in case of non existing in file

    Returns:
        gpd.GeoDataFrame: Vector with the raster values (if dissolve is not set)
    """
    if dask.is_chunked(xda):
        LOGGER.warning("'_vectorize' function is not lazy yet. Computing the raster.")
        xda = xda.compute()

    # Manage values
    wanted_values = None

    # Handle all types of values
    # For now, geoutils doesn't handle a list of values
    if values is not None:
        # We want to keep the values given as input
        if keep_values:
            if is_iterable(values):
                target_values = "all"
                wanted_values = values
            else:
                target_values = values

            if not is_iterable(values):
                target_values = values
            else:
                target_values = "all"
                wanted_values = tuple(
                    i for i in np.unique(xda) if (i not in values) and (not np.isnan(i))
                )
        # We want to discard the values given as input
        else:
            if not is_iterable(values):
                values = (values,)

            target_values = "all"
            wanted_values = tuple(
                i for i in np.unique(xda) if (i not in values) and (not np.isnan(i))
            )
    else:
        target_values = "all"

    LOGGER.debug(f"Vectorizing with {target_values=}")

    gdf = xda.rst.polygonize(target_values=target_values)

    # Keep only selected values
    if wanted_values is not None:
        gdf = gdf.loc[np.isin(gdf["raster_value"], wanted_values)]

    # Return valid geometries
    gdf = geometry.make_valid(gdf)

    # Dissolve if needed
    if dissolve:
        gdf = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs).dissolve()

    return gdf


@any_raster_to_xr_ds
@_works_on_xds
def get_valid_vector(xda: AnyRasterType, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get the valid data of a raster, returned as a vector.

    Pay attention that multiple polygons can be returned.
    If you want only the footprint of the raster, please use :py:func:`rasters.get_footprint`.

    .. WARNING::
        With dask usage, this function is not lazy (yet) (depending on _vectorize)

    Args:
        xda (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
        default_nodata (int): Default values for nodata in case of non-existing in file

    Returns:
        gpd.GeoDataFrame: Nodata Vector

    Example:
        >>> raster_path = "path/to/raster.tif"
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
    return vectorize(xda, values=None, dissolve=True, default_nodata=default_nodata)


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def mask(
    xda: AnyRasterType,
    shapes: gpd.GeoDataFrame | BaseGeometry | list,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Masking a dataset:
    setting nodata outside the given shapes, but without cropping the raster to the shapes extent.

    The original nodata is kept and completed with the nodata provided by the shapes.

    Overload of rasterio mask function in order to create a :code:`xarray`.

    The :code:`mask` function docs can be seen `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html>`_.
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        xda (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
        shapes (gpd.GeoDataFrame | BaseGeometry | list): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted)
        **kwargs: Other geoutils.clip options

    Returns:
         AnyXrDataStructure: Masked array as a xarray

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> shape_path = "path/to/shapes.geojson"  # Any vector that geopandas can read
        >>> shapes = gpd.read_file(shape_path)
        >>> mask1 = mask(raster_path, shapes)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     mask2 = mask(ds, shapes)
        >>>
        >>> # Assert those two approaches give the same result
        >>> mask1 == mask2
        True
    """
    all_touched = kwargs.get("all_touched")

    return xda.copy(data=xda.rst.clip(shapes, all_touched=all_touched).data)


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def paint(
    xds: AnyRasterType,
    shapes: gpd.GeoDataFrame | BaseGeometry | list,
    value: int,
    invert: bool = False,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Painting a dataset: setting values inside the given shapes. To set outside the shape, set invert=True.
    Pay attention that this behavior is the opposite of the :code:`rasterio.mask` function.

    The original nodata is kept.
    This means if your shapes intersects the original nodata,
    the value of the pixel will be set to nodata rather than to the wanted value.

    Overload of rasterio mask function in order to create a :code:`xarray`.
    The :code:`mask` function docs can be seen `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html>`_.

    Args:
        xds (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
        shapes (gpd.GeoDataFrame | BaseGeometry | list): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted)
        value (int): Value to set on the shapes.
        invert (bool): If invert is True, set value outside the shapes.
        **kwargs: Other rasterio.mask options

    Returns:
         AnyXrDataStructure: Painted array as a xarray

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> shape_path = "path/to/shapes.geojson"  # Any vector that geopandas can read
        >>> shapes = gpd.read_file(shape_path)
        >>> paint1 = paint(raster_path, shapes, value=100)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     paint2 = paint(ds, shapes, value=100)
        >>>
        >>> # Assert those two approaches give the same result
        >>> paint1 == paint2
        True
    """
    # Fill na values in order to not interfere with the mask function
    nodata = xds.rst.nodata
    xds_fill = xds.fillna(nodata) if nodata is not None else xds

    # Use classic option
    arr = mask(xds_fill, shapes=shapes, nodata=value, invert=not invert, **kwargs)

    # Create and fill na values created by the mask to the wanted value
    painted_xds = xds.copy(data=arr)
    painted_xds = painted_xds.fillna(value)

    # Keep all attrs after fillna
    painted_xds.rio.update_attrs(xds.attrs, inplace=True)
    painted_xds.rio.update_encoding(xds.encoding, inplace=True)

    # Set back nodata to keep the original nodata
    if nodata is not None:
        painted_xds = set_nodata(painted_xds, nodata)

    # Convert back to xarray
    return painted_xds


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def crop(
    xds: AnyRasterType,
    shapes: gpd.GeoDataFrame | BaseGeometry | list,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Cropping a dataset:
    setting nodata outside the given shapes AND cropping the raster to the shapes extent.

    Overload of :py:func:`rioxarray.clip` or :py:func:`xds.odc.crop` in case of a dask array

    Args:
        xds (:any:`AnyRasterType`): Path to the raster or a rasterio dataset or a xarray
        shapes (gpd.GeoDataFrame | BaseGeometry | list): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted)
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesn't exist, set to 0.
        **kwargs: Other :code:`rioxarray.clip` options

    Returns:
         AnyXrDataStructure: Cropped array as a xarray

    Examples:
        >>> raster_path = "path/to/raster.tif"
        >>> shape_path = "path/to/shapes.geojson"  # Any vector that geopandas can read
        >>> shapes = gpd.read_file(shape_path)
        >>> xds2 = crop(raster_path, shapes)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     xds2 = crop(ds, shapes)
        >>>
        >>> # Assert those two approaches give the same result
        >>> xds1 == xds2
        True
    """
    all_touched = kwargs.get("all_touched")
    crop_xda = xds.rst.clip(mask=shapes, all_touched=all_touched).rst.crop(shapes)
    crop_xda = set_metadata(crop_xda, xds)
    return crop_xda


def __read__any_raster_to_rio_ds(function: Callable) -> Callable:
    """
    Specific declination of rasters_rio.any_raster_to_rio_ds for this specific case, handling the xarray object differently.

    Allows a function to ingest AnyRasterType and convert it into a rasterio.DatasetReader:

    - a path (:code:`Path`, :code:`CloudPath` or :code:`str`)
    - a :code:`rasterio.DatasetWriter` or :code:`rasterio.DatasetReader`
    - :code:`rasterio` dataset after reading, its array and metadata: (np.ndarray, dict)

    But returns directly any xarray object

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
            import rasterio

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

            from rasterio import MemoryFile

            with MemoryFile() as memfile:
                #  Open in write mode
                with memfile.open(**meta, BIGTIFF=rasters_rio.bigtiff_value(arr)) as ds:
                    ds.write(arr)
                # Open in read mode (otherwise rioxarray will fail)
                with memfile.open() as ds:
                    LOGGER.warning(
                        "Giving a tuple as the function's input makes you load the result in memory. Consider using an xarray structure, a path or a rasterio Dataset instead."
                    )
                    out = function(ds, *args, **kwargs, chunks=None)

                    with contextlib.suppress(AttributeError):
                        out = out.compute()

        # Return given xarray object as is
        elif isinstance(any_raster_type, (xr.DataArray, xr.Dataset)):
            out = any_raster_type

        # Run the fct directly on the input (which should be a rasterio Dataset). If not, this will fail and it's expected.
        else:
            out = function(any_raster_type, *args, **kwargs)
        return out

    return wrapper


# Is lazy
@dask.guard_compute(max_computes=0)
def get_window(xda: AnyRasterType, window: Any):
    """
    Get a window from any type of input

    Args:
        xda (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        window (Any): Anything that can be returned as a window. In case of iterable, assumption is made it's geographic bounds. For pixel, please provide a Window directly.

    Returns:
        Window: Rasterio window

    """
    return rasters_rio._get_col_row_window(window, xda.rst.crs, xda.rst.transform)


def get_new_shape(
    src_width,
    src_height,
    src_res,
    resolution: tuple | list | float = None,
    size: tuple | list = None,
    window: Window = None,
) -> (int, int, bool):
    """
    Get the new shape (height, width) of a resampled raster.
    Get also the downsampling_factor to select properly which overview to use. (>=1)

    Size overrides resolution

    Args:
        ds (AnyRasterType): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        resolution (tuple | list | float): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (tuple | list): Size of the array (width, height). Not used if resolution is provided.
        window (Window): Window to be read

    Returns:
        (int, int, bool): Height, width, downsampling_factor

    """
    # By default, set downsampling_factor to 1
    downsampling_factor = 1

    def _get_new_dim(dim: int, res_old: float, res_new: float) -> (int, float):
        """
        Get the new dimension in pixels and the downsampling_factor used in GeoUtils.

        GeoUtils takes the downsample factor as a scalar with the definition downsample = new / old resolution.
        i.e. downsample = 2 on a 10m raster means "give me the raster at a 20m resolution"
        -> downsample = new_res / old_res = 20 / 10 = 2

        Args:
            dim (int): Old dimension
            res_old (float): Old resolution
            res_new (float): New resolution

        Returns:
            (int, bool): New dimension, downsampling_factor factor (new / old)
        """
        new_dim = int(np.round(dim * res_old / res_new))
        return new_dim, res_new / res_old

    # By default keep original shape
    if window is None:
        new_height = src_height
        new_width = src_width
    else:
        new_height = window.height
        new_width = window.width

    # Compute new shape
    if size is not None:
        try:
            if new_height == size[1] and new_width == size[0]:
                downsampling_factor = 1
            else:
                new_height = size[1]
                new_width = size[0]
                downsampling_factor = min(size[1] / new_height, size[0] / new_width)
        except (TypeError, KeyError) as exc:
            raise ValueError(
                f"Size should be None or a 2 digits iterable: {size}"
            ) from exc
    elif resolution is not None:
        if isinstance(resolution, (int, float)):
            new_height, downsampling_h = _get_new_dim(
                new_height, src_res[1], resolution
            )
            new_width, downsampling_w = _get_new_dim(new_width, src_res[0], resolution)
        else:
            try:
                if len(resolution) != 2:
                    raise ValueError(
                        "We should have a resolution for X and Y dimensions"
                    )

                if resolution[0] is not None:
                    new_width, downsampling_h = _get_new_dim(
                        new_width, src_res[0], resolution[0]
                    )
                else:
                    downsampling_h = 1

                if resolution[1] is not None:
                    new_height, downsampling_w = _get_new_dim(
                        new_height, src_res[1], resolution[1]
                    )
                else:
                    downsampling_w = 1
            except (TypeError, KeyError) as exc:
                raise ValueError(
                    f"Resolution should be None or a 2 digits iterable: {resolution}"
                ) from exc

        downsampling_factor = min(downsampling_h, downsampling_w)
    else:
        downsampling_factor = 1

    # downsampling_factor cannot be inferior to 1
    downsampling_factor = max(1, downsampling_factor)

    return new_height, new_width, downsampling_factor


@__read__any_raster_to_rio_ds
def read(
    ds: AnyRasterType,
    resolution: tuple | list | float = None,
    size: tuple | list = None,
    window: Any = None,
    resampling: Resampling = Resampling.nearest,
    masked: bool = True,
    indexes: int | list = None,
    as_type: Any = None,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Read a raster dataset from a :

    - :code:`xarray` (compatibility issues)
    - :code:`rasterio.Dataset`
    - :code:`rasterio` opened data (array, metadata)
    - a path.

    The resolution can be provided (in dataset unit) as:

    - a tuple or a list of (X, Y) resolutions
    - a float, in which case X resolution = Y resolution
    - None, in which case the dataset resolution will be used

    Uses `rioxarray.open_rasterio <https://vylor.github.io/rioxarray/stable/rioxarray.html#rioxarray-open-rasterio>`_.

    .. WARNING::
        With dask usage, this function is not lazy (yet) when downsampling! (becaise of a odc.geo issue: https://github.com/opendatacube/odc-geo/issues/236)

    Args:
        ds (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
        resolution (tuple | list | float): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (tuple | list): Size of the array (width, height). Overrides resolution.
        window (Any): Anything that can be returned as a window (i.e. path, gpd.GeoPandas, Iterable, rasterio.Window...).
            In case of an iterable, assumption is made it corresponds to geographic bounds.
            For pixel, please provide a rasterio.Window directly.
        resampling (Resampling): Resampling method
        masked (bool): Get a masked array
        indexes (int | list): Indexes of the band to load. Load the whole array if None. Starts at 1 like GDAL.
        as_type (Any): Type in which to load the array
        **kwargs: Optional keyword arguments to pass into rioxarray.open_rasterio().

    Returns:
        AnyXrDataStructure: Masked xarray corresponding to the raster data and its metadata

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> xds1 = read(raster_path)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>    xds2 = read(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> xds1 == xds2
        True
    """

    # Manage chunks
    chunks = kwargs.pop("chunks", dask.get_default_chunks())

    # Manage window
    if window is not None:
        window = rasters_rio.get_window(ds, window)

    # Get new height and width
    new_height, new_width, downsampling_factor = get_new_shape(
        src_width=ds.width,
        src_height=ds.height,
        src_res=ds.res,
        resolution=resolution,
        size=size,
        window=window,
    )

    with gu.open_raster(
        ds.name,
        downsample=downsampling_factor,
        default_name=path.get_filename(ds.name),
        masked=masked,
        squeeze=False,
        chunks=chunks,
        **kwargs,
    ) as xda:
        # Windows
        if window is not None:
            xda = xda.rio.isel_window(window)

        # Indexes
        if indexes is not None:
            if not isinstance(indexes, list):
                indexes = [indexes]

            # Open only wanted bands
            if 0 in indexes:
                raise ValueError("Indexes should start at 1.")

            ok_indexes = np.isin(indexes, xda.band)
            if any(~ok_indexes):
                LOGGER.warning(
                    f"Non-available index: {[idx for i, idx in enumerate(indexes) if not ok_indexes[i]]} for {ds.name}"
                )

            xda = xda.isel(
                band=[
                    idx - 1 for ok, idx in zip(ok_indexes, indexes, strict=True) if ok
                ]
            )

            with contextlib.suppress(AttributeError):
                # Set new long name: Bands nb are idx + 1
                xda.long_name = tuple(
                    name for i, name in enumerate(xda.long_name) if i + 1 in indexes
                )

        # Manage dst fixed size
        if xda.rst.height != new_height or xda.rst.width != new_width:
            LOGGER.debug(
                "Forcing the correct shape or resolution by reprojecting the data."
            )
            # Manage nodata
            xda = reproject(
                xda,
                shape=(new_height, new_width),
                resampling=resampling,
                dst_crs=xda.rst.crs,
                use_dask=chunks is not None,
            )

        # Convert to wanted type
        if as_type:
            if np.dtype(as_type).kind in "ui":
                LOGGER.warning(
                    "You asked for integer arrays. Please note that this kind of dtypes is not handled by GeoUtils and may fail silently."
                )
                xda = xda.fillna(xda.rst.nodata)
            xda = xda.astype(as_type)

    return xda


def __save_cog_with_dask(
    xds: AnyXrDataStructure, nodata, dtype, output_path, compute, kwargs
):
    from odc.geo import cog, xr  # noqa

    # Fill the nodata to the destination value (instead of NaN)
    # Convert to the wanted dtype
    # Set the nodata value in the array
    xds = xds.fillna(nodata).astype(dtype).rio.set_nodata(nodata, inplace=False)

    # Set nodata also for odc, otherwise it can throw an overflow error
    xds.odc.nodata = nodata

    # Save the COG as a delayed object
    delayed = cog.save_cog_with_dask(
        xds,
        str(output_path),
        **kwargs,
    )

    # Compute the delayed object if needed
    if compute:
        LOGGER.debug(f"Writing your COG '{path.get_filename(output_path)}' with Dask.")
        delayed.compute(optimize_graph=True)

    return delayed


@any_raster_to_xr_ds
def write(
    xds: AnyXrDataStructure,
    output_path: AnyPathStrType = None,
    tags: dict = None,
    write_cogs_with_dask: bool = True,
    compute: bool = True,
    **kwargs,
) -> None:
    """
    Write raster to disk.
    (encapsulation of :code:`rasterio`'s function, because for now :code:`rioxarray` to_raster doesn't work as expected)

    Metadata will be created with the :code:`xarray` metadata (i.e. width, height, count, type...)
    The driver is :code:`GTiff` by default, and no nodata value is provided.
    The file will be compressed if the raster is a mask (saved as uint8).

    If not overwritten, sets the nodata according to :code:`dtype`:

    - :code:`uint8`: 255
    - :code:`int8`: -128
    - :code:`uint16`, :code:`uint32`, :code:`int32`, :code:`int64`, :code:`uint64`, :code:`int`: 65535
    - :code:int16, :code:`float32`, :code:`float64`, :code:`float128`, :code:`float`: -9999

    Default parameters

    - Compress with :code:`LZW` option by default. To disable it, add the :code:`compress=None` parameter.
    - :code:`predictor` set to `2` for float data, to `3` for interger data by default. To disable it, add the :code:`predictor=None` parameter.
    - :code:`tiled` set to `True` by default.
    - :code:`driver` is :code:`GTiff` by default. BigTiff option is set according to the estimated output weight

    .. WARNING::
        With dask usage, this function is not lazy by default. Use :code:`compute=False` with a delayed return.

    Args:
        xds (AnyXrDataStructure): Path to the raster or a rasterio dataset or a xarray
        output_path (AnyPathStrType): Path where to save it (directories should be existing)
        tags (dict): Tags that will be written in your file
        write_cogs_with_dask (bool): If odc-geo and imagecodecs are installed, write your COGs with Dask.
            Otherwise, the array will be loaded into memory before writing it on disk (and can cause MemoryErrors). Only used with chunked data.
        compute (bool): If True (default) and data is a dask array, then compute and save the data immediately.
            If False, return a dask Delayed object. Call ".compute()" on the Delayed object to compute the result later.
            Call ``dask.compute(delayed1, delayed2)`` to save multiple delayed files at once. Only useful with chunked data.
        **kwargs: Overloading metadata, ie :code:`nodata=255` or :code:`dtype=np.uint8`

    Examples:
        >>> raster_path = "path/to/raster.tif"
        >>> raster_out = "path/to/out.tif"
        >>>
        >>> # Read raster
        >>> xds = read(raster_path)
        >>>
        >>> # Rewrite it
        >>> write(xds, raster_out)
    """
    delayed = None
    use_dask = dask.is_chunked(xds)
    compute = compute and use_dask

    # Prune empty kwargs to avoid throwing GDAL warnings/errors
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Manage dtype
    dtype = np.dtype(kwargs.get("dtype", xds.encoding.get("dtype", xds.dtype)))
    xds.encoding["dtype"] = dtype.name

    # Write nodata
    nodata = kwargs.pop("nodata", get_nodata_value_from_dtype(dtype))
    xds = xds.rio.write_nodata(nodata, encoded=True)

    # Bigtiff if needed
    bigtiff = rasters_rio.bigtiff_value(xds)

    # Force GTiff by default
    kwargs["driver"] = kwargs.get("driver", "GTiff")

    # Manage COGs or other drivers attributes
    is_cog = kwargs["driver"] == "COG"
    is_gtiff = kwargs["driver"] == "GTiff"
    is_zarr = kwargs["driver"] == "Zarr"

    if is_cog:
        kwargs.pop("tiled", None)

        # Default compression to deflate for COGs
        kwargs["compress"] = kwargs.get("compress", "deflate")

        if dtype == np.int8:  # pragma: no cover
            LOGGER.warning(
                "For some reason, it is impossible to write int8 COGs. "
                "Your data will be converted to uint8. "
                "In case of casting issues (i.e. negative values), please save it to int16."
            )
    elif is_zarr:
        # Get default client's lock
        kwargs["lock"] = kwargs.get("lock", dask.get_dask_lock("rio"))
        kwargs["compress"] = kwargs.get("compress", "zstd")
        kwargs["tiled"] = kwargs.get("tiled", True)
    else:
        # Get default client's lock
        kwargs["lock"] = kwargs.get("lock", dask.get_dask_lock("rio"))

        # Set tiles by default
        kwargs["tiled"] = kwargs.get("tiled", True)

        # Default compression to LZW
        if is_gtiff:
            kwargs["compress"] = kwargs.get("compress", "lzw")
        # Else, don't set any default compression

    # Manage predictors according to dtype and compression
    if (
        not is_zarr
        and kwargs.get("compress", "").lower() in ["lzw", "deflate", "zstd"]
        and "predictor" not in kwargs
    ):
        if dtype.kind == "f":
            kwargs["predictor"] = "3"
        else:
            kwargs["predictor"] = "2"

    # -- Write --
    is_written = False

    # Write COGs
    blocksize = None
    if is_cog:
        blocksize = 128 if (xds.rio.height < 1000 or xds.rio.width < 1000) else 512

        if use_dask and write_cogs_with_dask:
            try:
                # Filter out and convert kwargs to avoid any error
                da_kwargs = {
                    "stats": np.dtype(dtype).itemsize >= 4,
                    # Other default arguments
                    "compression": kwargs["compress"].upper(),
                    "level": kwargs.get("level"),
                    "compressionargs": kwargs.get("compressionargs"),
                    "overview_resampling": kwargs.get("overview_resampling", "nearest"),
                }

                # Cannot give a None blockwise to "save_cog_with_dask"
                if blocksize is not None:
                    da_kwargs["blocksize"] = blocksize

                predictor = kwargs.get("predictor")
                if predictor is not None:
                    da_kwargs["predictor"] = int(predictor)

                # Write cog on disk
                try:
                    try:
                        delayed = __save_cog_with_dask(
                            xds, nodata, dtype, output_path, compute, da_kwargs
                        )
                    except Exception:
                        # Remove computing statistics for some problematic (for now) dtypes (we need the ability to cast 999999 inside it)
                        # OverflowError: Python integer 999999 out of bounds for xxx
                        # https://github.com/opendatacube/odc-geo/issues/189#issuecomment-2513450481
                        da_kwargs["stats"] = False
                        delayed = __save_cog_with_dask(
                            xds, nodata, dtype, output_path, compute, da_kwargs
                        )

                    is_written = True
                except SystemError:
                    # Sometimes, some weird multithread errors happen
                    # SystemError: Objects/tupleobject.c:927: bad argument to internal function
                    is_written = False

            except (ModuleNotFoundError, KeyError):  # pragma: no cover
                # COGs cannot be written via dask via rioxarray for the moment
                LOGGER.debug(
                    "Loading raster in memory as COGs cannot be written with Dask via 'rioxarray' for the moment. "
                    "Please install 'odc-geo' and 'imagecodecs' for Dask handling."
                )

                xds = xds.load()
            except AttributeError:
                # Numpy array, not dask arrays
                pass
        else:
            LOGGER.debug(
                "Loading raster in memory as COG has been asked not to be written by Dask. Be careful about MemoryErrors!"
            )
            xds = xds.load()
            if blocksize is not None:
                kwargs["BLOCKSIZE"] = blocksize

        if not use_dask and not is_written:
            # Write with windows as we don't want to_raster to blow up the RAM (only if dask is not used)
            # Does nothing if writing with dask
            use_windowed = xds.rio.height * xds.rio.width > 20000 * 20000
            if use_windowed:
                LOGGER.debug(
                    f"Writing '{path.get_filename(output_path)}' big file (h: {xds.rio.height} x w: {xds.rio.width}) with rioxarray's 'windowed' method to avoid memory issues."
                )
            kwargs["windowed"] = use_windowed

    # Default write on disk
    if not is_written:
        LOGGER.debug(
            f"Writing '{path.get_filename(output_path)}' (h: {xds.rio.height} x w: {xds.rio.width}) to disk."
        )

        # WORKAROUND: Pop _FillValue attribute (if existing)
        if "_FillValue" in xds.attrs:  # pragma: no cover
            xds.attrs.pop("_FillValue")

        if not is_zarr:
            kwargs["BIGTIFF"] = bigtiff
            kwargs["NUM_THREADS"] = perf.get_max_cores()

        delayed = xds.rst.to_file(
            str(output_path),
            tags=tags,
            compute=compute,
            **misc.remove_empty_values(kwargs),
        )
    return delayed


# Is lazy
@dask.guard_compute(max_computes=0)
def _collocate_dataarray(
    reference: xr.DataArray,
    other: AnyXrDataStructure,
    resampling: Resampling = Resampling.nearest,
    **kwargs,
) -> xr.DataArray:
    if other.rio.shape == reference.rio.shape and other.rio.crs == reference.rio.crs:
        LOGGER.debug(
            "Collocating equivalent rasters by aligning the coordinates onto the reference's ones."
        )
        # Same rasters, but just a bit shifted max (i.e. error in float64 coordinates)
        # Should do this (but done anyway in the end)
        # collocated_xda = other.assign_coords(
        #     {
        #         "x": reference.x,
        #         "y": reference.y,
        #     }
        # )
        collocated_xda = other
    else:
        collocated_xda = other.rst.reproject(
            ref=reference, resampling=resampling, **kwargs
        )

    return collocated_xda


# Is lazy
@dask.guard_compute(max_computes=0)
def collocate(
    reference: AnyXrDataStructure,
    other: AnyXrDataStructure,
    resampling: Resampling = Resampling.nearest,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Collocate two georeferenced arrays:
    forces the *other* raster to be exactly georeferenced onto the *reference* raster by reprojection.

    .. WARNING::
        With dask usage, this function is only lazy if odc.geo is installed

    Args:
        reference (:any:`AnyXrDataStructure`): Reference xarray
        other (:any:`AnyXrDataStructure`): Other xarray
        resampling (:class:`rasterio:rasterio.enums.Resampling`): Resampling method

    Returns:
        AnyXrDataStructure: Collocated xarray

    Examples:
        >>> reference_path = "path/to/reference.tif"
        >>> other_path = "path/to/other.tif"
        >>> col_path = "path/to/collocated.tif"
        >>>
        >>> # Collocate the other to the reference
        >>> col_xds = collocate(read(reference_path), read(other_path), Resampling.bilinear)
        >>>
        >>> # Write it
        >>> write(col_xds, col_path)
    """
    if isinstance(other, xr.DataArray):
        collocated_xds = _collocate_dataarray(
            reference=reference, other=other, resampling=resampling, **kwargs
        )
    else:
        try:
            # Code inspired by the 'reproject_match' code from RasterDataset
            collocated_xds = xr.Dataset(attrs=other.attrs)
            for var in other.rio.vars:
                # Get spatial dimensions from current array
                x_dim, y_dim = other[var].rio.x_dim, other[var].rio.y_dim

                if isinstance(reference, xr.DataArray):
                    ref = reference
                else:
                    # assert we have the same spatial dimensions in ref and other
                    ref = reference[var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

                # Colocate dataarray by dataarray
                collocated_xds[var] = _collocate_dataarray(
                    other=other[var].rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim),
                    reference=ref,
                    resampling=resampling,
                    **kwargs,
                )
            collocated_xds = collocated_xds.rio.set_spatial_dims(
                x_dim=other.rio.x_dim, y_dim=other.rio.y_dim
            )
        except Exception:  # pragma: no cover
            # For datasets, maybe the code here over is a bit much, so use the reproject match function as a backup (just in case)
            collocated_xds = _collocate_dataarray(
                reference=reference, other=other, resampling=resampling, **kwargs
            )

    # Assign directly the coordinates to correct tiny differences in coordinates due to float precision

    # From odc-geo
    STANDARD_SPATIAL_DIMS = [
        ("y", "x"),
        ("yc", "xc"),
        ("latitude", "longitude"),
        ("lat", "lon"),
    ]
    coord_dict = {}

    for dims in STANDARD_SPATIAL_DIMS:
        if dims[0] in reference.dims and dims[1] in reference.dims:
            coord_dict = {
                dims[0]: reference[dims[0]],
                dims[1]: reference[dims[1]],
            }
    if not coord_dict:  # pragma: no cover
        raise ValueError(
            "No standard spatial dimensions (y, x) or (latitude, longitude) found in reference. "
            f"Something is wrong with your reference dimensions: ({reference.dims})."
        )

    collocated_xds = collocated_xds.assign_coords(coord_dict)

    # Set back attributes and encoding
    collocated_xds.rio.update_attrs(other.attrs, inplace=True)
    collocated_xds.rio.update_encoding(other.encoding, inplace=True)

    return collocated_xds


# Is not lazy yet in GeoUtils
@dask.guard_compute(nof_computes=1)
@any_raster_to_xr_ds
def sieve(
    xds: AnyRasterType, sieve_thresh: int, connectivity: int = 4
) -> AnyXrDataStructure:
    """
    Sieving, overloads rasterio function with raster shaped like an image: :code:`(1, h, w)`.

    .. WARNING::
        Your data is casted by force into :code:`np.uint8`, so be sure that your data is classified.

    .. WARNING::
        With dask usage, this function is not lazy (yet)!

    Args:
        xds (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
        sieve_thresh (int): Sieving threshold in pixels
        connectivity (int): Connectivity, either 4 or 8

    Returns:
        (AnyXrDataStructure): Sieved xarray

    Example:
        >>> raster_path = "path/to/raster.tif"  # classified raster
        >>>
        >>> # Rewrite it
        >>> sieved_xds = sieve(raster_path, sieve_thresh=20)
        >>>
        >>> # Write it
        >>> raster_out = "path/to/raster_sieved.tif"
        >>> write(sieved_xds, raster_out)
    """
    if xds.dtype.kind not in "biu":
        LOGGER.warning(
            "Sieving takes classified rasters. Converting the raster to uint8."
        )
        # Preserve attributes and encodings
        xds = xds.copy(data=xds.astype("uint8"))

    return xds.rst.sieve(size=sieve_thresh, connectivity=connectivity)


def get_dim_img_path(
    dim_path: AnyPathStrType, img_name: str = "*", get_list: bool = False
) -> list | AnyPathType:
    """
    Get the image path (:code:`.img`) from a :code:`BEAM-DIMAP` data.

    A :code:`BEAM-DIMAP` file cannot be opened by rasterio, although its :code:`.img` file can.

    Args:
        dim_path (AnyPathStrType): DIM path (.dim or .data)
        img_name (str): .img file name (or regex), in case there are multiple .img files (i.e. for S3 data)
        get_list

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
    return rasters_rio.get_dim_img_path(dim_path, img_name, get_list)


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def get_extent(xds: AnyRasterType) -> gpd.GeoDataFrame:
    """
    Get the extent of a raster as a :code:`geopandas.Geodataframe`.

    Args:
        xds (AnyRasterType): Path to the raster or a rasterio dataset or a xarray

    Returns:
        gpd.GeoDataFrame: Extent as a :code:`geopandas.Geodataframe`

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>>
        >>> extent1 = get_extent(raster_path)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     extent2 = get_extent(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> extent1 == extent2
        True
    """
    return vectors.get_geodf(geom=[*xds.rst.bounds], crs=xds.rst.crs)


@any_raster_to_xr_ds
def get_footprint(xds: AnyRasterType) -> gpd.GeoDataFrame:
    """
    Get real footprint of the product (without nodata, *in french == emprise utile*)

    .. WARNING::
        With dask usage, this function is not lazy (yet) (depending on _vectorize)

    Args:
        xds (AnyRasterType): Path to the raster or a rasterio dataset or a xarray
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
        True
    """
    footprint = get_valid_vector(xds)
    return geometry.get_wider_exterior(footprint)


# Is lazy
@dask.guard_compute(max_computes=0)
def merge_vrt(
    crs_paths: list,
    crs_merged_path: AnyPathStrType,
    abs_path: bool = False,
    **kwargs,
) -> None:
    """
    Merge rasters as a VRT. Uses :code:`gdalbuildvrt`.

    See here: https://gdal.org/programs/gdalbuildvrt.html

    Creates VRT with relative paths!

    This function handles files of different projection by creating intermediate VRT used for warping (longer to open).

    All VRTs will be written with relative paths.

    Args:
        crs_paths (list): Path of the rasters to be merged with the same CRS
        crs_merged_path (AnyPathStrType): Path to the merged raster
        abs_path (bool): VRT with absolute paths. If not, VRT with relative paths (default)
        kwargs: Other gdlabuildvrt arguments

    Example:
        >>> paths_utm32630 = ["path/to/raster1.tif", "path/to/raster2.tif", "path/to/raster3.tif"]
        >>> paths_utm32631 = ["path/to/raster4.tif", "path/to/raster5.tif"]
        >>>
        >>> mosaic_32630 = "path/to/mosaic_32630.vrt"
        >>> mosaic_32631 = "path/to/mosaic_32631.vrt"
        >>>
        >>> # Create mosaic, one per CRS or not (longer to open)
        >>> merge_vrt(paths_utm32630, mosaic_32630)
        >>> merge_vrt(paths_utm32631, mosaic_32631, {"-srcnodata":255, "-vrtnodata":0})
    """
    return rasters_rio.merge_vrt(crs_paths, crs_merged_path, abs_path, **kwargs)


# Is lazy
@dask.guard_compute(max_computes=0)
def merge_gtiff(rasters_paths: list, merged_path: AnyPathStrType, **kwargs) -> None:
    """
    Merge rasters as a GeoTiff.

    This function handles files of different projection by creating intermediate VRT used for warping.

    Using :code:`rasterio.merge` with default arguments behind the hood so you can provide any valid argument into the kwargs.
    For example, to modify the merging method, you can pass :code:`method="max"`

    .. WARNING::
        With dask usage, this function is not lazy

    Args:
        crs_paths (list): Path of the rasters to be merged with the same CRS
        crs_merged_path (AnyPathStrType): Path to the merged raster
        kwargs: Other rasterio.merge arguments.
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
    rasters = [
        gu.Raster(str(rasters_path)) if path.is_path(rasters_path) else rasters_path
        for rasters_path in rasters_paths
    ]

    merged_raster = gu.raster.merge_rasters(
        rasters=rasters,
        **misc.select_dict(
            kwargs,
            ["reference", "merge_algorithm", "resampling_method", "use_ref_bounds"],
        ),
    ).to_xarray("Merged")
    if merged_path:
        dtype = kwargs.pop("dtype", rasters[0].dtype)
        write(merged_raster, merged_path, dtype=dtype, **kwargs)


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
    return rasters_rio.unpackbits(array, nof_bits)


# Is lazy
@dask.guard_compute(max_computes=0)
def read_bit_array(
    bit_mask: xr.DataArray | np.ndarray, bit_id: int | list
) -> np.ndarray | list:
    """
    Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number :code:`bit_id`)

    Args:
        bit_mask (np.ndarray): Bit array to read
        bit_id (int): Bit ID of the slice to be read
          Example: read the bit 0 of the mask as a cloud mask (Theia)

    Returns:
        np.ndarray | list: Binary mask or list of binary masks if a list of bit_id is given

    Example:
        >>> bit_array = np.random.randint(5, size=[3,3])
        array([[1, 1, 3],
               [4, 2, 0],
               [4, 3, 2]], dtype=uint8)
        >>>
        >>> # Get the 2nd bit array
        >>> read_bit_array(bit_array, 2)
        array([[0, 0, 0],
               [1, 0, 0],
               [1, 0, 0]], dtype=uint8)
    """
    if isinstance(bit_mask, np.ndarray):
        bit_mask = np.nan_to_num(bit_mask)
    elif isinstance(bit_mask, xr.DataArray):
        orig_dtype = bit_mask.encoding.get("dtype")
        bit_mask = bit_mask.fillna(0).data
        if orig_dtype is not None and bit_mask.dtype != orig_dtype:
            bit_mask = bit_mask.astype(orig_dtype)
    else:
        try:
            bit_mask = bit_mask.fillna(0)
        except AttributeError:
            # dask array
            bit_mask = np.nan_to_num(bit_mask)

    return rasters_rio.read_bit_array(bit_mask, bit_id)


# Is lazy
@dask.guard_compute(max_computes=0)
def read_uint8_array(
    bit_mask: xr.DataArray | np.ndarray, bit_id: int | list
) -> np.ndarray | list:
    """
    Read 8 bit arrays as a succession of binary masks.

    Forces array to :code:`np.uint8`.

    See :py:func:`rasters.read_bit_array`.

    Args:
        bit_mask (np.ndarray): Bit array to read
        bit_id (int): Bit ID of the slice to be read
          Example: read the bit 0 of the mask as a cloud mask (Theia)

    Returns:
        np.ndarray | list: Binary mask or list of binary masks if a list of bit_id is given
    """
    if isinstance(bit_mask, np.ndarray):
        bit_mask = np.nan_to_num(bit_mask)
    elif isinstance(bit_mask, xr.DataArray):
        bit_mask = bit_mask.fillna(0).data
    else:
        try:
            bit_mask = bit_mask.fillna(0)
        except AttributeError:
            # dask array
            bit_mask = np.nan_to_num(bit_mask)

    return read_bit_array(bit_mask.astype(np.uint8), bit_id)


# Is lazy
@dask.guard_compute(max_computes=0)
def set_metadata(
    naked_xda: AnyXrDataStructure, mtd_xda: AnyXrDataStructure, new_name: str = None
) -> AnyXrDataStructure:
    """
    Set metadata from a :code:`xr.DataArray` to another (including :code:`rioxarray` metadata such as encoded_nodata and crs).

    Useful when performing operations on xarray that result in metadata loss such as sums.

    Args:
        naked_xda (xr.DataArray): DataArray to complete
        mtd_xda (xr.DataArray): DataArray with the correct metadata
        new_name (str): New name for naked DataArray

    Returns:
        xr.DataArray: Complete DataArray

    Example:
        >>> # xda: some xr.DataArray
        >>> sum = xda + xda  # Sum loses its metadata here
        <xarray.DataArray 'xda' (band: 1, y: 322, x: 464)>
        array([[[nan, nan, nan, ..., nan, nan, nan],
                [nan, nan, nan, ..., nan, nan, nan],
                [nan, nan, nan, ..., nan, nan, nan],
                ...,
                [nan, nan, nan, ...,  2., nan, nan],
                [nan, nan, nan, ...,  2., nan, nan],
                [nan, nan, nan, ...,  2., nan, nan]]])
        Coordinates:
          * band         (band) int32 1
          * y            (y) float64 4.798e+06 4.798e+06 ... 4.788e+06 4.788e+06
          * x            (x) float64 5.411e+05 5.411e+05 ... 5.549e+05 5.55e+05
        >>>
        >>> # We need to set the metadata back (and we can set a new name)
        >>> sum = set_metadata(sum, xda, new_name="sum")
        <xarray.DataArray 'sum' (band: 1, y: 322, x: 464)>
        array([[[nan, nan, nan, ..., nan, nan, nan],
                [nan, nan, nan, ..., nan, nan, nan],
                [nan, nan, nan, ..., nan, nan, nan],
                ...,
                [nan, nan, nan, ...,  2., nan, nan],
                [nan, nan, nan, ...,  2., nan, nan],
                [nan, nan, nan, ...,  2., nan, nan]]])
        Coordinates:
          * band         (band) int32 1
          * y            (y) float64 4.798e+06 4.798e+06 ... 4.788e+06 4.788e+06
          * x            (x) float64 5.411e+05 5.411e+05 ... 5.549e+05 5.55e+05
            spatial_ref  int32 0
        Attributes: (12/13)
            grid_mapping:              spatial_ref
            BandName:                  Band_1
            RepresentationType:        ATHEMATIC
            STATISTICS_COVARIANCES:    0.2358157950609785
            STATISTICS_MAXIMUM:        2
            STATISTICS_MEAN:           1.3808942647686
            ...                        ...
            STATISTICS_SKIPFACTORX:    1
            STATISTICS_SKIPFACTORY:    1
            STATISTICS_STDDEV:         0.48560665546817
            STATISTICS_VALID_PERCENT:  80.07
            original_dtype:            uint8
    """
    from rioxarray.exceptions import MissingCRS

    with contextlib.suppress(MissingCRS):
        naked_xda.rst.set_crs(mtd_xda.rst.crs)

    if new_name:
        naked_xda = naked_xda.rename(new_name)
        naked_xda.attrs["long_name"] = new_name

    naked_xda.rio.update_attrs(mtd_xda.attrs, inplace=True)
    naked_xda.rio.update_encoding(mtd_xda.encoding, inplace=True)
    naked_xda.rst.nodata = mtd_xda.rst.nodata

    return naked_xda


# Is lazy
@dask.guard_compute(max_computes=0)
def set_nodata(xda: xr.DataArray, nodata_val: float) -> xr.DataArray:
    """
    Set nodata to a xarray that have no default nodata value.

    In the data array, the no data will be set to :code:`np.nan`.
    The encoded value can be retrieved with :code:`get_nodata_value_from_xr`.

    Args:
        xda (xr.DataArray): DataArray
        nodata_val (float): Nodata value

    Returns:
        xr.DataArray: DataArray with nodata set

    Example:
        >>> A = xr.DataArray(dims=("x", "y"), data=np.zeros((3,3), dtype=np.uint8))
        >>> A[0, 0] = 1
        <xarray.DataArray (x: 3, y: 3)>
        array([[1, 0, 0],
               [0, 0, 0],
               [0, 0, 0]], dtype=uint8)
        Dimensions without coordinates: x, y
        >>>
        >>> A_nodata = set_nodata(A, 0)
        <xarray.DataArray (x: 3, y: 3)>
        array([[ 1., nan, nan],
               [nan, nan, nan],
               [nan, nan, nan]])
        Dimensions without coordinates: x, y
    """
    # Where removes encoding so save them and set them back in place
    encoding = xda.encoding

    # Set nodata in the array
    xda = xda.where(xda.data != nodata_val)

    # Set encoding back
    xda.rio.update_encoding(encoding, inplace=True)

    # Set nodata in the attributes and encoding
    xda.rio.write_nodata(nodata_val, encoded=True, inplace=True)
    return xda


# Is lazy
@dask.guard_compute(max_computes=0)
def where(
    cond, if_true, if_false, master_xda: xr.DataArray = None, new_name: str = ""
) -> xr.DataArray:
    """
    Overloads :code:`xr.where` with:

    - setting metadata of :code:`master_xda`
    - preserving the nodata pixels of the :code:`master_xda`

    If :code:`master_xda` is None, use it like :code:`xr.where`.
    Else, it outputs a :code:`xarray.DataArray` with the same dtype than :code:`master_xda`.

    .. WARNING::
        If you don't give a :code:`master_xda`,
        it is better to pass numpy arrays to :code:`if_false` and :code:`if_true` keywords
        as passing xarrays interfers with the output metadata (you may lose the CRS and so on).
        Just pass :code:`if_true=true_xda.data` inplace of :code:`if_true=true_xda` and the same for :code:`if_false`

    Args:
        cond (scalar, array, Variable, DataArray or Dataset): Conditional array
        if_true (scalar, array, Variable, DataArray or Dataset): What to do if :code:`cond` is True
        if_false (scalar, array, Variable, DataArray or Dataset):  What to do if :code:`cond` is False
        master_xda: Master :code:`xr.DataArray` used to set the metadata and the nodata
        new_name (str): New name of the array

    Returns:
        xr.DataArray: Where array with correct mtd and nodata pixels

    Example:
        >>> A = xr.DataArray(dims=("x", "y"), data=[[1, 0, 5], [np.nan, 0, 0]])
        >>> mask_A = rasters.where(A > 3, 0, 1, A, new_name="mask_A")
        <xarray.DataArray 'mask_A' (x: 2, y: 3)>
        array([[ 1.,  1.,  0.],
               [nan,  1.,  1.]])
        Dimensions without coordinates: x, y
    """
    # Enforce condition
    where_xda = xr.where(cond, if_true, if_false)

    if master_xda is not None:
        # Convert to master dtype
        if where_xda.dtype != master_xda.dtype:
            where_xda = where_xda.astype(master_xda.dtype)

        # Convert to datarray if needed
        if not isinstance(where_xda, xr.DataArray):
            where_xda = master_xda.copy(data=where_xda)

        # Set nodata to nan
        where_xda = where_xda.where(~np.isnan(master_xda))

        # Set mtd
        where_xda = set_metadata(where_xda, master_xda, new_name=new_name)

    return where_xda


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def hillshade(
    xds: AnyRasterType, azimuth: float = 315, altitude: float = 45, **kwargs
) -> AnyXrDataStructure:
    """
    Compute the hillshade of a DEM from an azimuth and zenith angle (in degrees).

    Goal: replace `gdaldem CLI <https://gdal.org/programs/gdaldem.html>`_

    NB: altitude = 90 - zenith

    Args:
        xds (AnyRasterType): Path to the DEM, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        azimuth (float): Azimuth of the light, in degrees. 0 if it comes from the top of the raster, 90 from the east, ...
        zenith (float): Zenith angle in degrees (90 is the top, zenith = 90 - altitude)

    Returns:
        AnyXrDataStructure: Hillshade
    """
    if "zenith" in kwargs:
        logs.deprecation_warning(
            "'zenith' is deprecated. Use 'altitude' instead (altitude = 90-zenith)."
        )
        altitude = 90 - kwargs.pop("zenith")

    import xdem  # noqa

    hillshade = xds.dem.hillshade(azimuth=azimuth, altitude=altitude, **kwargs)

    hillshade = hillshade.rename(kwargs.get("name", "hillshade"))
    hillshade.attrs["long_name"] = "hillshade"

    return hillshade


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def slope(
    xds: AnyRasterType, in_pct: bool = False, in_rad: bool = False, **kwargs
) -> AnyXrDataStructure:
    """
    Compute the slope of a DEM (in degrees).

    Goal: replace `gdaldem CLI <https://gdal.org/programs/gdaldem.html>`_

    Args:
        xds (AnyRasterType): Path to the DEM, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        in_pct (bool): Outputs slope in percents
        in_rad (bool): Outputs slope in radians. Not taken into account if :code:`in_pct == True`

    Returns:
        AnyXrDataStructure: Slope
    """
    import xdem  # noqa

    slope = xds.dem.slope(degrees=not in_rad, **kwargs)

    if in_pct:
        if not in_rad:
            slope *= DEG_2_RAD
        slope = 100 * np.tan(xds)

    slope = slope.rename(kwargs.get("name", "slope"))
    slope.attrs["long_name"] = "slope"

    return slope


# Is lazy
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def aspect(xds: AnyRasterType, **kwargs) -> AnyXrDataStructure:
    """
    Compute the aspect of a DEM.

    Args:
        xds (AnyRasterType): Path to the DEM, its dataset, its :code:`xarray` or a tuple containing its array and metadata

    Returns:
        AnyXrDataStructure: Aspect
    """
    import xdem  # noqa

    aspect = xds.dem.aspect(**kwargs)

    aspect = aspect.rename(kwargs.get("name", "aspect"))
    aspect.attrs["long_name"] = "aspect"

    return aspect


# TODO: add other DEM-related functions like 'curvature', etc if needed. Create a dedicated module?


def from_deg_to_meters(
    deg: float, lat: float = None, decimals: float = 2, average_lat_lon: bool = False
) -> float:
    """
    Approximately convert a distance in degrees to a distance in meters.

    Only true at the Equator if lat is not given.
    Very false for longitude distance elsewhere.

    If lat is given

    See https://wiki.openstreetmap.org/wiki/Precision_of_coordinates

    Args:
        deg (float): Distance in degrees
        lat (float): Latitude (useful if the distance is longitudinal)
        decimals (float): To round the distance, as the precision can be poor here
        average_lat_lon (bool): Average between lat distance and lon distance

    Returns:
        float: Distance in meters
    """
    dist = float(deg) * DEG_LAT_TO_M

    if lat is not None:
        dist_lon = abs(dist * np.cos(lat * DEG_2_RAD))
        dist = (dist + dist_lon) / 2 if average_lat_lon else dist_lon

    return np.round(dist, decimals)


def from_meters_to_deg(
    meters: float, lat: float = None, decimals: float = 7, average_lat_lon: bool = False
) -> float:
    """
    Approximately convert a distance in meters to a distance in degrees. Only true at the Equator.
    Very false for longitude distance elsewhere.

    See https://wiki.openstreetmap.org/wiki/Precision_of_coordinates

    Args:
        meters (float): Distance in meters
        lat (float): Latitude (useful if the distance is longitudinal)
        decimals (float): To round the distance, as the precision can be poor here. 7 decimals is equivalent to a centimetric precision
        average_lat_lon (bool): Average between lat distance and lon distance

    Returns:
        float: Distance in degrees
    """
    dist = float(meters) / DEG_LAT_TO_M

    if lat is not None:
        dist_lon = abs(dist / np.cos(lat * DEG_2_RAD))
        dist = (dist + dist_lon) / 2 if average_lat_lon else dist_lon

    return np.round(dist, decimals)


# Is lazy
@dask.guard_compute(max_computes=0)
def classify(
    raster: xr.DataArray,
    bins: list,
    values: list,
    right: bool = False,
    new_name: str = None,
) -> xr.DataArray:
    """
    Classify a raster according to the given bin edges and its corresponding values.

    Example of usecase: classifying a fire severity.

    Args:
        raster (xr.DataArray): Raster to classify
        bins (list): Bin edges. Should be monotonic.
        values (list): Values to set for each bin. There should be one value more than the number of bins.
        right (bool): Indicating whether the intervals include the right or the left bin edge. Default behavior is (right==False) indicating that the interval does not include the right edge. The left bin end is open in this case, i.e., bins[i-1] <= x < bins[i] is the default behavior for monotonically increasing bins.
        new_name (str): New name to be set

    Returns:
        xr.DataArray: Classified raster

    Examples:
        >>> dnbr = rasters.read(r"dNBR.tif")
        >>> fire_severity = classify(dnbr, [0.27, 0.66], [2, 3, 4])
    """
    assert len(values) == len(bins) + 1, (
        "The number of values should equal the number of bins plus one."
    )

    assert None not in bins, f"'None' cannot be a bin: {bins}"

    if dask.is_chunked(raster):
        import dask.array as da

        # Get the classes values
        if not isinstance(raster, da.Array):
            values = da.from_array(values, chunks="auto")

        # Get the digitized indexes according to input bins
        arr = values.vindex[
            xr.apply_ufunc(np.digitize, raster, bins, right, dask="allowed").data
        ]
    else:
        # Get the classes values
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        # Get the digitized indexes according to input bins
        arr = values[xr.apply_ufunc(np.digitize, raster, bins, right)]

    # Create DataArray and set nodata
    classified_raster = raster.copy(data=arr).where(~np.isnan(raster))

    # Assign new name (if existing)
    if new_name is not None:
        classified_raster = classified_raster.rename(new_name)
        classified_raster.attrs["long_name"] = new_name
    return classified_raster


# Is lazy (even if DEM is cached?)
@dask.guard_compute(max_computes=0)
@any_raster_to_xr_ds
def reproject(
    src_xda: AnyRasterType,
    pixel_size: float = None,
    shape: tuple = None,
    resampling: Resampling = Resampling.bilinear,
    dst_crs: CRS = None,
    # dst_transform: Affine = None,
    # gcps: control.GroundControlPoint = None,
    nodata: float = None,
    num_threads: int = None,
    name: str = None,
    # under this, arguments only for reprojecting with RPCs
    # rpcs: rpc.RPC = None,
    dem_path: str = None,
    extent: gpd.GeoDataFrame = None,
    vcrs: str | int = None,
    caching_folder: AnyPathStrType = None,
    **kwargs,
) -> xr.DataArray:
    """
    Function handling all cases of reprojection, with and without RPCS, with and without dask or chunked arrays.

    Wrappers around:
    - odc.geo.xr_reproject for chunked arrays (in a dask-backed way)
    - rioxarray.reprjoect for xarray data
    - rasterio.reproject as a fallback (especially for Python 3.9)

    .. WARNING::
        RPC management is much more complex and necessits more arguments.

        RPC reprojection is not daskified yet. See this `issue<https://github.com/opendatacube/odc-geo/issues/193>`_.

        Cloud-stored DEM cannot be used as is. They will be cached and then use. Performance can be affected.

        Orthorectifying data with RPC is using a DEM. This DEM should have its height based on the ellispoid.
        If your DEM doesn't have a vertical CRS set, please provide it with vcrs.

        Note that in some cases, the vertical CRS is set by default.
        This library recognizes files with :code:`Copernicus` or :code:`COPDEM` in it (based on :code:`EGM08`).
        :code:`xdem` may recognize other files. See `the documentation<https://xdem.readthedocs.io/en/stable/vertical_ref.html>`_ for more insights.

    Args:
        src_xda (AnyRasterType): Raster to be reprojected
        pixel_size (float): Destination pixel size (in the destination CRS units). If not provided, the pixel size will be deducted from other parameters.
        shape (tuple): Destination shape (height, width). If not provided, the shape will be deducted from other parameters. Supersedes pixel_size.
        resampling (Resampling): Resampling method. Defaults to Resampling.bilinear.
        dst_crs (CRS): Destination CRS. If not provided, the CRS will be retrieved from the source raster.
        dst_transform (Affine): Destination transfgorm. If not provided, the transform will be computed from other parameters.
        gcps ((control.GroundControlPoint, CRS)): Ground Control Points
        nodata (float): Nodata value for the destination raster. If not provided, the nodata value will be retrieved from the source raster.
        num_threads (int): Number of threads to use for parallel processing. If not provided, set to your number of CPU minus 2.
        use_dask (bool): Flag to use dask for parallel processing. If not provided, the flag will be set to None and dask used if the data is chunked.
        ortho_path (AnyPathStrType): Path to the orthorectified array
        name (str): Name of the output DataArray.
        rpcs (rpc.RPC): RPC to orthorectify some raster
        dem_path (AnyPathStrType): Path to the DEM, only used if rpcs is given.
        extent (gpd.GeoDataFrame): GeoDataFrame containing extent of the raster to extract the DEM, only used if rpcs is given.
        vcrs (str | int): Vertical CRS of the DEM, only used if rpcs is given. Can be automatically deducted in some cases (COPDEM, etc.). Better to set it tu be sure.
        caching_folder (AnyPathStrType): Folder where to cache temporary files. If not provided, a temporary folder will be created and deleted in the end of the process. Only used if rpcs is given.
        **kwargs: Other arguments to pass to reproject or write

    Returns:
        xr.DataArray: Reprojected raster

    Examples:
        Example with RPCs
        >>> # Imports
        >>> import logging
        >>> from sertit import AnyPath, logs, rasters, vectors
        >>> logs.init_logger(logging.getLogger("sertit"))
        >>>
        >>> # Data
        >>> spot_path = AnyPath("IMG_SPOT7_MS_001_A", "DIM_SPOT7_MS_201602150257025_SEN_1671661101.XML")
        >>> copdem_path = AnyPath("Copernicus_DSM_10_S07_00_E106_00_DEM.tif")
        >>>
        >>> # Reproject
        >>> with rasterio.open(spot_path) as ds:
        >>>     rpc_reproj = rasters.reproject(spot_path, rpcs=ds.rpcs, dem_path=copdem_path, dst_crs="epsg:4326", nodata=0)
        [WARNING] - Reprojection with RPCs doesn't work with Dask. Computing the raster.
        [DEBUG] - Orthorectifying data with Copernicus_DSM_10_S07_00_E106_00_DEM.tif
        >>>
        >>> # See if we have a CRS
        >>> rpc_reproj.rio.crs.to_epsg()
        4326
        >>>
        >>> # See if the corrdinates are correctly set (lon = 106,x, lat = -6,x)
        >>> # These coordinates correspond to the tile of the DEM in input (S07_00_E106_00)
        >>> rpc_reproj.coords
        Coordinates:
          * x            (x) float64 40kB 106.6 106.6 106.6 106.6 ... 107.0 107.0 107.0
          * y            (y) float64 46kB -6.0 -6.0 -6.0 -6.0 ... -6.421 -6.421 -6.421
          * band         (band) int64 32B 1 2 3 4
            spatial_ref  int64 8B 0

        Example of very simple reprojections
        >>> # Reproject to change the pixel size of the raster
        >>> rasters.reproject(xda, pixel_size=60)

        >>> # Reproject to a specific shape
        >>> rasters.reproject(xda, shape=(161, 232))

        >>> # Reproject to a specific CRS
        >>> rasters.reproject(xda, dst_crs="EPSG:4326")
    """

    # Get num threads
    if num_threads is None:
        num_threads = perf.get_max_cores()

    tmp_caching_folder = None
    transformer_options = None
    if src_xda.rst.rpcs is not None:
        # Convert to Paths
        dem_path = AnyPath(dem_path)

        # Manage caching folder if needed
        if caching_folder is not None:
            caching_folder = AnyPath(caching_folder)
        else:
            tmp_caching_folder = TemporaryDirectory()
            caching_folder = AnyPath(tmp_caching_folder.name)

        # Get src crs (check that)
        src_crs = src_xda.rio.crs
        if not src_crs:
            # RPCs are always in 4326 by convention
            # https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-with-other-georeferencing-metadata
            src_crs = EPSG_4326

            # CRS has to be set for rioxarray
            src_xda.rio.write_crs(src_crs, inplace=True)

        # RPC_DEM doesn't work with cloud-based DEM
        dem_path = __preprocess_dem(dem_path, extent, caching_folder, vcrs, **kwargs)

        # Update GDAL parameters
        transformer_options = __rpc_gdal_options(dem_path)

    reprojected_xda = src_xda.rst.reproject(
        crs=dst_crs,
        res=pixel_size,
        grid_size=shape,
        nodata=nodata,
        resampling=resampling,
        transformer_options=transformer_options,
        silent=False,
        n_threads=num_threads,
        memory_limit=64,
    )

    # Rename DataArray
    if name:
        reprojected_xda = reprojected_xda.rename(name)
        reprojected_xda.attrs["long_name"] = name

    if tmp_caching_folder is not None:
        tmp_caching_folder.cleanup()

    return reprojected_xda


def __preprocess_dem(
    dem_path: AnyPathType,
    src_extent: gpd.GeoDataFrame,
    caching_folder: AnyPathType,
    vcrs=str | int,
    **kwargs,
) -> AnyPathType:
    """
    # Caching #
    RPC_DEM doesn't work with cloud-based DEM
    Read it to the extent of the product and save it on disk

    # Adjusting #
    Make sure DEM's height is relative to an ellipsoid
    https://stereopipeline.readthedocs.io/en/latest/next_steps.html#conv-to-ellipsoid
    A DEM relative to a geoid/areoid must be converted so that its heights are relative to an ellipsoid. This must be done for any Copernicus and SRTM DEMs.

    Args:
        dem_path (AnyPathType): DEM path
        src_extent (gpd.GeoDataFrame): Source extent
        caching_folder (AnyPathType): Folder where to cache the DEM
        vcrs (str | int): Vertical CRS to use. "Ellipsoid", "EGM08", "EGM96" or EPSG code

    Returns:
        AnyPathType: Path to the cached DEM
    """
    LOGGER.debug(f"Orthorectifying data with {dem_path}")

    # -- Manage vertical CRS --
    ellipsoid_dem_path = caching_folder / f"{dem_path.stem}_ellipsoid{dem_path.suffix}"
    if not ellipsoid_dem_path.is_file():
        windowed_dem_path = (
            caching_folder / f"{dem_path.stem}_windowed{dem_path.suffix}"
        )
        if not windowed_dem_path.is_file():
            LOGGER.debug("Subset the DEM to the extent of your data")
            write(
                # Don't chunk on reading if it's to write on disk directly
                read(dem_path, window=src_extent, chunks=None),
                windowed_dem_path,
                dtype=np.float32,
                driver="GTiff",
            )
        else:
            windowed_dem_path = dem_path

        import xdem

        LOGGER.debug("Creating new DEM object with input vertical CRS")
        dem = xdem.DEM(str(windowed_dem_path), vcrs=vcrs)

        # Set vertical CRS
        if dem.vcrs is None:
            # Try to automatically detect DEM VCRS (if we got here, it's that the automatic procedure coming from xdem has failed)
            dem_name = path.get_filename(dem_path)

            # COPDEM30 is largely used in Sertit
            if "Copernicus" in dem_name or "COPDEM" in dem_name:
                dem.set_vcrs("EGM08")

        if dem.vcrs is None:  # pragma: no cover
            LOGGER.warning(
                "Impossible to detect vertical CRS of your DEM. Orthorectification may be inaccurate. If needed, pass 'vcrs' to the function to adjust."
            )
            ellipsoid_dem_path = windowed_dem_path
        else:
            # Convert to ellipsoid vertical CRS
            LOGGER.debug(
                "Convert the DEM to Ellipsoid vertical coordinate reference system."
            )
            dem.to_vcrs("Ellipsoid", inplace=True)

            # Save back on disk
            dem.to_file(ellipsoid_dem_path)

    # Cache DEM in the end only if needed (i.e. not needed if vertically reprojected)
    if path.is_cloud_path(ellipsoid_dem_path):
        cached_dem_path = caching_folder / ellipsoid_dem_path.name
        if not cached_dem_path.is_file():
            LOGGER.warning(
                "gdalwarp cannot process DEM stored on cloud with 'RPC_DEM' argument, "
                "hence cloud-stored DEM cannot be used with non-orthorectified DIMAP data. "
                f"(DEM: {dem_path}). "
                "The DEM will be cached before the operation."
            )

            write(
                # Don't chunk on reading if it's to write on disk directly
                read(ellipsoid_dem_path, window=src_extent, chunks=None),
                cached_dem_path,
                dtype=np.float32,
                driver="GTiff",
            )

            LOGGER.debug("DEM cached.")
        ellipsoid_dem_path = cached_dem_path

    return ellipsoid_dem_path


def __rpc_gdal_options(dem_path: AnyPathStrType):
    """
    Add RPC-specific option to GDAL kwargs (topp be passed to rioxarray of rasterio)

    Args:
        dem_path  (AnyPathStrType): DEM path
        kwargs (dict): Reprojection kwargs

    Returns:
        dict: Updated options
    """
    # ------- DEM VALIDITY -------
    # DEM needs to have correct datum (WSG84 ellipsoid and not a geoid datum) -> how to check that?
    # COPDEM-30 is not OK!
    # https://xdem.readthedocs.io/en/stable/vertical_ref.html

    # Set RPC options
    transformer_options = {
        "RPC_DEM": str(dem_path),
        "BIGTIFF": "IF_NEEDED",
    }
    # https://gis.stackexchange.com/questions/328366/gdalwarp-orthorectification-worldview-3-not-using-rpc-projection-properly (says it needs 'error threshold = 0')
    # It seems fixed by default for RPC, i.e. from gdalwarp: https://gdal.org/en/stable/programs/gdalwarp.html#cmdoption-gdalwarp-et
    # Error threshold for transformation approximation, expressed as a number of source pixels.
    # Defaults to 0.125 pixels unless the RPC_DEM transformer option is specified, in which case an exact transformer, i.e. err_threshold=0, will be used.
    # We are using RPC_DEM so it should be fine

    # DEM nodata
    import rasterio

    with rasterio.open(dem_path) as src:
        dem_nodata = src.nodata

    if dem_nodata is not None:
        transformer_options.update({"RPC_DEM_MISSING_VALUE": dem_nodata})

    # OSR_USE_ETMERC:
    # Warning 1: OSR_USE_ETMERC is a legacy configuration option, which now has only effect when set to NO (YES is the default). Use OSR_USE_APPROX_TMERC=YES instead
    # kwargs.update(
    #     {
    #         "OSR_USE_ETMERC": "YES",
    #     }
    # )

    # Other threads found on RPC reprojection
    # https://gis.stackexchange.com/questions/456358/run-gdalwarp-to-rectify-a-raster-rpc-embedded-with-dem-but-got-error-too-many

    return transformer_options
