# -*- coding: utf-8 -*-
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
Raster tools

You can use this only if you have installed sertit[full] or sertit[rasters]
"""
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Polygon

try:
    import rasterio
    import rioxarray
    from rasterio import features
    from rasterio.enums import Resampling
    from rioxarray.exceptions import MissingCRS
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "Please install 'rioxarray' to use the 'rasters' package."
    ) from ex

from sertit import geometry, logs, path, rasters_rio, vectors
from sertit.types import AnyPathStrType, AnyPathType, AnyXrDataStructure

MAX_CORES = rasters_rio.MAX_CORES
PATH_XARR_DS = Union[str, AnyXrDataStructure, rasterio.DatasetReader]
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


def get_nodata_value(dtype) -> int:
    """
    Get default nodata value:

    Args:
        dtype: Dtype for the wanted nodata. Best if numpy's dtype.

    Returns:
        int: Nodata value

    Examples:
        >>> rasters.get_nodata_value("uint8")
        255

        >>> rasters.get_nodata_value("uint16")
        65535

        >>> rasters.get_nodata_value("int8")
        -128

        >>> rasters.get_nodata_value("float")
        -9999
    """
    return rasters_rio.get_nodata_value(dtype)


def path_xarr_dst(function: Callable) -> Callable:
    """
    Path, :code:`xarray.Dataset`, :code:`xarray.DataArray`,or dataset decorator. Allows a function to ingest:

    - a path
    - a :code:`xarray`
    - a :code:`rasterio` dataset

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
    def path_or_xarr_or_dst_wrapper(path_or_ds: PATH_XARR_DS, *args, **kwargs) -> Any:
        """
        Path or dataset wrapper
        Args:
            path_or_ds (PATH_XARR_DS): Raster path or its dataset
            *args: args
            **kwargs: kwargs

        Returns:
            Any: regular output
        """
        if isinstance(path_or_ds, xr.DataArray):
            out = function(path_or_ds, *args, **kwargs)
        elif isinstance(path_or_ds, xr.Dataset):
            # Try on the whole dataset
            try:
                out = function(path_or_ds, *args, **kwargs)
            except Exception:
                # Try on every dataarray
                try:
                    xds_dict = {}
                    convert_to_xdataset = False
                    for var in path_or_ds.data_vars:
                        xds_dict[var] = function(path_or_ds[var], *args, **kwargs)
                        if isinstance(xds_dict[var], xr.DataArray):
                            convert_to_xdataset = True

                    # Convert in dataset if we have dataarrays, else keep the dict
                    if convert_to_xdataset:
                        xds = xr.Dataset(xds_dict)
                    else:
                        xds = xds_dict
                    return xds
                except Exception as ex:
                    raise TypeError("Function not available for xarray.Dataset") from ex
        else:
            # Get name
            if path.is_path(path_or_ds):
                name = str(path_or_ds)
                path_or_ds = str(path_or_ds)
            else:
                name = path_or_ds.name

            with rioxarray.open_rasterio(
                path_or_ds,
                masked=True,
                default_name=name,
                chunks=kwargs.pop("chunks", True),
            ) as xds:
                out = function(xds, *args, **kwargs)
        return out

    return path_or_xarr_or_dst_wrapper


def path_arr_dst(function: Callable) -> Callable:
    """
    Path, :code:`xarray`, (array, metadata) or dataset decorator.
    Allows a function to ingest:

    - a path
    - a :code:`xarray`
    - a :code:`rasterio` dataset
    - :code:`rasterio` open data: (array, meta)

    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function

    Example:
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
    def path_or_arr_or_dst_wrapper(
        path_or_arr_or_ds: Union[str, rasterio.DatasetReader], *args, **kwargs
    ) -> Any:
        """
        Path or dataset wrapper
        Args:
            path_or_arr_or_ds (Union[str, rasterio.DatasetReader]): Raster path or its dataset
            *args: args
            **kwargs: kwargs

        Returns:
            Any: regular output
        """
        try:
            out = function(path_or_arr_or_ds, *args, **kwargs)
        except Exception as ex:
            if path.is_path(path_or_arr_or_ds):
                with rasterio.open(str(path_or_arr_or_ds)) as ds:
                    out = function(ds, *args, **kwargs)
            elif isinstance(path_or_arr_or_ds, tuple):
                arr, meta = path_or_arr_or_ds
                from rasterio import MemoryFile

                with MemoryFile() as memfile:
                    with memfile.open(
                        **meta, BIGTIFF=rasters_rio.bigtiff_value(arr)
                    ) as ds:
                        ds.write(arr)
                        out = function(ds, *args, **kwargs)
            else:
                # Try if xarray is importable
                try:
                    if isinstance(path_or_arr_or_ds, (xr.DataArray, xr.Dataset)):
                        file_path = path_or_arr_or_ds.encoding["source"]
                        with rasterio.open(file_path) as ds:
                            out = function(ds, *args, **kwargs)
                    else:
                        raise ex
                except Exception:
                    raise ex
        return out

    return path_or_arr_or_dst_wrapper


@path_xarr_dst
def get_nodata_mask(xds: AnyXrDataStructure) -> np.ndarray:
    """
    .. deprecated:: 1.36.0
       Use :py:func:`rasters.get_data_mask` instead.
    """
    logs.deprecation_warning("This function is deprecated. Use 'get_data_mask' instead")
    return get_data_mask(xds)


@path_xarr_dst
def get_data_mask(xds: AnyXrDataStructure) -> np.ndarray:
    """
    Get nodata mask from a xarray.

    .. WARNING::
        Sets 1 where the data is valid and 0 where it is not!

    Args:
        xds (AnyXrDataStructure): Array to evaluate

    Returns:
        np.ndarray: Pixelwise nodata array

    Example:
        >>> diag_arr = xr.DataArray(data=np.diag([1, 2, 3]))
        >>> diag_arr.rio.write_nodata(0, inplace=True)
        <xarray.DataArray (dim_0: 3, dim_1: 3)>
        array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])
        Dimensions without coordinates: dim_0, dim_1
        >>>
        >>> # Get the data mask from this array
        >>> get_data_mask(diag_arr)
        array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]], dtype=uint8)

    """

    nodata = xds.rio.nodata

    try:
        is_nan = np.isnan(nodata)
    except TypeError:
        is_nan = False

    if is_nan:
        nodata_pos = np.isnan(xds.data)
    else:
        nodata_pos = xds.data == nodata

    return np.where(nodata_pos, 0, 1).astype(np.uint8)


@path_xarr_dst
def rasterize(
    xds: PATH_XARR_DS,
    vector: Union[gpd.GeoDataFrame, AnyPathStrType],
    value_field: str = None,
    default_nodata: int = 0,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Rasterize a vector into raster format.

    Note that passing :code:`merge_alg = MergeAlg.add` will add the vector values to the given raster

    See: https://pygis.io/docs/e_raster_rasterize.html

    Use :code:`value_field` to create a raster with multiple values. If set to :code:`None`, the rtaster will be binary.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray, used as base for the vector's rasterization data (shape, etc.)
        vector (Union[gpd.GeoDataFrame, AnyPathStrType]): Vector to be rasterized
        value_field (str): Field of the vector with the values to be burnt on the raster (should be scalars). If let to None, the raster will be binary.
        default_nodata (int): Default nodata of the raster (outside the vector in the raster extent)

    Returns:
        AnyXrDataStructure: Rasterized vector

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> vec_path = "path/to/vector.shp"
        >>> rasterize(raster_path, vec_path, value_field="classes")
    """
    # Use classic option
    arr, meta = rasters_rio.rasterize(
        xds, vector, value_field, default_nodata, **kwargs
    )
    if len(arr.shape) != 3:
        arr = np.expand_dims(arr, axis=0)

    # Change nodata
    rasterized_xds = xds.copy(data=arr)
    rasterized_xds = set_nodata(rasterized_xds, nodata_val=meta["nodata"])
    return rasterized_xds


@path_xarr_dst
def _vectorize(
    xds: PATH_XARR_DS,
    values: Union[None, int, list] = None,
    keep_values: bool = True,
    dissolve: bool = False,
    get_nodata: bool = False,
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
        - You will get a classified polygon with data (value=0)/nodata pixels. To

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        get_nodata (bool): Get nodata vector (raster values are set to 0, nodata values are the other ones)
        default_nodata (int): Default values for nodata in case of non existing in file

    Returns:
        gpd.GeoDataFrame: Vector with the raster values (if dissolve is not set)
    """
    # Manage nodata value
    uint8_nodata = 255
    if xds.rio.encoded_nodata is not None:
        nodata = uint8_nodata
    else:
        nodata = default_nodata

    if get_nodata:
        data = get_data_mask(xds)
        nodata_arr = None
    else:
        xds_uint8 = xds.fillna(uint8_nodata)
        data = xds_uint8.data.astype(np.uint8)

        # Manage values
        if values is not None:
            if not isinstance(values, list):
                values = [values]

            # If we want a dissolved vector, just set 1instead of real values
            arr_vals = 1 if dissolve else data
            if keep_values:
                true = arr_vals
                false = nodata
            else:
                true = nodata
                false = arr_vals

            # Update data array
            data = np.where(np.isin(data, values), true, false).astype(np.uint8)

        # Get nodata array
        nodata_arr = rasters_rio.get_data_mask(
            data, has_nodata=False, default_nodata=nodata
        )

        if data.dtype != np.uint8:
            raise TypeError("Your data should be classified (np.uint8).")

    # WARNING: features.shapes do NOT accept dask arrays !
    if not isinstance(data, (np.ndarray, np.ma.masked_array)):
        data = data.compute()
    if nodata_arr is not None and not isinstance(
        nodata_arr, (np.ndarray, np.ma.masked_array)
    ):
        nodata_arr = nodata_arr.compute()

    # Get shapes (on array or on mask to get nodata vector)
    shapes = features.shapes(data, mask=nodata_arr, transform=xds.rio.transform())

    # Convert to geodataframe
    gdf = vectors.shapes_to_gdf(shapes, xds.rio.crs)

    # Return valid geometries
    gdf = geometry.make_valid(gdf)

    # Dissolve if needed
    if dissolve:
        gdf = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs).dissolve()

    return gdf


@path_xarr_dst
def vectorize(
    xds: PATH_XARR_DS,
    values: Union[None, int, list] = None,
    keep_values: bool = True,
    dissolve: bool = False,
    default_nodata: int = 0,
) -> gpd.GeoDataFrame:
    """
    Vectorize a :code:`xarray` to get the class vectors.

    If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value,
    with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.

    .. WARNING::
        - Your data is casted by force into np.uint8, so be sure that your data is classified.
        - This could take a while as the computing time directly depends on the number of polygons to vectorize.

        Please be careful.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Classes Vector

    Example:
        >>> raster_path = "path/to/raster.tif"
        >>> vec1 = vectorize(raster_path)
        >>>
        >>> # or
        >>> with rasterio.open(raster_path) as ds:
        >>>     vec2 = vectorize(ds)
        >>>
        >>> # Assert those two approaches give the same result
        >>> vec1 == vec2
        True
    """
    return _vectorize(
        xds,
        values=values,
        keep_values=keep_values,
        dissolve=dissolve,
        get_nodata=False,
        default_nodata=default_nodata,
    )


@path_xarr_dst
def get_valid_vector(xds: PATH_XARR_DS, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get the valid data of a raster, returned as a vector.

    Pay attention that every nodata pixel will appear too.
    If you want only the footprint of the raster, please use :py:func:`rasters.get_footprint`.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
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
    nodata = _vectorize(
        xds, values=None, get_nodata=True, default_nodata=default_nodata
    )
    return nodata[
        nodata.raster_val != 0
    ]  # 0 is the values of not nodata put there by rasterio


@path_xarr_dst
def get_nodata_vector(
    ds: rasters_rio.PATH_ARR_DS, default_nodata: int = 0
) -> gpd.GeoDataFrame:
    """
    Get the nodata vector of a raster as a vector.

    Pay attention that every nodata pixel will appear too.
    If you want only the footprint of the raster, please use :py:func:`rasters.get_footprint`.

    Args:
        ds (PATH_ARR_DS): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        default_nodata (int): Default values for nodata in case of non-existing in file

    Returns:
        gpd.GeoDataFrame: Nodata Vector

    Examples:
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
    return nodata[nodata.raster_val == 0]


@path_xarr_dst
def mask(
    xds: PATH_XARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Masking a dataset:
    setting nodata outside of the given shapes, but without cropping the raster to the shapes extent.

    The original nodata is kept and completed with the nodata provided by the shapes.

    Overload of rasterio mask function in order to create a :code:`xarray`.

    The :code:`mask` function docs can be seen `here <https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html>`_.
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a :code:`GeoDataFrame` is passed, in which case it will automatically be converted)
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

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
    # Use classic option
    arr, meta = rasters_rio.mask(xds, shapes=shapes, nodata=nodata, **kwargs)

    masked_xds = xds.copy(data=arr)

    if nodata:
        masked_xds = set_nodata(masked_xds, nodata)

    # Convert back to xarray
    return masked_xds


@path_xarr_dst
def paint(
    xds: PATH_XARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
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
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
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
    if xds.rio.encoded_nodata is not None:
        xds_fill = xds.fillna(xds.rio.encoded_nodata)
    elif xds.rio.nodata is not None:
        xds_fill = xds.fillna(xds.rio.nodata)
    else:
        xds_fill = xds

    # Use classic option
    arr, meta = rasters_rio.mask(
        xds_fill, shapes=shapes, nodata=value, invert=not invert, **kwargs
    )

    # Create and fill na values created by the mask to the wanted value
    painted_xds = xds.copy(data=arr)
    painted_xds = painted_xds.fillna(value)

    # Keep all attrs after fillna
    painted_xds.rio.update_attrs(xds.attrs, inplace=True)
    painted_xds.rio.update_encoding(xds.encoding, inplace=True)

    # Set back nodata to keep the original nodata
    if xds.rio.encoded_nodata is not None:
        painted_xds = set_nodata(painted_xds, xds.rio.encoded_nodata)

    # Convert back to xarray
    return painted_xds


@path_xarr_dst
def crop(
    xds: PATH_XARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Cropping a dataset:
    setting nodata outside the given shapes AND cropping the raster to the shapes extent.

    Overload of
    `rioxarray.clip <https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.clip>`_
    function in order to create a masked_array.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
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
    if nodata:
        xds = set_nodata(xds, nodata)

    if isinstance(shapes, (gpd.GeoDataFrame, gpd.GeoSeries)):
        shapes = shapes.to_crs(xds.rio.crs).geometry

    if "from_disk" not in kwargs:
        kwargs["from_disk"] = True  # WAY FASTER

    # Clip keeps encoding and attrs
    return xds.rio.clip(shapes, **kwargs)


@path_arr_dst
def read(
    ds: rasters_rio.PATH_ARR_DS,
    resolution: Union[tuple, list, float] = None,
    size: Union[tuple, list] = None,
    window: Any = None,
    resampling: Resampling = Resampling.nearest,
    masked: bool = True,
    indexes: Union[int, list] = None,
    chunks: Union[int, tuple, dict] = "auto",
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

    Uses `rioxarray.open_rasterio <https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray-open-rasterio>`_.
    For Dask usage, you can look at the
    `rioxarray tutorial <https://corteva.github.io/rioxarray/stable/examples/dask_read_write.html>`_.

    Args:
        ds (PATH_ARR_DS): Path to the raster or a rasterio dataset or a xarray
        resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided.
        window (Any): Anything that can be returned as a window (i.e. path, gpd.GeoPandas, Iterable, rasterio.Window...).
            In case of an iterable, assumption is made it corresponds to geographic bounds.
            For pixel, please provide a rasterio.Window directly.
        resampling (Resampling): Resampling method
        masked (bool): Get a masked array
        indexes (Union[int, list]): Indexes of the band to load. Load the whole array if None. Starts at 1 like GDAL.
        chunks (int, tuple or dict): Chunk sizes along each dimension, e.g., 5, (5, 5) or {'x': 5, 'y': 5}.
            If chunks is provided, it used to load the new DataArray into a dask array.
            Chunks can also be set to True or "auto" to choose sensible chunk sizes
            according to dask.config.get("array.chunk-size").
        as_type (Any): Type in which to load the array
        **kwargs: Optional keyword arguments to pass into rioxarray.open_rasterio().

    Returns:
        Union[AnyXrDataStructure]: Masked xarray corresponding to the raster data and its metadata

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
    if window is not None:
        window = rasters_rio.get_window(ds, window)

    # Get new height and width
    new_height, new_width, do_resampling = rasters_rio.get_new_shape(
        ds, resolution, size, window
    )

    # Read data (and load it to discard lock)
    with xr.set_options(keep_attrs=True):
        with rioxarray.set_options(export_grid_mapping=False):
            with rioxarray.open_rasterio(
                ds, default_name=path.get_filename(ds.name), chunks=chunks, **kwargs
            ) as xda:
                orig_dtype = xda.dtype

                # Windows
                if window is not None:
                    xda = xda.rio.isel_window(window).load()

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
                            f"Non available index: {[idx for i, idx in enumerate(indexes) if not ok_indexes[i]]} for {ds.name}"
                        )

                    xda = xda.isel(
                        band=[idx - 1 for ok, idx in zip(ok_indexes, indexes) if ok]
                    )

                    try:
                        # Set new long name: Bands nb are idx + 1
                        xda.long_name = tuple(
                            name
                            for i, name in enumerate(xda.long_name)
                            if i + 1 in indexes
                        )
                    except AttributeError:
                        pass

                # Manage resampling
                if do_resampling:
                    factor_h = xda.rio.height / new_height
                    factor_w = xda.rio.width / new_width

                    # Manage 2 ways of resampling, coarsen being faster than reprojection
                    # TODO: find a way to match rasterio's speed
                    if factor_h.is_integer() and factor_w.is_integer():
                        xda = xda.coarsen(x=int(factor_w), y=int(factor_h)).mean()
                    else:
                        xda = xda.rio.reproject(
                            xda.rio.crs,
                            shape=(new_height, new_width),
                            resampling=resampling,
                        )

                # Convert to wanted type
                if as_type:
                    # Modify the type as wanted by the user
                    # TODO: manage nodata and uint/int numbers
                    xda = xda.astype(as_type)

                # Mask if necessary
                if masked:
                    # Set nodata not in opening due to some performance issues
                    xda = set_nodata(xda, ds.meta["nodata"])

                # Set original dtype
                xda.encoding["dtype"] = orig_dtype

    return xda


@path_xarr_dst
def write(
    xds: AnyXrDataStructure, path: AnyPathStrType, tags: dict = None, **kwargs
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

    Args:
        xds (AnyXrDataStructure): Path to the raster or a rasterio dataset or a xarray
        path (AnyPathStrType): Path where to save it (directories should be existing)
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
    # Manage dtype
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
    else:
        dtype = xds.dtype

    if isinstance(dtype, str):
        # Convert to numpy dtype
        dtype = getattr(np, dtype)
    xds.encoding["dtype"] = dtype

    # Write nodata
    xds.rio.write_nodata(
        kwargs.pop("nodata", get_nodata_value(dtype)), encoded=True, inplace=True
    )

    # WORKAROUND: Pop _FillValue attribute (if existing)
    if "_FillValue" in xds.attrs:
        xds.attrs.pop("_FillValue")

    # Default compression to LZW
    if "compress" not in kwargs:
        kwargs["compress"] = "lzw"

    if (
        kwargs["compress"].lower() in ["lzw", "deflate", "zstd"]
        and "predictor" not in kwargs  # noqa: W503
    ):
        if xds.encoding["dtype"] in [np.float32, np.float64, float]:
            kwargs["predictor"] = "3"
        else:
            kwargs["predictor"] = "2"

    # Bigtiff if needed
    bigtiff = rasters_rio.bigtiff_value(xds)

    # Manage tiles
    if "tiled" not in kwargs:
        kwargs["tiled"] = True

    # Force GTiff
    kwargs["driver"] = kwargs.get("driver", "GTiff")

    # Write on disk
    xds.rio.to_raster(
        str(path), BIGTIFF=bigtiff, NUM_THREADS=MAX_CORES, tags=tags, **kwargs
    )


def collocate(
    reference: AnyXrDataStructure,
    other: AnyXrDataStructure,
    resampling: Resampling = Resampling.nearest,
    **kwargs,
) -> AnyXrDataStructure:
    """
    Collocate two georeferenced arrays:
    forces the *other* raster to be exactly georeferenced onto the *reference* raster by reprojection.

    Args:
        reference (AnyXrDataStructure): Reference xarray
        other (AnyXrDataStructure): Other xarray
        resampling (Resampling): Resampling method

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
        old_dtype = other.dtype
        collocated_xds = (
            other.astype(reference.dtype)
            .rio.reproject_match(reference, resampling=resampling)
            .astype(old_dtype)
        )
    else:
        collocated_xds = other.rio.reproject_match(
            reference, resampling=resampling, **kwargs
        )

    # Bug for now, tiny difference in coords
    collocated_xds = collocated_xds.assign_coords(
        {
            "x": reference.x,
            "y": reference.y,
        }
    )

    # Set back attributes and encoding
    collocated_xds.rio.update_attrs(other.attrs, inplace=True)
    collocated_xds.rio.update_encoding(other.encoding, inplace=True)

    return collocated_xds


@path_xarr_dst
def sieve(
    xds: PATH_XARR_DS, sieve_thresh: int, connectivity: int = 4
) -> AnyXrDataStructure:
    """
    Sieving, overloads rasterio function with raster shaped like an image: :code:`(1, h, w)`.

    .. WARNING::
        Your data is casted by force into :code:`np.uint8`, so be sure that your data is classified.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
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
    assert connectivity in [4, 8]

    # Use this trick to make the sieve work
    mask = np.squeeze(np.where(np.isnan(xds.data), 0, 1).astype(np.uint8))
    data = np.squeeze(xds.data.astype(np.uint8))

    # Sieve
    try:
        sieved_arr = features.sieve(
            data, size=sieve_thresh, connectivity=connectivity, mask=mask
        )
    except TypeError:
        # Manage dask arrays that fails with rasterio sieve
        sieved_arr = features.sieve(
            data.compute(),
            size=sieve_thresh,
            connectivity=connectivity,
            mask=mask.compute(),
        )

    # Set back nodata and expand back dim
    sieved_arr = sieved_arr.astype(xds.dtype)
    try:
        sieved_arr[np.isnan(np.squeeze(xds.data))] = np.nan
    except ValueError:
        # Manage integer files
        pass
    sieved_arr = np.expand_dims(sieved_arr, axis=0)
    sieved_xds = xds.copy(data=sieved_arr)

    return sieved_xds


def get_dim_img_path(dim_path: AnyPathStrType, img_name: str = "*") -> AnyPathType:
    """
    Get the image path from a :code:`BEAM-DIMAP` data.

    A :code:`BEAM-DIMAP` file cannot be opened by rasterio, although its :code:`.img` file can.

    Args:
        dim_path (AnyPathStrType): DIM path (.dim or .data)
        img_name (str): .img file name (or regex), in case there are multiple .img files (i.e. for S3 data)

    Returns:
        AnyPathType: .img file

    Example:
        >>> dim_path = "path/to/dimap.dim"  # BEAM-DIMAP image
        >>> img_path = get_dim_img_path(dim_path)
        >>>
        >>> # Read raster
        >>> raster, meta = read(img_path)
    """
    return rasters_rio.get_dim_img_path(dim_path, img_name)


@path_xarr_dst
def get_extent(xds: PATH_XARR_DS) -> gpd.GeoDataFrame:
    """
    Get the extent of a raster as a :code:`geopandas.Geodataframe`.

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray

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
    return vectors.get_geodf(geom=[*xds.rio.bounds()], crs=xds.rio.crs)


@path_xarr_dst
def get_footprint(xds: PATH_XARR_DS) -> gpd.GeoDataFrame:
    """
    Get real footprint of the product (without nodata, *in french == emprise utile*)

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
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


def merge_vrt(
    crs_paths: list,
    crs_merged_path: AnyPathStrType,
    abs_path: bool = False,
    **kwargs,
) -> None:
    """
    Merge rasters as a VRT. Uses :code:`gdalbuildvrt`.

    See here: https://gdal.org/programs/gdalbuildvrt.html

    Creates VRT with relative paths !

    This function handles files of different projection by create intermediate VRT used for warping (longer to open).

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


def merge_gtiff(crs_paths: list, crs_merged_path: AnyPathStrType, **kwargs) -> None:
    """
    Merge rasters as a GeoTiff.

    .. WARNING::
        They should have the same CRS otherwise the mosaic will be false !

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
    return rasters_rio.merge_gtiff(crs_paths, crs_merged_path, **kwargs)


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


def read_bit_array(
    bit_mask: Union[xr.DataArray, np.ndarray], bit_id: Union[list, int]
) -> Union[np.ndarray, list]:
    """
    Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number :code:`bit_id`)

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
        >>>
        >>> # Get the 2nd bit array
        >>> read_bit_array(bit_array, 2)
        array([[0, 0, 0],
               [1, 0, 0],
               [1, 0, 0]], dtype=uint8)
    """
    if isinstance(bit_mask, xr.DataArray):
        bit_mask = bit_mask.data

    return rasters_rio.read_bit_array(bit_mask, bit_id)


def read_uint8_array(
    bit_mask: Union[xr.DataArray, np.ndarray], bit_id: Union[list, int]
) -> Union[np.ndarray, list]:
    """
    Read 8 bit arrays as a succession of binary masks.

    Forces array to :code:`np.uint8`.

    See :py:func:`rasters.read_bit_array`.

    Args:
        bit_mask (np.ndarray): Bit array to read
        bit_id (int): Bit ID of the slice to be read
          Example: read the bit 0 of the mask as a cloud mask (Theia)

    Returns:
        Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given
    """
    return read_bit_array(bit_mask.astype(np.uint8), bit_id)


def set_metadata(
    naked_xda: xr.DataArray, mtd_xda: xr.DataArray, new_name=None
) -> xr.DataArray:
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
    try:
        naked_xda.rio.write_crs(mtd_xda.rio.crs, inplace=True)
    except MissingCRS:
        pass

    if new_name:
        naked_xda = naked_xda.rename(new_name)

    naked_xda.rio.update_attrs(mtd_xda.attrs, inplace=True)
    naked_xda.rio.update_encoding(mtd_xda.encoding, inplace=True)
    naked_xda.rio.set_nodata(mtd_xda.rio.nodata, inplace=True)

    return naked_xda


def set_nodata(xda: xr.DataArray, nodata_val: Union[float, int]) -> xr.DataArray:
    """
    Set nodata to a xarray that have no default nodata value.

    In the data array, the no data will be set to :code:`np.nan`.
    The encoded value can be retrieved with :code:`xda.rio.encoded_nodata`.

    Args:
        xda (xr.DataArray): DataArray
        nodata_val (Union[float, int]): Nodata value

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
    encoding = xda.encoding
    attrs = xda.attrs

    xda = xda.where(xda.data != nodata_val)
    xda.rio.write_nodata(nodata_val, encoded=True, inplace=True)

    # Set back attributes and encoding
    xda.rio.update_attrs(attrs, inplace=True)
    xda.rio.update_encoding(encoding, inplace=True)

    return xda


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


@path_xarr_dst
def hillshade(
    xds: PATH_XARR_DS, azimuth: float = 315, zenith: float = 45
) -> AnyXrDataStructure:
    """
    Compute the hillshade of a DEM from an azimuth and elevation angle (in degrees).

    Goal: replace `gdaldem CLI <https://gdal.org/programs/gdaldem.html>`_

    NB: altitude = zenith

    References:

    - `1 <https://www.neonscience.org/resources/learning-hub/tutorials/create-hillshade-py>`_
    - `2 <http://webhelp.esri.com/arcgisdesktop/9.2/index.cfm?TopicName=How%20Hillshade%20works>`_

    Args:
        xds (PATH_XARR_DS): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        azimuth (float): Azimuth angle in degrees
        zenith (float): Zenith angle in degrees

    Returns:
        AnyXrDataStructure: Hillshade
    """
    # Use classic option
    arr, meta = rasters_rio.hillshade(xds, azimuth=azimuth, zenith=zenith)

    return xds.copy(data=arr)


@path_xarr_dst
def slope(
    xds: PATH_XARR_DS, in_pct: bool = False, in_rad: bool = False
) -> AnyXrDataStructure:
    """
    Compute the slope of a DEM (in degrees).

    Goal: replace `gdaldem CLI <https://gdal.org/programs/gdaldem.html>`_

    Args:
        xds (PATH_XARR_DS): Path to the raster, its dataset, its :code:`xarray` or a tuple containing its array and metadata
        in_pct (bool): Outputs slope in percents
        in_rad (bool): Outputs slope in radians. Not taken into account if :code:`in_pct == True`

    Returns:
        AnyXrDataStructure: Slope
    """
    # Use classic option
    arr, meta = rasters_rio.slope(xds, in_pct=in_pct, in_rad=in_rad)

    return xds.copy(data=arr)
