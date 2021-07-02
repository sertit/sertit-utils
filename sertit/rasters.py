# -*- coding: utf-8 -*-
# Copyright 2021, SERTIT-ICube - France, https://sertit.unistra.fr/
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
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import xarray
from cloudpathlib import CloudPath
from rioxarray.exceptions import MissingCRS

from sertit.rasters_rio import PATH_ARR_DS, path_arr_dst

try:
    import geopandas as gpd
    import rasterio
    import rioxarray
    import xarray as xr
    from rasterio import features
    from rasterio.enums import Resampling
    from shapely.geometry import Polygon
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "Please install 'rioxarray' and 'geopandas' to use the 'rasters' package."
    ) from ex

from sertit import files, rasters_rio, vectors

MAX_CORES = os.cpu_count() - 2
PATH_XARR_DS = Union[str, xr.DataArray, xr.Dataset, rasterio.DatasetReader]
"""
Types:

- Path
- rasterio Dataset
- `xarray.DataArray` and `xarray.Dataset`
"""  # fmt:skip

XDS_TYPE = Union[xr.Dataset, xr.DataArray]
"""
Xarray types: xr.Dataset and xr.DataArray
"""  # fmt:skip


def path_xarr_dst(function: Callable) -> Callable:
    """
    Path, `xarray` or dataset decorator. Allows a function to ingest:

    - a path
    - a `xarray`
    - a `rasterio` dataset

    ```python
    >>> # Create mock function
    >>> @path_or_dst
    >>> def fct(dst):
    >>>     read(dst)
    >>>
    >>> # Test the two ways
    >>> read1 = fct("path\\to\\raster.tif")
    >>> with rasterio.open("path\\to\\raster.tif") as dst:
    >>>     read2 = fct(dst)
    >>>
    >>> # Test
    >>> read1 == read2
    True
    ```
    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
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
            if isinstance(path_or_ds, (str, Path, CloudPath)):
                name = str(path_or_ds)
                path_or_ds = str(path_or_ds)
            else:
                name = path_or_ds.name

            with rioxarray.open_rasterio(
                path_or_ds, masked=True, default_name=name
            ) as xds:
                out = function(xds, *args, **kwargs)
        return out

    return path_or_xarr_or_dst_wrapper


def to_np(xds: xarray.DataArray, dtype: Any = None) -> np.ndarray:
    """
    Convert the `xarray` to a `np.ndarray` with the correct nodata encoded.

    This is particularly useful when reading with `masked=True`.

    ```python
    >>> raster_path = "path\\to\\mask.tif"  # Classified raster in np.uint8 with nodata = 255
    >>> # We read with masked=True so the data is converted to float
    >>> xds = read(raster_path)
    <xarray.DataArray 'path/to/mask.tif' (band: 1, y: 322, x: 464)>
    [149408 values with dtype=float64]
    Coordinates:
      * band         (band) int32 1
      * y            (y) float64 4.798e+06 4.798e+06 ... 4.788e+06 4.788e+06
      * x            (x) float64 5.411e+05 5.411e+05 ... 5.549e+05 5.55e+05
        spatial_ref  int32 0
    >>> to_np(xds)  # Getting back np.uint8 and encoded nodata
    array([[[255, 255, 255, ..., 255, 255, 255],
        [255, 255, 255, ..., 255, 255, 255],
        [255, 255, 255, ..., 255, 255, 255],
        ...,
        [255, 255, 255, ...,   1, 255, 255],
        [255, 255, 255, ...,   1, 255, 255],
        [255, 255, 255, ...,   1, 255, 255]]], dtype=uint8)

    True
    ```
    Args:
        xds (xarray.DataArray): `xarray.DataArray` to convert
        dtype (Any): Dtype to convert to. If None, using the origin dtype if existing or its current dtype.

    Returns:

    """
    # Manage dtype
    if not dtype:
        dtype = xds.encoding.get("dtype", xds.dtype)

    # Manage nodata
    if xds.rio.encoded_nodata is not None:
        xds_fill = xds.fillna(xds.rio.encoded_nodata)
    else:
        xds_fill = xds

    # Cast to wanted dtype
    arr = xds_fill.data.astype(dtype)

    return arr


def get_nodata_mask(xds: XDS_TYPE) -> np.ndarray:
    """
    Get nodata mask from a xarray.

    ```python
    >>> diag_arr = xr.DataArray(data=np.diag([1, 2, 3]))
    >>> diag_arr.rio.write_nodata(0, inplace=True)
    <xarray.DataArray (dim_0: 3, dim_1: 3)>
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])
    Dimensions without coordinates: dim_0, dim_1

    >>> get_nodata_mask(diag_arr)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=uint8)
    ```

    Args:
        xds (XDS_TYPE): Array to evaluate

    Returns:
        np.ndarray: Pixelwise nodata array

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
        - If `get_nodata` is set to False:
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
    has_nodata = xds.rio.encoded_nodata is not None
    nodata = xds.rio.encoded_nodata if has_nodata else default_nodata

    if get_nodata:
        data = get_nodata_mask(xds)
        nodata_arr = None
    else:
        data = to_np(xds, dtype=np.uint8)
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

            data = np.where(np.isin(data, values), true, false).astype(np.uint8)

        if data.dtype != np.uint8:
            raise TypeError("Your data should be classified (np.uint8).")

        nodata_arr = rasters_rio.get_nodata_mask(
            data, has_nodata=False, default_nodata=nodata
        )

    # Get shapes (on array or on mask to get nodata vector)
    shapes = features.shapes(data, mask=nodata_arr, transform=xds.rio.transform())

    # Convert to geodataframe
    gdf = vectors.shapes_to_gdf(shapes, xds.rio.crs)

    # Dissolve if needed
    if dissolve:
        # Discard self-intersection and null geometries
        gdf = gdf.buffer(0)
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
    Vectorize a `xarray` to get the class vectors.

    If dissolved is False, it returns a GeoDataFrame with a GeoSeries per cluster of pixel value,
    with the value as an attribute. Else it returns a GeoDataFrame with a unique polygon.

    .. WARNING::
        - Your data is casted by force into np.uint8, so be sure that your data is classified.
        - This could take a while as the computing time directly depends on the number of polygons to vectorize.
            Please be careful.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> vec1 = vectorize(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     vec2 = vectorize(dst)
    >>> vec1 == vec2
    True
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Classes Vector
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
    Get the valid data of a raster as a vector.

    Pay attention that every nodata pixel will appear too.
    If you want only the footprint of the raster, please use `get_footprint`.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> nodata1 = get_nodata_vec(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     nodata2 = get_nodata_vec(dst)
    >>> nodata1 == nodata2
    True
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Nodata Vector

    """
    nodata = _vectorize(
        xds, values=None, get_nodata=True, default_nodata=default_nodata
    )
    return nodata[
        nodata.raster_val != 0
    ]  # 0 is the values of not nodata put there by rasterio


@path_xarr_dst
def get_nodata_vector(dst: PATH_ARR_DS, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get the nodata vector of a raster as a vector.

    Pay attention that every nodata pixel will appear too.
    If you want only the footprint of the raster, please use `get_footprint`.

    ```python
    >>> raster_path = "path\\to\\raster.tif"  # Classified raster, with no data set to 255
    >>> nodata1 = get_nodata_vec(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     nodata2 = get_nodata_vec(dst)
    >>> nodata1 == nodata2
    True
    ```

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Nodata Vector

    """
    nodata = _vectorize(
        dst, values=None, get_nodata=True, default_nodata=default_nodata
    )
    return nodata[nodata.raster_val == 0]


@path_xarr_dst
def mask(
    xds: PATH_XARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> XDS_TYPE:
    """
    Masking a dataset:
    setting nodata outside of the given shapes, but without cropping the raster to the shapes extent.

    The original nodata is kept and completed with the nodata provided by the shapes.

    Overload of rasterio mask function in order to create a `xarray`.

    The `mask` function docs can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> shape_path = "path\\to\\shapes.geojson"  # Any vector that geopandas can read
    >>> shapes = gpd.read_file(shape_path)
    >>> mask1 = mask(raster_path, shapes)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     mask2 = mask(dst, shapes)
    >>> mask1 == mask2
    True
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a `GeoDataFrame` is passed, in which case it will automatically be converted)
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         XDS_TYPE: Masked array as a xarray
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
) -> XDS_TYPE:
    """
    Painting a dataset: setting values inside the given shapes. To set outside the shape, set invert=True.
    Pay attention that this behavior is the opposite of the `rasterio.mask` function.

    The original nodata is kept.
    This means if your shapes intersects the original nodata,
    the value of the pixel will be set to nodata rather than to the wanted value.

    Overload of rasterio mask function in order to create a `xarray`.
    The `mask` function docs can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> shape_path = "path\\to\\shapes.geojson"  # Any vector that geopandas can read
    >>> shapes = gpd.read_file(shape_path)
    >>> paint1 = paint(raster_path, shapes, value=100)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     paint2 = paint(dst, shapes, value=100)
    >>> paint1 == paint2
    True
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a `GeoDataFrame` is passed, in which case it will automatically be converted)
        value (int): Value to set on the shapes.
        invert (bool): If invert is True, set value outside the shapes.
        **kwargs: Other rasterio.mask options

    Returns:
         XDS_TYPE: Painted array as a xarray
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
    setting nodata outside of the given shapes AND cropping the raster to the shapes extent.

    Overload of [`rioxarray`
    clip](https://corteva.github.io/rioxarray/stable/rioxarray.html#rioxarray.raster_array.RasterArray.clip)
    function in order to create a masked_array.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> shape_path = "path\\to\\shapes.geojson"  # Any vector that geopandas can read
    >>> shapes = gpd.read_file(shape_path)
    >>> xds2 = crop(raster_path, shapes)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     xds2 = crop(dst, shapes)
    >>> xds1 == xds2
    True
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a `GeoDataFrame` is passed, in which case it will automatically be converted)
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rioxarray.clip options

    Returns:
         XDS_TYPE: Cropped array as a xarray
    """
    if nodata:
        xds_new = xds.rio.write_nodata(nodata)
    else:
        xds_new = xds

    if isinstance(shapes, (gpd.GeoDataFrame, gpd.GeoSeries)):
        shapes = shapes.to_crs(xds.rio.crs).geometry

    if "from_disk" not in kwargs:
        kwargs["from_disk"] = True  # WAY FASTER

    return xds_new.rio.clip(shapes, **kwargs)


@path_arr_dst
def read(
    dst: PATH_ARR_DS,
    resolution: Union[tuple, list, float] = None,
    size: Union[tuple, list] = None,
    resampling: Resampling = Resampling.nearest,
    masked: bool = True,
    indexes: Union[int, list] = None,
) -> XDS_TYPE:
    """
    Read a raster dataset from a :

    - `xarray` (compatibility issues)
    - `rasterio.Dataset`
    - `rasterio` opened data (array, metadata)
    - a path.

    The resolution can be provided (in dataset unit) as:

    - a tuple or a list of (X, Y) resolutions
    - a float, in which case X resolution = Y resolution
    - None, in which case the dataset resolution will be used

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> xds1 = read(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>    xds2 = read(dst)
    >>> xds1 == xds2
    True
    ```

    Args:
        dst (PATH_ARR_DS): Path to the raster or a rasterio dataset or a xarray
        resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided.
        resampling (Resampling): Resampling method
        masked (bool): Get a masked array
        indexes (Union[int, list]): Indexes to load. Load the whole array if None.

    Returns:
        Union[XDS_TYPE]: Masked xarray corresponding to the raster data and its meta data

    """
    # Get new height and width
    new_height, new_width = rasters_rio.get_new_shape(dst, resolution, size)

    # Read data (and load it to discard lock)
    with xarray.set_options(keep_attrs=True):
        with rioxarray.open_rasterio(
            dst, default_name=files.get_filename(dst.name)
        ) as xda:
            orig_dtype = xda.dtype
            if indexes:
                if not isinstance(indexes, list):
                    indexes = [indexes]

                # Open only wanted bands
                xda = xda[np.isin(xda.band, indexes)]

                try:
                    # Set new long name: Bands nb are idx + 1
                    xda.long_name = tuple(
                        name for i, name in enumerate(xda.long_name) if i + 1 in indexes
                    )
                except AttributeError:
                    pass

            # Manage resampling
            if new_height != dst.height or new_width != dst.width:
                factor_h = dst.height / new_height
                factor_w = dst.width / new_width
                if factor_h.is_integer() and factor_w.is_integer():
                    xda = xda.coarsen(x=int(factor_w), y=int(factor_h)).mean()
                else:
                    xda = xda.rio.reproject(
                        xda.rio.crs,
                        shape=(new_height, new_width),
                        resampling=resampling,
                    )

            if masked:
                # Set nodata not in opening due to some performance issues
                xda = set_nodata(xda, dst.meta["nodata"])

            # Set original dtype
            xda.encoding["dtype"] = orig_dtype

    return xda


@path_xarr_dst
def write(xds: XDS_TYPE, path: Union[str, CloudPath, Path], **kwargs) -> None:
    """
    Write raster to disk.
    (encapsulation of `rasterio`'s function, because for now `rioxarray` to_raster doesn't work as expected)

    Metadata will be created with the `xarray` metadata (ie. width, height, count, type...)
    The driver is `GTiff` by default, and no nodata value is provided.
    The file will be compressed if the raster is a mask (saved as uint8).

    If not overwritten, sets the nodata according to `dtype`:

    - uint8: 255
    - int8: -128
    - uint16, uint32, int32, int64, uint64: 65535
    - int16, float32, float64, float128, float: -9999

    Compress with `LZW` option by default. To disable it, add the `compress=None` parameter.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> raster_out = "path\\to\\out.tif"

    >>> # Read raster
    >>> xds = read(raster_path)

    >>> # Rewrite it
    >>> write(xds, raster_out)
    ```

    Args:
        xds (XDS_TYPE): Path to the raster or a rasterio dataset or a xarray
        path (Union[str, CloudPath, Path]): Path where to save it (directories should be existing)
        **kwargs: Overloading metadata, ie `nodata=255` or `dtype=np.uint8`
    """
    if "nodata" in kwargs:
        xds.encoding["_FillValue"] = kwargs.pop("nodata")
    else:
        # Manage default nodata in function of dtype (default, for float = -9999)
        if "dtype" in kwargs:
            dtype = kwargs["dtype"]
        else:
            dtype = xds.dtype

        # Convert to numpy dtype
        if isinstance(dtype, str):
            dtype = getattr(np, dtype)

        if dtype == np.uint8:
            xds.encoding["_FillValue"] = 255
        elif dtype == np.int8:
            xds.encoding["_FillValue"] = -128
        elif dtype in [np.uint16, np.uint32, np.int32, np.int64, np.uint64, int]:
            xds.encoding["_FillValue"] = 65535
        elif dtype in [np.int16, np.float32, np.float64, float]:
            xds.encoding["_FillValue"] = -9999
        else:
            raise ValueError(
                f"Invalid dtype: {dtype}, should be convertible to numpy dtypes"
            )

    # Default compression to LZW
    if "compress" not in kwargs:
        kwargs["compress"] = "lzw"

    # WORKAROUND: Pop _FillValue attribute
    if "_FillValue" in xds.attrs:
        xds.attrs.pop("_FillValue")

    xds.rio.to_raster(str(path), BIGTIFF="IF_NEEDED", **kwargs)


def collocate(
    master_xds: XDS_TYPE,
    slave_xds: XDS_TYPE,
    resampling: Resampling = Resampling.nearest,
) -> XDS_TYPE:
    """
    Collocate two georeferenced arrays:
    forces the *slave* raster to be exactly georeferenced onto the *master* raster by reprojection.

    Use it like `OTB SuperImpose`.

    ```python
    >>> master_path = "path\\to\\master.tif"
    >>> slave_path = "path\\to\\slave.tif"
    >>> col_path = "path\\to\\collocated.tif"

    >>> # Collocate the slave to the master
    >>> col_xds = collocate(read(master_path), read(slave_path), Resampling.bilinear)

    >>> # Write it
    >>> write(col_xds, col_path)
    ```

    Args:
        master_xds (XDS_TYPE): Master xarray
        slave_xds (XDS_TYPE): Slave xarray
        resampling (Resampling): Resampling method

    Returns:
        XDS_TYPE: Collocated xarray

    """
    collocated_xds = slave_xds.rio.reproject_match(master_xds, resampling=resampling)
    collocated_xds = collocated_xds.assign_coords(
        {
            "x": master_xds.x,
            "y": master_xds.y,
        }
    )  # Bug for now, tiny difference in coords
    return collocated_xds


@path_xarr_dst
def sieve(
    xds: PATH_XARR_DS, sieve_thresh: int, connectivity: int = 4, dtype=np.uint8
) -> XDS_TYPE:
    """
    Sieving, overloads rasterio function with raster shaped like (1, h, w).

    .. WARNING::
        Your data is casted by force into `np.uint8`, so be sure that your data is classified.

    ```python
    >>> raster_path = "path\\to\\raster.tif"  # classified raster

    >>> # Rewrite it
    >>> sieved_xds = sieve(raster_path, sieve_thresh=20)

    >>> # Write it
    >>> raster_out = "path\\to\\raster_sieved.tif"
    >>> write(sieved_xds, raster_out)
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
        sieve_thresh (int): Sieving threshold in pixels
        connectivity (int): Connectivity, either 4 or 8
        dtype: Dtype of the xarray
            (if nodata is set, the xds.dtype is float whereas the values are meant to be ie in np.uint8)

    Returns:
        (XDS_TYPE): Sieved xarray
    """
    assert connectivity in [4, 8]

    # Use this trick to make the sieve work
    data = np.squeeze(to_np(xds, dtype))

    # Sieve
    sieved_arr = features.sieve(data, size=sieve_thresh, connectivity=connectivity)

    # Create back the xarray
    sieved_arr = np.expand_dims(sieved_arr.astype(xds.dtype), axis=0)
    sieved_xds = xds.copy(data=sieved_arr)

    # Set back nodata
    if xds.rio.encoded_nodata is not None:
        sieved_xds = set_nodata(sieved_xds, xds.rio.encoded_nodata)

    return sieved_xds


def get_dim_img_path(
    dim_path: Union[str, CloudPath, Path], img_name: str = "*"
) -> Union[CloudPath, Path]:
    """
    Get the image path from a *BEAM-DIMAP* data.

    A *BEAM-DIMAP* file cannot be opened by rasterio, although its .img file can.

    ```python
    >>> dim_path = "path\\to\\dimap.dim"  # BEAM-DIMAP image
    >>> img_path = get_dim_img_path(dim_path)

    >>> # Read raster
    >>> raster, meta = read(img_path)
    ```

    Args:
        dim_path (Union[str, CloudPath, Path]): DIM path (.dim or .data)
        img_name (str): .img file name (or regex), in case there are multiple .img files (ie. for S3 data)

    Returns:
        Union[CloudPath, Path]: .img file
    """
    return rasters_rio.get_dim_img_path(dim_path, img_name)


@path_xarr_dst
def get_extent(xds: PATH_XARR_DS) -> gpd.GeoDataFrame:
    """
    Get the extent of a raster as a `geopandas.Geodataframe`.

    ```python
    >>> raster_path = "path\\to\\raster.tif"

    >>> extent1 = get_extent(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     extent2 = get_extent(dst)
    >>> extent1 == extent2
    True
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray

    Returns:
        gpd.GeoDataFrame: Extent as a `geopandas.Geodataframe`
    """
    return vectors.get_geodf(geometry=[*xds.rio.bounds()], crs=xds.rio.crs)


@path_xarr_dst
def get_footprint(xds: PATH_XARR_DS) -> gpd.GeoDataFrame:
    """
    Get real footprint of the product (without nodata, in french == emprise utile)

    ```python
    >>> raster_path = "path\\to\\raster.tif"

    >>> footprint1 = get_footprint(raster_path)

    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     footprint2 = get_footprint(dst)
    >>> footprint1 == footprint2
    ```

    Args:
        xds (PATH_XARR_DS): Path to the raster or a rasterio dataset or a xarray
    Returns:
        gpd.GeoDataFrame: Footprint as a GeoDataFrame
    """
    footprint = get_valid_vector(xds)
    return vectors.get_wider_exterior(footprint)


def merge_vrt(
    crs_paths: list, crs_merged_path: Union[str, CloudPath, Path], **kwargs
) -> None:
    """
    Merge rasters as a VRT. Uses `gdalbuildvrt`.

    See here: https://gdal.org/programs/gdalbuildvrt.html

    Creates VRT with relative paths !

    .. WARNING::
        They should have the same CRS otherwise the mosaic will be false !

    ```python
    >>> paths_utm32630 = ["path\\to\\raster1.tif", "path\\to\\raster2.tif", "path\\to\\raster3.tif"]
    >>> paths_utm32631 = ["path\\to\\raster4.tif", "path\\to\\raster5.tif"]

    >>> mosaic_32630 = "path\\to\\mosaic_32630.vrt"
    >>> mosaic_32631 = "path\\to\\mosaic_32631.vrt"

    >>> # Create mosaic, one by CRS !
    >>> merge_vrt(paths_utm32630, mosaic_32630)
    >>> merge_vrt(paths_utm32631, mosaic_32631, {"-srcnodata":255, "-vrtnodata":0})
    ```

    Args:
        crs_paths (list): Path of the rasters to be merged with the same CRS
        crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster
        kwargs: Other gdlabuildvrt arguments
    """
    return rasters_rio.merge_vrt(crs_paths, crs_merged_path, **kwargs)


def merge_gtiff(
    crs_paths: list, crs_merged_path: Union[str, CloudPath, Path], **kwargs
) -> None:
    """
    Merge rasters as a GeoTiff.

    .. WARNING::
        They should have the same CRS otherwise the mosaic will be false !

    ```python
    >>> paths_utm32630 = ["path\\to\\raster1.tif", "path\\to\\raster2.tif", "path\\to\\raster3.tif"]
    >>> paths_utm32631 = ["path\\to\\raster4.tif", "path\\to\\raster5.tif"]

    >>> mosaic_32630 = "path\\to\\mosaic_32630.tif"
    >>> mosaic_32631 = "path\\to\\mosaic_32631.tif"

    # Create mosaic, one by CRS !
    >>> merge_gtiff(paths_utm32630, mosaic_32630)
    >>> merge_gtiff(paths_utm32631, mosaic_32631)
    ```

    Args:
        crs_paths (list): Path of the rasters to be merged with the same CRS
        crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster
        kwargs: Other rasterio.merge arguments
            More info [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html#rasterio.merge.merge)
    """
    return rasters_rio.merge_gtiff(crs_paths, crs_merged_path, **kwargs)


def unpackbits(array: np.ndarray, nof_bits: int) -> np.ndarray:
    """
    Function found here:
    https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types


    ```python
    >>> bit_array = np.random.randint(5, size=[3,3])
    array([[1, 1, 3],
           [4, 2, 0],
           [4, 3, 2]], dtype=uint8)

    # Unpack 8 bits (8*1, as itemsize of uint8 is 1)
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
    ```

    Args:
        array (np.ndarray): Array to unpack
        nof_bits (int): Number of bits to unpack

    Returns:
        np.ndarray: Unpacked array
    """
    return rasters_rio.unpackbits(array, nof_bits)


def read_bit_array(
    bit_mask: Union[xr.DataArray, np.ndarray], bit_id: Union[list, int]
) -> Union[np.ndarray, list]:
    """
    Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number bit_id)

    ```python
    >>> bit_array = np.random.randint(5, size=[3,3])
    array([[1, 1, 3],
           [4, 2, 0],
           [4, 3, 2]], dtype=uint8)

    # Get the 2nd bit array
    >>> read_bit_array(bit_array, 2)
    array([[0, 0, 0],
           [1, 0, 0],
           [1, 0, 0]], dtype=uint8)
    ```

    Args:
        bit_mask (np.ndarray): Bit array to read
        bit_id (int): Bit ID of the slice to be read
          Example: read the bit 0 of the mask as a cloud mask (Theia)

    Returns:
        Union[np.ndarray, list]: Binary mask or list of binary masks if a list of bit_id is given
    """
    if isinstance(bit_mask, xr.DataArray):
        bit_mask = bit_mask.data

    return rasters_rio.read_bit_array(bit_mask, bit_id)


def read_uint8_array(
    bit_mask: Union[xr.DataArray, np.ndarray], bit_id: Union[list, int]
) -> Union[np.ndarray, list]:
    """
    Read 8 bit arrays as a succession of binary masks.

    Forces array to `np.uint8`.

    See `read_bit_array`.

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
    Set metadata from a `xr.DataArray` to another (including `rioxarray` metadata such as encoded_nodata and crs).

    Useful when performing operations on xarray that result in metadata loss such as sums.

    ```python
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
    ```

    Args:
        naked_xda (xr.DataArray): DataArray to complete
        mtd_xda (xr.DataArray): DataArray with the correct metadata
        new_name (str): New name for naked DataArray

    Returns:
        xr.DataArray: Complete DataArray
    """
    try:
        naked_xda.rio.write_crs(mtd_xda.rio.crs, inplace=True)
    except MissingCRS:
        pass

    if new_name:
        naked_xda = naked_xda.rename(new_name)
    naked_xda.encoding = mtd_xda.encoding

    naked_xda.rio.update_attrs(mtd_xda.attrs, inplace=True)
    naked_xda.rio.set_nodata(mtd_xda.rio.nodata, inplace=True)

    return naked_xda


def set_nodata(xda: xr.DataArray, nodata_val: Union[float, int]) -> xr.DataArray:
    """
    Set nodata to a xarray that have no default nodata value.

    In the data array, the no data will be set to `np.nan`.
    The encoded value can be retrieved with `xda.rio.encoded_nodata`.

    ```python
    >>> A = xr.DataArray(dims=("x", "y"), data=np.zeros((3,3), dtype=np.uint8))
    >>> A[0, 0] = 1
    <xarray.DataArray (x: 3, y: 3)>
    array([[1, 0, 0],
           [0, 0, 0],
           [0, 0, 0]], dtype=uint8)
    Dimensions without coordinates: x, y

    >>> A_nodata = set_nodata(A, 0)
    <xarray.DataArray (x: 3, y: 3)>
    array([[ 1., nan, nan],
           [nan, nan, nan],
           [nan, nan, nan]])
    Dimensions without coordinates: x, y
    ```

    Args:
        xda (xr.DataArray): DataArray
        nodata_val (Union[float, int]): Nodata value

    Returns:
        xr.DataArray: DataArray with nodata set
    """
    xda_nodata = xda.where(xda.data != nodata_val)
    xda_nodata.encoding = xda.encoding
    xda_nodata.rio.update_attrs(xda.attrs, inplace=True)
    xda_nodata.rio.write_nodata(nodata_val, encoded=True, inplace=True)
    return xda_nodata


def where(
    cond, if_true, if_false, master_xda: xr.DataArray = None, new_name: str = ""
) -> xr.DataArray:
    """
    Overloads `xr.where` with:

    - setting metadata of `master_xda`
    - preserving the nodata pixels of the `master_xda`

    If `master_xda` is None, use it like `xr.where`.
    Else, it outputs a `xarray.DataArray` with the same dtype than `master_xda`.

    .. WARNING::
        If you don't give a `master_xda`,
        it is better to pass numpy arrays to `if_false` and `if_true` keywords
        as passing xarrays interfers with the output metadata (you may lose the CRS and so on).
        Just pass `if_true=true_xda.data` inplace of `if_true=true_xda` and the same for `if_false`

    ```python
    >>> A = xr.DataArray(dims=("x", "y"), data=[[1, 0, 5], [np.nan, 0, 0]])
    >>> mask_A = rasters.where(A > 3, 0, 1, A, new_name="mask_A")
    <xarray.DataArray 'mask_A' (x: 2, y: 3)>
    array([[ 1.,  1.,  0.],
           [nan,  1.,  1.]])
    Dimensions without coordinates: x, y
    ```
    Args:
        cond (scalar, array, Variable, DataArray or Dataset): Conditional array
        if_true (scalar, array, Variable, DataArray or Dataset): What to do if `cond` is True
        if_false (scalar, array, Variable, DataArray or Dataset):  What to do if `cond` is False
        master_xda: Master `xr.DataArray` used to set the metadata and the nodata
        new_name (str): New name of the array

    Returns:
        xr.DataArray: Where array with correct mtd and nodata pixels
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
