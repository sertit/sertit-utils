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

You can use this only if you have installed sertit[full] or sertit[rasters_rio]
"""
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from cloudpathlib import AnyPath, CloudPath

try:
    import geopandas as gpd
    import rasterio
    from rasterio import MemoryFile, features
    from rasterio import mask as rmask
    from rasterio import merge, warp
    from rasterio.enums import Resampling
    from shapely.geometry import Polygon
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "Please install 'rasterio' and 'geopandas' to use the 'rasters_rio' package."
    ) from ex

from sertit import files, misc, strings, vectors

MAX_CORES = os.cpu_count() - 2
PATH_ARR_DS = Union[str, tuple, rasterio.DatasetReader]
"""
Types:

- Path
- Rasterio open data: (array, meta)
- rasterio Dataset
- `xarray`
"""  # fmt:skip


def path_arr_dst(function: Callable) -> Callable:
    """
    Path, `xarray`, (array, metadata) or dataset decorator.
    Allows a function to ingest:

    - a path
    - a `xarray`
    - a `rasterio` dataset
    - `rasterio` open data: (array, meta)

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
        if isinstance(path_or_arr_or_ds, (str, Path, CloudPath)):
            with rasterio.open(str(path_or_arr_or_ds)) as dst:
                out = function(dst, *args, **kwargs)
        elif isinstance(path_or_arr_or_ds, tuple):
            arr, meta = path_or_arr_or_ds
            with MemoryFile() as memfile:
                with memfile.open(**meta) as dst:
                    dst.write(arr)
                    out = function(dst, *args, **kwargs)
        else:
            out = None
            # Try if xarray is importable
            try:
                import xarray as xr

                if isinstance(path_or_arr_or_ds, (xr.DataArray, xr.Dataset)):
                    meta = {
                        "driver": "GTiff",
                        "dtype": path_or_arr_or_ds.dtype,
                        "nodata": path_or_arr_or_ds.rio.encoded_nodata,
                        "width": path_or_arr_or_ds.rio.width,
                        "height": path_or_arr_or_ds.rio.height,
                        "count": path_or_arr_or_ds.rio.count,
                        "crs": path_or_arr_or_ds.rio.crs,
                        "transform": path_or_arr_or_ds.rio.transform(),
                    }
                    with MemoryFile() as memfile:
                        with memfile.open(**meta) as dst:
                            xds = path_or_arr_or_ds.copy()
                            if xds.rio.encoded_nodata is not None:
                                xds = xds.fillna(xds.rio.encoded_nodata)
                            dst.write(xds.data)
                            out = function(dst, *args, **kwargs)
            except ModuleNotFoundError:
                pass

            if not out:
                out = function(path_or_arr_or_ds, *args, **kwargs)
        return out

    return path_or_arr_or_dst_wrapper


@path_arr_dst
def get_new_shape(
    dst: PATH_ARR_DS, resolution: Union[tuple, list, float], size: Union[tuple, list]
) -> (int, int):
    """
    Get the new shape (height, width) of a resampled raster.

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided.

    Returns:
        (int, int): Height, width

    """

    def _get_new_dim(dim: int, res_old: float, res_new: float) -> int:
        """
        Get the new dimension in pixels
        Args:
            dim (int): Old dimension
            res_old (float): Old resolution
            res_new (float): New resolution

        Returns:
            int: New dimension
        """
        return int(np.round(dim * res_old / res_new))

    # By default keep original shape
    new_height = dst.height
    new_width = dst.width

    # Compute new shape
    if resolution is not None:
        if isinstance(resolution, (int, float)):
            new_height = _get_new_dim(dst.height, dst.res[1], resolution)
            new_width = _get_new_dim(dst.width, dst.res[0], resolution)
        elif resolution is None:
            pass
        else:
            try:
                if len(resolution) != 2:
                    raise ValueError(
                        "We should have a resolution for X and Y dimensions"
                    )

                if resolution[0] is not None:
                    new_width = _get_new_dim(dst.width, dst.res[0], resolution[0])

                if resolution[1] is not None:
                    new_height = _get_new_dim(dst.height, dst.res[1], resolution[1])
            except (TypeError, KeyError):
                raise ValueError(
                    f"Resolution should be None, 2 floats or a castable to a list: {resolution}"
                )
    elif size is not None:
        try:
            new_height = size[1]
            new_width = size[0]
        except (TypeError, KeyError):
            raise ValueError(f"Size should be None or a castable to a list: {size}")

    return new_height, new_width


def update_meta(arr: Union[np.ndarray, np.ma.masked_array], meta: dict) -> dict:
    """
    Basic metadata update from a numpy array. Updates everything that we can find in the array:

    - `dtype`: array dtype,
    - `count`: first dimension of the array if the array is in 3D, else 1
    - `height`: second dimension of the array
    - `width`: third dimension of the array
    - `nodata`: if a masked array is given, nodata is its fill_value

    .. WARNING::
        The array's shape is interpreted in rasterio's way (count, height, width) !

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> with rasterio.open(raster_path) as dst:
    >>>      meta = dst.meta
    >>>      arr = dst.read()
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
    ```

    Args:
        arr (Union[np.ndarray, np.ma.masked_array]): Array from which to update the metadata
        meta (dict): Metadata to update

    Returns:
        dict: Update metadata

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
    array: Union[np.ma.masked_array, np.ndarray],
    has_nodata: bool,
    default_nodata: int = 0,
) -> np.ndarray:
    """
    Get nodata mask from a masked array.

    The nodata may not be set before, then pass a nodata value that will be evaluated on the array.

    ```python
    >>> diag_arr = np.diag([1,2,3])
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])

    >>> get_nodata_mask(diag_arr, has_nodata=False)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]], dtype=uint8)

    >>> get_nodata_mask(diag_arr, has_nodata=False, default_nodata=1)
    array([[0, 1, 1],
           [1, 1, 1],
           [1, 1, 1]], dtype=uint8)
    ```

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
        nodata_mask = np.where(array != default_nodata, 1, 0).astype(np.uint8)
    else:
        nodata_mask = np.where(array.mask, 0, 1).astype(np.uint8)

    return nodata_mask


@path_arr_dst
def _vectorize(
    dst: PATH_ARR_DS,
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
        If `get_nodata` is set to False:
            - Please only use this function on a classified raster.
            - This could take a while as the computing time directly depends on the number of polygons to vectorize.
                Please be careful.
        Else:
            - You will get a classified polygon with data (value=0)/nodata pixels.

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        get_nodata (bool): Get nodata vector (raster values are set to 0, nodata values are the other ones)
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Vector
    """
    # Get the shapes
    array = dst.read(masked=True)

    # Manage nodata value
    has_nodata = dst.nodata is not None
    nodata = dst.nodata if has_nodata else default_nodata

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
    nodata_mask = get_nodata_mask(data, has_nodata=False, default_nodata=nodata)

    # Get shapes (on array or on mask to get nodata vector)
    shapes = features.shapes(
        nodata_mask if get_nodata else data,
        mask=None if get_nodata else nodata_mask,
        transform=dst.transform,
    )

    # Convert to geodataframe
    gdf = vectors.shapes_to_gdf(shapes, dst.crs)

    # Dissolve if needed
    if dissolve:
        # Discard self-intersection and null geometries
        gdf = gdf.buffer(0)
        gdf = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs).dissolve()

    return gdf


@path_arr_dst
def vectorize(
    dst: PATH_ARR_DS,
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

    ```python
    >>> raster_path = "path\\to\\raster.tif"  # Classified raster, with no data set to 255
    >>> vec1 = vectorize(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     vec2 = vectorize(dst)
    >>> vec1 == vec2
    True
    ```

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        values (Union[None, int, list]): Get only the polygons concerning this/these particular values
        keep_values (bool): Keep the passed values. If False, discard them and keep the others.
        dissolve (bool): Dissolve all the polygons into one unique. Only works if values are given.
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Classes Vector
    """
    return _vectorize(
        dst,
        values=values,
        get_nodata=False,
        keep_values=keep_values,
        dissolve=dissolve,
        default_nodata=default_nodata,
    )


@path_arr_dst
def get_valid_vector(dst: PATH_ARR_DS, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get the valid data of a raster as a vector.

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
    return nodata[
        nodata.raster_val != 0
    ]  # 0 is the values of not nodata put there by rasterio


@path_arr_dst
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
    return nodata[
        nodata.raster_val == 0
    ]  # 0 is the values of not nodata put there by rasterio


@path_arr_dst
def _mask(
    dst: PATH_ARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    do_crop: bool = False,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Overload of rasterio mask function in order to create a masked_array.

    The `mask` function docs can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).

    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a `GeoDataFrame` is passed, in which case it will automatically be converted.
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        do_crop (bool): Whether to crop the raster to the extent of the shapes. Default is False.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, dict): Cropped array as a masked array and its metadata
    """
    if isinstance(shapes, (gpd.GeoDataFrame, gpd.GeoSeries)):
        shapes = shapes.to_crs(dst.crs).geometry
    elif not isinstance(shapes, list):
        shapes = [shapes]

    # Set nodata
    if not nodata:
        if dst.nodata:
            nodata = dst.nodata
        else:
            nodata = 0

    # Crop dataset
    msk, trf = rmask.mask(dst, shapes, nodata=nodata, crop=do_crop, **kwargs)

    # Create masked array
    nodata_mask = np.where(msk == nodata, 1, 0).astype(np.uint8)
    mask_array = np.ma.masked_array(msk, nodata_mask, fill_value=nodata)

    # Update meta
    out_meta = update_meta(mask_array, dst.meta)
    out_meta["transform"] = trf

    return mask_array, out_meta


@path_arr_dst
def mask(
    dst: PATH_ARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Masking a dataset:
    setting nodata outside of the given shapes, but without cropping the raster to the shapes extent.

    Overload of rasterio mask function in order to create a masked_array.

    The `mask` function docs can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> shape_path = "path\\to\\shapes.geojson"  # Any vector that geopandas can read
    >>> shapes = gpd.read_file(shape_path)
    >>> masked_raster1, meta1 = mask(raster_path, shapes)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     masked_raster2, meta2 = mask(dst, shapes)
    >>> masked_raster1 == masked_raster2
    True
    >>> meta1 == meta2
    True
    ```

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a `GeoDataFrame` is passed, in which case it will automatically be converted.
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, dict): Masked array as a masked array and its metadata
    """
    return _mask(dst, shapes=shapes, nodata=nodata, do_crop=False, **kwargs)


@path_arr_dst
def crop(
    dst: PATH_ARR_DS,
    shapes: Union[gpd.GeoDataFrame, Polygon, list],
    nodata: Optional[int] = None,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Cropping a dataset:
    setting nodata outside of the given shapes AND cropping the raster to the shapes extent.

    **HOW:**

    Overload of rasterio mask function in order to create a masked_array.

    The `mask` function docs can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> shape_path = "path\\to\\shapes.geojson"  # Any vector that geopandas can read
    >>> shapes = gpd.read_file(shape_path)
    >>> cropped_raster1, meta1 = crop(raster_path, shapes)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>     cropped_raster2, meta2 = crop(dst, shapes)
    >>> cropped_raster1 == cropped_raster2
    True
    >>> meta1 == meta2
    True
    ```

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        shapes (Union[gpd.GeoDataFrame, Polygon, list]): Shapes with the same CRS as the dataset
            (except if a `GeoDataFrame` is passed, in which case it will automatically be converted.
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, dict): Cropped array as a masked array and its metadata
    """
    return _mask(dst, shapes=shapes, nodata=nodata, do_crop=True, **kwargs)


@path_arr_dst
def read(
    dst: PATH_ARR_DS,
    resolution: Union[tuple, list, float] = None,
    size: Union[tuple, list] = None,
    resampling: Resampling = Resampling.nearest,
    masked: bool = True,
    **kwargs,
) -> (np.ma.masked_array, dict):
    """
    Read a raster dataset from a `rasterio.Dataset` or a path.

    The resolution can be provided (in dataset unit) as:

    - a tuple or a list of (X, Y) resolutions
    - a float, in which case X resolution = Y resolution
    - None, in which case the dataset resolution will be used

    Tip:
    Use index with a list of one element to keep a 3D array

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> raster1, meta1 = read(raster_path)
    >>> # or
    >>> with rasterio.open(raster_path) as dst:
    >>>    raster2, meta2 = read(dst)
    >>> raster1 == raster2
    True
    >>> meta1 == meta2
    True
    ```

    Args:
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
        resolution (Union[tuple, list, float]): Resolution of the wanted band, in dataset resolution unit (X, Y)
        size (Union[tuple, list]): Size of the array (width, height). Not used if resolution is provided.
        resampling (Resampling): Resampling method (nearest by default)
        masked (bool): Get a masked array, `True` by default (whereas it is False by default in rasterio)
        **kwargs: Other dst.read() arguments such as indexes.

    Returns:
        np.ma.masked_array, dict: Masked array corresponding to the raster data and its meta data

    """
    # Get new height and width
    new_height, new_width = get_new_shape(dst, resolution, size)

    # Manage out_shape
    if "indexes" in kwargs:
        if isinstance(kwargs["indexes"], int):
            out_shape = (new_height, new_width)
        else:
            out_shape = (len(kwargs["indexes"]), new_height, new_width)
    else:
        out_shape = (dst.count, new_height, new_width)

    # Read data
    array = dst.read(
        out_shape=out_shape,
        resampling=resampling,
        masked=masked,
        **kwargs,
    )

    # Update meta
    dst_transform = dst.transform * dst.transform.scale(
        (dst.width / new_width), (dst.height / new_height)
    )
    dst_meta = dst.meta.copy()
    dst_meta.update(
        {
            "height": new_height,
            "width": new_width,
            "transform": dst_transform,
            "dtype": array.dtype,
            "nodata": dst.nodata,
        }
    )

    return array, dst_meta


def write(
    raster: Union[np.ma.masked_array, np.ndarray],
    meta: dict,
    path: Union[str, CloudPath, Path],
    **kwargs,
) -> None:
    """
    Write raster to disk (encapsulation of rasterio's function)

    Metadata will be copied and updated with raster's information (ie. width, height, count, type...)
    The driver is GTiff by default, and no nodata value is provided.
    The file will be compressed if the raster is a mask (saved as uint8)

    ```python
    >>> raster_path = "path\\to\\raster.tif"
    >>> raster_out = "path\\to\\out.tif"

    >>> # Read raster
    >>> raster, meta = read(raster_path)

    >>> # Rewrite it on disk
    >>> write(raster, meta, raster_out)
    ```

    Args:
        raster (Union[np.ma.masked_array, np.ndarray]): Raster to save on disk
        meta (dict): Basic metadata that will be copied and updated with raster's information
        path (Union[str, CloudPath, Path]): Path where to save it (directories should be existing)
        **kwargs: Overloading metadata, ie `nodata=255`
    """
    # Manage raster type (impossible to write boolean arrays)
    if raster.dtype == bool:
        raster = raster.astype(np.uint8)

    # Update metadata
    out_meta = meta.copy()

    # Update raster to be sure to write down correct nodata pixels
    if isinstance(raster, np.ma.masked_array):
        raster[raster.mask] = raster.fill_value

    # Force compression and driver (but can be overwritten by kwargs)
    out_meta["driver"] = "GTiff"

    # Compress to LZW by default
    out_meta["compress"] = kwargs.get("compress", "lzw")

    # Bigtiff if needed
    out_meta["BIGTIFF"] = "IF_NEEDED"

    # Update metadata with array data
    out_meta = update_meta(raster, out_meta)

    # Update metadata with additional params
    for key, val in kwargs.items():
        out_meta[key] = val

    # Manage raster shape
    if len(raster.shape) == 2:
        raster = np.expand_dims(raster, axis=0)

    # Write product
    with rasterio.open(str(path), "w", **out_meta) as dst:
        dst.write(raster)


def collocate(
    master_meta: dict,
    slave_arr: Union[np.ma.masked_array, np.ndarray],
    slave_meta: dict,
    resampling: Resampling = Resampling.nearest,
) -> (Union[np.ma.masked_array, np.ndarray], dict):
    """
    Collocate two georeferenced arrays:
    forces the *slave* raster to be exactly georeferenced onto the *master* raster by reprojection.

    Use it like `OTB SuperImpose`.

    ```python
    >>> master_path = "path\\to\\master.tif"
    >>> slave_path = "path\\to\\slave.tif"
    >>> col_path = "path\\to\\collocated.tif"

    >>> # Just open the master data
    >>> with rasterio.open(master_path) as master_dst:
    >>>     # Read slave
    >>>     slave, slave_meta = read(slave_path)

    >>>     # Collocate the slave to the master
    >>>     col_arr, col_meta = collocate(master_dst.meta,
    >>>                                   slave,
    >>>                                   slave_meta,
    >>>                                   Resampling.bilinear)

    >>> # Write it
    >>> write(col_arr, col_path, col_meta)
    ```

    Args:
        master_meta (dict): Master metadata
        slave_arr (np.ma.masked_array): Slave array to be collocated
        slave_meta (dict): Slave metadata
        resampling (Resampling): Resampling method

    Returns:
        np.ma.masked_array, dict: Collocated array and its metadata

    """
    collocated_arr = np.zeros(
        (master_meta["count"], master_meta["height"], master_meta["width"]),
        dtype=master_meta["dtype"],
    )
    warp.reproject(
        source=slave_arr,
        destination=collocated_arr,
        src_transform=slave_meta["transform"],
        src_crs=slave_meta["crs"],
        dst_transform=master_meta["transform"],
        dst_crs=master_meta["crs"],
        src_nodata=slave_meta["nodata"],
        dst_nodata=slave_meta["nodata"],
        resampling=resampling,
        num_threads=MAX_CORES,
    )

    meta = master_meta.copy()
    meta.update(
        {
            "dtype": slave_meta["dtype"],
            "driver": slave_meta["driver"],
            "nodata": slave_meta["nodata"],
        }
    )

    return collocated_arr, meta


def sieve(
    array: Union[np.ma.masked_array, np.ndarray],
    out_meta: dict,
    sieve_thresh: int,
    connectivity: int = 4,
) -> (Union[np.ma.masked_array, np.ndarray], dict):
    """
    Sieving, overloads rasterio function with raster shaped like (1, h, w).

    Forces the output to `np.uint8` (as only classified rasters should be sieved)

    ```python
    >>> raster_path = "path\\to\\raster.tif"  # classified raster

    >>> # Read raster
    >>> raster, meta = read(raster_path)

    >>> # Rewrite it
    >>> sieved, sieved_meta = sieve(raster, meta, sieve_thresh=20)

    >>> # Write it
    >>> raster_out = "path\\to\\raster_sieved.tif"
    >>> write(sieved, raster_out, sieved_meta)
    ```

    Args:
        array (Union[np.ma.masked_array, np.ndarray]): Array to sieve
        out_meta (dict): Metadata to update
        sieve_thresh (int): Sieving threshold in pixels
        connectivity (int): Connectivity, either 4 or 8

    Returns:
        (Union[np.ma.masked_array, np.ndarray], dict): Sieved array and updated meta
    """
    assert connectivity in [4, 8]

    # Read extraction array
    expand = False
    if len(array.shape) == 3 and array.shape[0] == 1:
        array = np.squeeze(array)  # Use this trick to make the sieve work
        expand = True

    # Convert to np.uint8 if needed
    dtype = np.uint8
    meta = out_meta.copy()
    if meta["dtype"] != dtype:
        array = array.astype(dtype)
        meta["dtype"] = dtype

    # Sieve
    result_array = np.empty(array.shape, dtype=array.dtype)
    features.sieve(
        array, size=sieve_thresh, out=result_array, connectivity=connectivity
    )

    # Use this trick to get the array back to 'normal'
    if expand:
        result_array = np.expand_dims(result_array, axis=0)

    return result_array, meta


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
    dim_path = AnyPath(dim_path)
    if dim_path.suffix == ".dim":
        dim_path = dim_path.with_suffix(".data")

    assert dim_path.suffix == ".data" and dim_path.is_dir()

    return files.get_file_in_dir(dim_path, img_name, extension="img", exact_name=True)


@path_arr_dst
def get_extent(dst: PATH_ARR_DS) -> gpd.GeoDataFrame:
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
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata

    Returns:
        gpd.GeoDataFrame: Extent as a `geopandas.Geodataframe`
    """
    return vectors.get_geodf(geometry=[*dst.bounds], crs=dst.crs)


@path_arr_dst
def get_footprint(dst: PATH_ARR_DS) -> gpd.GeoDataFrame:
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
        dst (PATH_ARR_DS): Path to the raster, its dataset, its `xarray` or a tuple containing its array and metadata
    Returns:
        gpd.GeoDataFrame: Footprint as a GeoDataFrame
    """
    footprint = get_valid_vector(dst)

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
        crs_paths (list): Path of the rasters to be merged with the same CRS)
        crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster
        kwargs: Other gdlabuildvrt arguments
    """

    for idp, path in enumerate(crs_paths):
        crs_paths[idp] = path

    # Manage cloud paths (gdalbuildvrt needs url or true filepaths)
    crs_merged_path = AnyPath(crs_merged_path)
    if isinstance(crs_merged_path, CloudPath):
        crs_merged_path = AnyPath(crs_merged_path.fspath)

    for i, crs_path in enumerate(crs_paths):
        path = AnyPath(crs_path)
        if isinstance(path, CloudPath):
            path = AnyPath(path.fspath)
        crs_paths[i] = path

    # Create relative paths
    vrt_root = os.path.dirname(crs_merged_path)
    try:
        rel_paths = [
            strings.to_cmd_string(str(files.real_rel_path(path, vrt_root)))
            for path in crs_paths
        ]
        rel_vrt = strings.to_cmd_string(
            str(files.real_rel_path(crs_merged_path, vrt_root))
        )
    except ValueError:
        rel_paths = crs_paths
        rel_vrt = crs_merged_path

    # Run cmd
    arg_list = [val for item in kwargs.items() for val in item]
    vrt_cmd = ["gdalbuildvrt", rel_vrt, *rel_paths, *arg_list]
    misc.run_cli(vrt_cmd, cwd=vrt_root)


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
        crs_paths (list): Path of the rasters to be merged with the same CRS)
        crs_merged_path (Union[str, CloudPath, Path]): Path to the merged raster
        kwargs: Other rasterio.merge arguments
            More info [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html#rasterio.merge.merge)
    """
    # Open datasets for merging
    crs_datasets = []
    try:
        for path in crs_paths:
            crs_datasets.append(rasterio.open(str(path)))

        # Merge all datasets
        merged_array, merged_transform = merge.merge(crs_datasets, **kwargs)
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
        for dataset in crs_datasets:
            dataset.close()

    # Save merge datasets
    write(
        merged_array, crs_datasets[0].meta, crs_merged_path, transform=merged_transform
    )


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
    dtype = array.dtype
    if dtype == np.uint8:
        unpacked = np.unpackbits(
            np.expand_dims(array, axis=-1), axis=-1, count=nof_bits, bitorder="little"
        )
    else:
        xshape = list(array.shape)
        array = array.reshape([-1, 1])
        msk = 2 ** np.arange(nof_bits, dtype=array.dtype).reshape([1, nof_bits])
        unpacked = (
            (array & msk).astype(bool).astype(np.uint8).reshape(xshape + [nof_bits])
        )

    return unpacked


def read_bit_array(
    bit_mask: np.ndarray, bit_id: Union[list, int]
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
