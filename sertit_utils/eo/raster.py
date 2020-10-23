""" Raster tools """
import os
from typing import Union, Optional
import affine
import numpy as np
import geopandas as gpd
from rasterio.enums import Resampling
from shapely.geometry import Polygon
import rasterio
from rasterio import features, warp, mask

from sertit_utils.core import file_utils

MAX_CORES = os.cpu_count() - 2


def vectorize(path: str, on_mask: bool = False, default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Vectorize a raster

    Args:
        path (str): Path to the raster
        on_mask (bool): Work only on mask (to get no data for instance)
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Vector
    """
    with rasterio.open(path) as dst:
        # Get the shapes
        array = dst.read(masked=True)

        # Nodata mask
        if dst.nodata is not None:  # Unspecified nodata is set to None by rasterio
            nodata_mask = np.where(array.mask, 0, 1).astype(np.uint8)
        else:
            nodata_mask = np.where(array != default_nodata, 1, 0).astype(np.uint8)

        # Get shapes (on array or on mask to get nodata vector)
        shapes = features.shapes(nodata_mask if on_mask else array.data,
                                 mask=None if on_mask else nodata_mask,
                                 transform=dst.transform)

        # Convert results to geopandas and save it
        # TODO: FutureWarning: Assigning CRS to a GeoDataFrame without a geometry column is now deprecated
        gpd_results = gpd.GeoDataFrame(shapes, columns=["geometry", "raster_val"], crs=dst.crs)
        if not gpd_results.empty:
            # Convert to proper polygons
            def to_polygons(val):
                """ Convert to polygon """
                # Donut cases
                if len(val["coordinates"]) > 1:
                    poly = Polygon(val["coordinates"][0], val["coordinates"][1:])
                else:
                    poly = Polygon(val["coordinates"][0])

                # Note: it doesn't check if polygons are valid or not !
                # If needed, do:
                # if not poly.is_valid:
                #   poly = poly.buffer(1.0E-9)
                return poly

            gpd_results.geometry = gpd_results.geometry.apply(to_polygons)

    if on_mask:
        gpd_results = gpd_results[gpd_results.raster_val != 0]

    return gpd_results


def ma_mask(ds_to_mask: rasterio.DatasetReader,
            extent: Union[Polygon, list],
            nodata: Optional[int] = None,
            crop=False) -> (np.ma.masked_array, affine.Affine):
    """
    Overload of rasterio mask function to create a masked_array

    Args:
        ds_to_mask (rasterio.DatasetReader): Dataset to mask
        extent (Polygon): Extent
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        crop (bool): Whether to crop the raster to the extent of the shapes. Default is False.

    Returns:
         (np.ma.masked_array, affine.Affine): Cropped array as a masked array and the new transform
    """
    if isinstance(extent, Polygon):
        extent = [Polygon]

    # Set nodata
    if not nodata:
        if ds_to_mask.nodata:
            nodata = ds_to_mask.nodata
        else:
            nodata = 0

    # Crop dataset
    crop, trf = mask.mask(ds_to_mask, extent, nodata=nodata, crop=crop)

    # Create masked array
    mask_arr = np.where(crop == nodata, 1, 0).astype(np.uint8)
    crop_mask = np.ma.masked_array(crop, mask_arr, fill_value=nodata)

    return crop_mask, trf


def collocate(master_meta: dict,
              slave_arr: np.ma.masked_array,
              slave_meta: dict,
              resampling: Resampling = Resampling.nearest) -> (np.ma.masked_array, dict):
    """
    Collocate two georeferenced arrays

    Args:
        master_meta (dict): Master metadata
        slave_arr (np.ma.masked_array): Slave array to be collocated
        slave_meta (dict): Slave metadata
        resampling (Resampling): Resampling method

    Returns:
        np.ma.masked_array, dict: Collocated array and its metadata

    """
    collocated_arr = np.zeros((master_meta["count"], master_meta["height"], master_meta["width"]),
                              dtype=master_meta["dtype"])
    warp.reproject(source=slave_arr,
                   destination=collocated_arr,
                   src_transform=slave_meta["transform"],
                   src_crs=slave_meta["crs"],
                   dst_transform=master_meta["transform"],
                   dst_crs=master_meta["crs"],
                   src_nodata=slave_meta["nodata"],
                   dst_nodata=slave_meta["nodata"],
                   resampling=resampling,
                   num_threads=MAX_CORES)

    meta = master_meta.copy()
    meta.update({"dtype": slave_meta["dtype"],
                 "driver": slave_meta["driver"],
                 "nodata": slave_meta["nodata"]})

    return collocated_arr, meta


def read(dataset: rasterio.DatasetReader,
         resolution: Union[list, float],
         resampling: Resampling = Resampling.nearest) -> (np.ma.masked_array, dict):
    """
    Read a raster dataset
    Args:
        dataset (rasterio.DatasetReader): Raster dataset to read
        resolution (list, int): Resolution of the wanted band, in dataset resolution unit (X, Y)
        resampling (Resampling): Resampling method

    Returns:
        np.ma.masked_array, dict: Masked array corresponding to the raster data and its meta data

    """
    if isinstance(resolution, float):
        resolution = [resolution, resolution]

    if len(resolution) != 2:
        raise ValueError("We should have a resolution for X and Y dimensions")

    # Compute new shape
    new_height = int(dataset.height * dataset.res[1] / resolution[1])
    new_width = int(dataset.width * dataset.res[0] / resolution[0])

    # Read data
    array = dataset.read(out_shape=(dataset.count, new_height, new_width),
                         resampling=resampling,
                         masked=True)

    # Update meta
    dst_transform = dataset.transform * dataset.transform.scale((dataset.width / new_width),
                                                                (dataset.height / new_height))
    dst_meta = dataset.meta.copy()
    dst_meta.update({"height": int(new_height),
                     "width": int(new_width),
                     "transform": dst_transform,
                     "dtype": array.dtype,
                     "nodata": dataset.nodata})

    return array, dst_meta


def write(raster: Union[np.ma.masked_array, np.ndarray],
          path: str,
          meta: dict,
          **kwargs) -> None:
    """
    Write raster to disk (encapsulation of rasterio's function)

    Metadata will be copied and updated with raster's information (ie. width, height, count, type...)
    The driver is GTiff by default, and no nodata value is provided.
    The file will be compressed if the raster is a mask (saved as uint8)

    Args:
        raster (Union[np.ma.masked_array, np.ndarray]): Raster to save on disk
        path (str): Path where to save it (directories should be existing)
        meta (dict): Basic metadata that will be copied and updated with raster's information
        **kwargs: Overloading metadata, ie `nodata=255`
    """
    # Manage raster type (impossible to write boolean arrays)
    if raster.dtype == bool:
        raster = raster.astype(np.uint8)

    # Update metadata
    out_meta = meta.copy()

    # Force compression and driver (but can be overwritten by kwargs)
    out_meta["driver"] = "GTiff"
    # Compress only if uint8 data
    if raster.dtype == np.uint8:
        out_meta['compress'] = "lzw"

    # Update metadata with additional params
    for key, val in kwargs.items():
        out_meta[key] = val

    # Manage raster shape
    shape = raster.shape
    if len(shape) == 2:
        raster = np.expand_dims(raster, axis=0)
        count = 1
    else:
        count = shape[0]

    # Stored in rasterio's way
    width = shape[-1]
    height = shape[-2]

    # Update metadata that can be derived from raster
    out_meta["dtype"] = raster.dtype
    out_meta["count"] = count
    out_meta["height"] = height
    out_meta["width"] = width

    # Write product
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(raster)


def get_dim_img_path(dim_path: str, img_name: str = '*') -> list:
    """
    Get the image path from a DIMAP data.

    A BEAM-DIMAP file cannot be opened by rasterio, although its .img file can.

    Args:
        dim_path (str): DIM path (.dim or .data)
        img_name (str): .img file name (or regex), in case there are multiple .img files (ie. for S3 data)

    Returns:
        list: .img file
    """
    if dim_path.endswith(".dim"):
        dim_path = dim_path.replace(".dim", ".data")

    assert dim_path.endswith(".data") and os.path.isdir(dim_path)

    return file_utils.get_file_in_dir(dim_path, img_name, extension='img')
