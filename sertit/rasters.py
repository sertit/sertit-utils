""" Raster tools """
import os
from typing import Union, Optional, Any, Callable
import affine
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
from shapely.geometry import Polygon
import rasterio
from rasterio import features, warp, mask as rmask, merge

from sertit import misc, files, vectors, strings

MAX_CORES = os.cpu_count() - 2


def _to_polygons(val: Any) -> Polygon:
    """
    Convert to polygon (to be used in pandas) -> convert the geometry column

    Args:
        val (Any): Pandas value that has a "coordinates" field

    Returns:
        Polygon: Pandas value as a Polygon
    """
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


def path_or_dst(function: Callable) -> Callable:
    """
    Path or dataset decorator: allows a function to ingest a path or a rasterio dataset
    Args:
        function (Callable): Function to decorate

    Returns:
        Callable: decorated function
    """

    def path_or_dst_wrapper(path_or_ds: Union[str, rasterio.DatasetReader], *args, **kwargs) -> Any:
        """
        Path or dataset wrapper
        Args:
            path_or_ds (Union[str, rasterio.DatasetReader]): Raster path or its dataset
            *args: args
            **kwargs: kwargs

        Returns:
            Any: regular output
        """
        if isinstance(path_or_ds, str):
            with rasterio.open(path_or_ds) as dst:
                out = function(dst, *args, **kwargs)
        else:
            out = function(path_or_ds, *args, **kwargs)
        return out

    return path_or_dst_wrapper


def get_nodata_mask(array: np.ma.masked_array,
                    has_nodata: bool,
                    default_nodata: int = 0) -> np.ma.masked_array:
    """
    Get nodata mask from a masked array.

    The nodata may not be set before, then pass a nodata value that will be evaluated on the array.

    Args:
        array (np.ma.masked_array): Array to evaluate
        has_nodata (bool): If the array as its nodata specified. If not, using default_nodata.
        default_nodata (int): Default nodata used if the array's nodata is not set

    Returns:
        np.ma.masked_array: Pixelwise nodata array

    """
    # Nodata mask
    if has_nodata:  # Unspecified nodata is set to None by rasterio
        nodata_mask = np.where(array.mask, 0, 1).astype(np.uint8)
    else:
        nodata_mask = np.where(array != default_nodata, 1, 0).astype(np.uint8)

    return nodata_mask


@path_or_dst
def _vectorize(dst: Union[str, rasterio.DatasetReader],
               get_nodata: bool = False,
               default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Vectorize a raster, both to get classes or nodata.

    **WARNING**:
    If `get_nodata` is set to False:
        - Please only use this function on a classified raster.
        - This could take a while as the computing time directly depends on the number of polygons to vectorize.
            Please be careful.
    Else:
        - You will get a classified polygon with data (value=0)/nodata pixels. To

    Args:
        dst (str): Path to the raster or its dataset
        get_nodata (bool): Get nodata vector (raster values are set to 0, nodata values are the other ones)
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Vector
    """
    # Get the shapes
    array = dst.read(masked=True)

    # Nodata mask
    nodata_mask = get_nodata_mask(array, dst.nodata is not None, default_nodata)

    # Get shapes (on array or on mask to get nodata vector)
    shapes = features.shapes(nodata_mask if get_nodata else array.data,
                             mask=None if get_nodata else nodata_mask,
                             transform=dst.transform)

    # Convert results to pandas (because of invalid geometries) and save it
    pd_results = pd.DataFrame(shapes, columns=["geometry", "raster_val"])
    if not pd_results.empty:
        # Convert to proper polygons(correct geometries)
        pd_results.geometry = pd_results.geometry.apply(_to_polygons)

    # Convert to geodataframe with correct geometry
    gpd_results = gpd.GeoDataFrame(pd_results, geometry=pd_results.geometry, crs=dst.crs)

    return gpd_results


@path_or_dst
def vectorize(dst: Union[str, rasterio.DatasetReader],
              default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Vectorize a raster to get the class vectors

    **WARNING**:
    - Please only use this function on a classified raster.
    - This could take a while as the computing time directly depends on the number of polygons to vectorize.
        Please be careful.

    Args:
        dst (str): Path to the raster or its dataset
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Classes Vector
    """
    return _vectorize(dst, get_nodata=False, default_nodata=default_nodata)


@path_or_dst
def get_nodata_vec(dst: Union[str, rasterio.DatasetReader],
                   default_nodata: int = 0) -> gpd.GeoDataFrame:
    """
    Get nodata vector

    Args:
        dst (str): Path to the raster or its dataset
        default_nodata (int): Default values for nodata in case of non existing in file
    Returns:
        gpd.GeoDataFrame: Nodata Vector

    """
    nodata = _vectorize(dst, get_nodata=True, default_nodata=default_nodata)
    return nodata[nodata.raster_val != 0]


@path_or_dst
def _mask(dst: Union[str, rasterio.DatasetReader],
          shapes: Union[Polygon, list],
          nodata: Optional[int] = None,
          msk=False,
          **kwargs) -> (np.ma.masked_array, affine.Affine):
    """
    Overload of rasterio mask function in order to create a masked_array.

    The `mask` function doc can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).

    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        dst (rasterio.DatasetReader): Dataset to mask
        shapes (Union[Polygon, list]): Shapes
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        msk (bool): Whether to crop the raster to the extent of the shapes. Default is False.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, affine.Affine): Cropped array as a masked array and the new transform
    """
    if isinstance(shapes, Polygon):
        shapes = [Polygon]

    # Set nodata
    if not nodata:
        if dst.nodata:
            nodata = dst.nodata
        else:
            nodata = 0

    # Crop dataset
    msk, trf = rmask.mask(dst, shapes, nodata=nodata, crop=msk, **kwargs)

    # Create masked array
    nodata_mask = np.where(msk == nodata, 1, 0).astype(np.uint8)
    mask_array = np.ma.masked_array(msk, nodata_mask, fill_value=nodata)

    return mask_array, trf


@path_or_dst
def mask(dst: Union[str, rasterio.DatasetReader],
         shapes: Union[Polygon, list],
         nodata: Optional[int] = None,
         **kwargs) -> (np.ma.masked_array, affine.Affine):
    """
    Masking a dataset:
    setting nodata outside of the given shapes, but without cropping the raster to the shapes extent.

    HOW:
    Overload of rasterio mask function in order to create a masked_array.
    The `mask` function doc can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        dst (rasterio.DatasetReader): Dataset to mask
        shapes (Union[Polygon, list]): Shapes
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, affine.Affine): Cropped array as a masked array and the new transform
    """
    return _mask(dst, shapes=shapes, nodata=nodata, msk=False, **kwargs)


@path_or_dst
def crop(dst: Union[str, rasterio.DatasetReader],
         shapes: Union[Polygon, list],
         nodata: Optional[int] = None,
         **kwargs) -> (np.ma.masked_array, affine.Affine):
    """
    Cropping a dataset:
    setting nodata outside of the given shapes AND cropping the raster to the shapes extent.

    HOW:
    Overload of rasterio mask function in order to create a masked_array.
    The `mask` function doc can be seen [here](https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html).
    It basically masks a raster with a vector mask, with the possibility to crop the raster to the vector's extent.

    Args:
        dst (rasterio.DatasetReader): Dataset to mask
        shapes (Union[Polygon, list]): Shapes
        nodata (int): Nodata value. If not set, uses the ds.nodata. If doesnt exist, set to 0.
        **kwargs: Other rasterio.mask options

    Returns:
         (np.ma.masked_array, affine.Affine): Cropped array as a masked array and the new transform
    """
    return _mask(dst, shapes=shapes, nodata=nodata, msk=True, **kwargs)


@path_or_dst
def read(dst: Union[str, rasterio.DatasetReader],
         resolution: Union[list, float] = None,
         resampling: Resampling = Resampling.nearest,
         masked=True) -> (np.ma.masked_array, dict):
    """
    Read a raster dataset from a `rasterio.Dataset`.
    Args:
        dst (rasterio.DatasetReader): Raster dataset to read
        resolution (list, int): Resolution of the wanted band, in dataset resolution unit (X, Y)
        resampling (Resampling): Resampling method
        masked (bool); Get a masked array

    Returns:
        np.ma.masked_array, dict: Masked array corresponding to the raster data and its meta data

    """

    def get_new_dim(dim: int, res_old: float, res_new: float) -> int:
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
    if isinstance(resolution, (int, float)):
        new_height = get_new_dim(dst.height, dst.res[1], resolution)
        new_width = get_new_dim(dst.width, dst.res[0], resolution)
    elif isinstance(resolution, list):
        if len(resolution) != 2:
            raise ValueError("We should have a resolution for X and Y dimensions")

        if resolution[0] is not None:
            new_width = get_new_dim(dst.width, dst.res[0], resolution[0])

        if resolution[1] is not None:
            new_height = get_new_dim(dst.height, dst.res[1], resolution[1])
    elif resolution is None:
        pass
    else:
        raise ValueError(f"Resolution should be None, 2 floats or a list: {resolution}")

    # Read data
    array = dst.read(out_shape=(dst.count, new_height, new_width),
                     resampling=resampling,
                     masked=masked)

    # Update meta
    dst_transform = dst.transform * dst.transform.scale((dst.width / new_width),
                                                        (dst.height / new_height))
    dst_meta = dst.meta.copy()
    dst_meta.update({"height": new_height,
                     "width": new_width,
                     "transform": dst_transform,
                     "dtype": array.dtype,
                     "nodata": dst.nodata})

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

    # Update raster to be sure to write down correct nodata pixels
    if isinstance(raster, np.ma.masked_array):
        raster[raster.mask] = raster.fill_value

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


def collocate(master_meta: dict,
              slave_arr: Union[np.ma.masked_array, np.ndarray],
              slave_meta: dict,
              resampling: Resampling = Resampling.nearest) -> (Union[np.ma.masked_array, np.ndarray], dict):
    """
    Collocate two georeferenced arrays:
    forces the *slave* raster to be exactly georeferenced onto the *master* raster by reprojection.

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


def sieve(array: Union[np.ma.masked_array, np.ndarray],
          out_meta: dict,
          sieve_thresh: int,
          connectivity: int = 4) -> (Union[np.ma.masked_array, np.ndarray], dict):
    """
    Sieving, overloads rasterio function with raster shaped like (1, h, w).

    Forces the output to `np.uint8` (as only classified rasters should be sieved)

    Args:
        array (Union[np.ma.masked_array, np.ndarray]): Array to sieve
        out_meta (dict): Metadata to update
        sieve_thresh (int): Sieving threshold in pixels
        connectivity (int): Connectivity, either 4 or 8

    Returns:
        Union[np.ma.masked_array, np.ndarray], dict: Sieved array and updated meta
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
    if meta['dtype'] != dtype:
        array = array.astype(dtype)
        meta['dtype'] = dtype

    # Sieve
    result_array = np.empty(array.shape, dtype=array.dtype)
    features.sieve(array, size=sieve_thresh, out=result_array, connectivity=connectivity)

    # Use this trick to get the array back to 'normal'
    if expand:
        result_array = np.expand_dims(result_array, axis=0)

    return result_array, meta


def get_dim_img_path(dim_path: str, img_name: str = '*') -> list:
    """
    Get the image path from a *BEAM-DIMAP* data.

    A *BEAM-DIMAP* file cannot be opened by rasterio, although its .img file can.

    Args:
        dim_path (str): DIM path (.dim or .data)
        img_name (str): .img file name (or regex), in case there are multiple .img files (ie. for S3 data)

    Returns:
        list: .img file
    """
    if dim_path.endswith(".dim"):
        dim_path = dim_path.replace(".dim", ".data")

    assert dim_path.endswith(".data") and os.path.isdir(dim_path)

    return files.get_file_in_dir(dim_path, img_name, extension='img')


@path_or_dst
def get_extent(path_or_ds: Union[str, rasterio.DatasetReader]) -> gpd.GeoDataFrame:
    """
    Get the extent of a raster as a `geopandas.Geodataframe`.

    Args:
        path_or_ds (Union[str, rasterio.DatasetReader]): Raster path

    Returns:
        gpd.GeoDataFrame: Extent as a `geopandas.Geodataframe`
    """
    return vectors.get_geodf(geometry=[*path_or_ds.bounds], geom_crs=path_or_ds.crs)


@path_or_dst
def get_footprint(path_or_ds: Union[str, rasterio.DatasetReader]) -> gpd.GeoDataFrame:
    """
    Get real footprint of the product (without nodata, in french == emprise utile)

    Args:
        path_or_ds (Union[str, rasterio.DatasetReader]): Raster path
    Returns:
        gpd.GeoDataFrame: Footprint as a GeoDataFrame
    """
    footprint = get_nodata_vec(path_or_ds)

    # Get the footprint max (discard small holes stored in other polygons)
    footprint = footprint[footprint.area == np.max(footprint.area)]

    # Only select the exterior of this footprint(sometimes some holes persist, especially when merging SAR data)
    if not footprint.empty:
        footprint_poly = Polygon(list(footprint.exterior.iat[0].coords))
        footprint = gpd.GeoDataFrame(geometry=[footprint_poly], crs=footprint.crs)

        # Resets index as we only got one polygon left which should have index 0
        footprint.reset_index(inplace=True)

    return footprint


def merge_vrt(crs_paths: list, crs_merged_path: str, **kwargs) -> None:
    """
    Merge rasters as a VRT. Uses `gdalbuildvrt`.

    See here: https://gdal.org/programs/gdalbuildvrt.html

    Creates VRT with relative paths !

    **WARNING:** They should have the same CRS !

    Args:
        crs_paths (list): Path of the rasters to be merged with the same CRS)
        crs_merged_path (str): Path to the merged raster
        kwargs (dict): Other gdlabuildvrt arguments
    """
    # Create relative paths
    vrt_root = os.path.dirname(crs_merged_path)
    rel_paths = [strings.to_cmd_string(files.real_rel_path(path, vrt_root)) for path in crs_paths]
    rel_vrt = strings.to_cmd_string(files.real_rel_path(crs_merged_path, vrt_root))

    # Run cmd
    arg_list = [val for item in kwargs.items() for val in item]
    vrt_cmd = ["gdalbuildvrt", rel_vrt, *rel_paths, *arg_list]
    misc.run_cli(vrt_cmd, cwd=vrt_root)


def merge_gtiff(crs_paths: list, crs_merged_path: str) -> None:
    """
    Merge rasters as a GeoTiff.

    **WARNING:** They should have the same CRS !

    Args:
        crs_paths (list): Path of the rasters to be merged with the same CRS)
        crs_merged_path (str): Path to the merged raster
    """
    # Open datasets for merging
    crs_datasets = []
    try:
        for path in crs_paths:
            crs_datasets.append(rasterio.open(path))

        # Merge all datasets
        merged_array, merged_transform = merge.merge(crs_datasets, method='max')
        merged_meta = crs_datasets[0].meta.copy()
        merged_meta.update({"driver": "GTiff",
                            "height": merged_array.shape[1],
                            "width": merged_array.shape[2],
                            "transform": merged_transform})
    finally:
        # Close all datasets
        for dataset in crs_datasets:
            dataset.close()

    # Save merge datasets
    write(merged_array, crs_merged_path, crs_datasets[0].meta, transform=merged_transform)


def unpackbits(array: np.ndarray, nof_bits: int) -> np.ndarray:
    """
    Function found here:
    https://stackoverflow.com/questions/18296035/how-to-extract-the-bits-of-larger-numeric-numpy-data-types
    Args:
        array (np.ndarray): Array to unpack
        nof_bits (int): Number of bits to unpack

    Returns:
        np.ndarray: Unpacked array
    """
    xshape = list(array.shape)
    array = array.reshape([-1, 1])
    msk = 2 ** np.arange(nof_bits).reshape([1, nof_bits])
    return (array & msk).astype(bool).astype(np.uint8).reshape(xshape + [nof_bits])


def read_bit_array(bit_mask: np.ndarray, bit_id: Union[list, int]) -> Union[np.ndarray, list]:
    """
    Read bit arrays as a succession of binary masks (sort of read a slice of the bit mask, slice number bit_id)

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
