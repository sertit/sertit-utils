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
Vectors tools

You can use this only if you have installed sertit[full] or sertit[vectors]
"""

import contextlib
import logging
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from cloudpathlib.exceptions import AnyPathTypeError
from shapely import Polygon, wkt

from sertit import AnyPath, files, geometry, logs, misc, path, strings
from sertit.logs import SU_NAME
from sertit.types import AnyPathStrType, AnyPathType

LOGGER = logging.getLogger(SU_NAME)

EPSG_4326 = "EPSG:4326"
WGS84 = EPSG_4326

EXT_TO_DRIVER = {
    ".shp": "ESRI Shapefile",
    ".kml": "KML",
    ".kmz": "KMZ",
    ".json": "GeoJSON",
    ".geojson": "GeoJSON",
    ".gml": "GML",
}

SHP_CO_FILES = [".dbf", ".prj", ".sbn", ".sbx", ".shx", ".sld"]


def is_geopandas_1_0():
    """Is geopandas over 1.0.0. Default engine changes, from fiona to pyogrio"""
    return misc.compare_version("geopandas", "1.0.0", ">=")


if is_geopandas_1_0():
    from pyogrio.errors import DataSourceError
    from shapely.errors import GEOSException

    CPLE_AppDefinedError = Exception
else:
    from fiona._err import CPLE_AppDefinedError
    from fiona.errors import DriverError as DataSourceError
    from fiona.errors import UnsupportedGeometryTypeError as GEOSException

# Handle errors changing from fiona to pyogrio
DataSourceError = DataSourceError
GEOSException = GEOSException
CPLE_AppDefinedError = CPLE_AppDefinedError


def to_utm_crs(lon: float, lat: float) -> "CRS":  # noqa: F821
    """
    .. deprecated:: 1.29.1
       Use `estimate_utm_crs <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.estimate_utm_crs.html>`_ instead, which directly returs a CRS instead of a string.

    Find the EPSG code of the UTM CRS from a lon/lat in WGS84.

    Args:
        lon (float): Longitude (WGS84, epsg:4326)
        lat (float): Latitude (WGS84, epsg:4326)

    Returns:
        CRS: UTM CRS

    Example:
        >>> to_utm_crs(lon=7.8, lat=48.6)  # Strasbourg
        <Derived Projected CRS: EPSG:32632>
        Name: WGS 84 / UTM zone 32N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - bounds: (6.0, 0.0, 12.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 32N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

    """
    # Manage the case with centroids etc. that are already written as arrays
    try:
        point = gpd.points_from_xy([lon], [lat])
    except ValueError:
        point = gpd.points_from_xy(lon, lat)

    return gpd.GeoDataFrame(geometry=point, crs=EPSG_4326).estimate_utm_crs()


def corresponding_utm_projection(lon: float, lat: float) -> str:
    """
    .. deprecated:: 1.29.1
       Use `estimate_utm_crs <https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.estimate_utm_crs.html>`_ instead, which directly returs a CRS instead of a string.

    Find the EPSG code of the UTM CRS from a lon/lat in WGS84.

    Args:
        lon (float): Longitude (WGS84, epsg:4326)
        lat (float): Latitude (WGS84, epsg:4326)

    Returns:
        CRS: UTM CRS

    Example:
        >>> to_utm_crs(lon=7.8, lat=48.6)  # Strasbourg
        <Derived Projected CRS: EPSG:32632>
        Name: WGS 84 / UTM zone 32N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - bounds: (6.0, 0.0, 12.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 32N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

    """
    logs.deprecation_warning(
        "Deprecated, use 'to_utm_crs' instead, which directly returs a CRS instead of a string."
    )
    return to_utm_crs(lon, lat).to_string()


def get_geodf(geom: Union[Polygon, list, gpd.GeoSeries], crs: str) -> gpd.GeoDataFrame:
    """
    Get a GeoDataFrame from a geometry and a crs

    Args:
        geom (Union[Polygon, list]): List of Polygons, or Polygon or bounds
        crs (str): CRS of the polygon

    Returns:
        gpd.GeoDataFrame: Geometry as a geodataframe

    Example:
        >>> poly = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
        >>> geodf = get_geodf(poly, crs=WGS84)
        >>> print(geodf)
                                                    geometry
        0  POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
    """
    if isinstance(geom, list):
        if isinstance(geom[0], Polygon):
            pass
        else:
            try:
                geom = [geometry.from_bounds_to_polygon(*geom)]
            except TypeError as ex:
                raise TypeError(
                    "Give the extent as 'left', 'bottom', 'right', and 'top'"
                ) from ex
    elif isinstance(geom, Polygon):
        geom = [geom]
    elif isinstance(geom, gpd.GeoSeries):
        geom = geom.geometry
    else:
        raise TypeError("geometry should be a list or a Polygon.")

    return gpd.GeoDataFrame(geometry=geom, crs=crs)


def set_kml_driver() -> None:
    """
    Set KML driver for Fiona data (use it at your own risks !)

    Only useful with :code:`geopandas<1.0.0` or with :code:`fiona`'s engine

    Example:
        >>> path = "path/to/kml.kml"
        >>> gpd.read_file(path, engine="fiona")
        fiona.errors.DriverError: unsupported driver: 'LIBKML'
        >>> set_kml_driver()
        >>> gpd.read_file(path, engine="fiona")
                       Name  ...                                           geometry
        0  CC679_new_AOI2_3  ...  POLYGON Z ((45.03532 32.49765 0.00000, 46.1947...
        [1 rows x 12 columns]

    """
    if not is_geopandas_1_0():
        import fiona

        drivers = fiona.drvsupport.supported_drivers

        if "LIBKML" not in drivers:
            drivers["LIBKML"] = "rw"
        if "KML" not in drivers:  # Just in case
            drivers["KML"] = "rw"


def get_aoi_wkt(aoi_path: AnyPathStrType, as_str: bool = True) -> Union[str, Polygon]:
    """
    Get AOI formatted as a WKT from files that can be read by Fiona (like shapefiles, ...)
    or directly from a WKT file. The use of KML has been forced (use it at your own risks !).

    See: https://fiona.readthedocs.io/en/latest/fiona.html#fiona.open

    It is assessed that:

    - only **one** polygon composes the AOI (as only the first one is read)
    - it should be specified in lat/lon (WGS84) if a WKT file is provided

    Args:
        aoi_path (AnyPathStrType): Absolute or relative path to an AOI.
            Its format should be WKT or any format read by Fiona, like shapefiles.
        as_str (bool): If True, return WKT as a str, otherwise as a shapely geometry

    Returns:
        Union[str, Polygon]: AOI formatted as a WKT stored in lat/lon

    Example:
        >>> path = "path/to/vec.geojson"  # OK with ESRI Shapefile, geojson, WKT, KML...
        >>> get_aoi_wkt(path)
        'POLYGON Z ((46.1947755465253067 32.4973553439109324 0.0000000000000000, 45.0353174370802520 32.4976496856158974
        0.0000000000000000, 45.0355748149750283 34.1139970085580018 0.0000000000000000, 46.1956059695554089
        34.1144793800670882 0.0000000000000000, 46.1947755465253067 32.4973553439109324 0.0000000000000000))'


    """
    aoi_path = AnyPath(aoi_path)
    if not aoi_path.is_file():
        raise FileNotFoundError(f"AOI file {aoi_path} does not exist.")

    if aoi_path.suffix == ".wkt":
        try:
            with open(aoi_path) as aoi_f:
                aoi = wkt.load(aoi_f)
        except Exception as ex:
            raise ValueError("AOI WKT cannot be read") from ex
    else:
        try:
            # Open file
            aoi_file = read(aoi_path, crs=EPSG_4326)

            # Get envelope polygon
            geom = aoi_file["geometry"]
            if len(geom) > 1:
                LOGGER.warning(
                    "Your AOI contains several polygons. Only the first will be treated !"
                )
            polygon = geom[0].convex_hull

            # Convert to WKT
            aoi = wkt.loads(str(polygon))

        except Exception as ex:
            raise ValueError("AOI cannot be read by Fiona") from ex

    # Convert to string if needed
    if as_str:
        aoi = wkt.dumps(aoi)

    LOGGER.debug("Specified AOI in WKT: %s", aoi)
    return aoi


def shapes_to_gdf(shapes: Generator, crs: str) -> gpd.GeoDataFrame:
    """
    Convert rasterio shapes to geodataframe

    Args:
        shapes (Generator): Shapes from rasterio
        crs: Wanted CRS of the vector. If None, using naive or origin CRS.

    Returns:
        gpd.GeoDataFrame: Shapes as a GeoDataFrame
    """

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
            pass

        return poly

    # Convert results to pandas (because of invalid geometries) and save it
    pd_results = pd.DataFrame(shapes, columns=["geometry", "raster_val"])

    if not pd_results.empty:
        # Convert to proper polygons(correct geometries)
        pd_results.geometry = pd_results.geometry.apply(_to_polygons)

    # Convert to geodataframe with correct geometry
    gdf = gpd.GeoDataFrame(pd_results, geometry=pd_results.geometry, crs=crs)

    # Return valid geometries
    gdf = geometry.make_valid(gdf)

    return gdf


def write(gdf: gpd.GeoDataFrame, path: AnyPathStrType, **kwargs) -> None:
    """
    Write vector to disk, managing the common drivers automatically.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to write on disk
        path (AnyPathStrType): Where to write on disk.

    Example:
        >>> write(my_vector, path="my_vector.kml")
    """
    path = AnyPath(path)

    driver = kwargs.pop("driver", None)
    if not driver:
        driver = EXT_TO_DRIVER.get(path.suffix)
        if driver == "KML":
            set_kml_driver()
        elif driver == "KMZ":
            raise NotImplementedError("Impossible to write a KMZ for now.")

    gdf.to_file(str(path), driver=driver, **misc.prune_dict(kwargs, ["window"]))


def copy(src_path: AnyPathStrType, dst_path: AnyPathStrType) -> AnyPathType:
    """
     Copy vector (handles shapefiles additional files)

    Args:
        src_path (AnyPathStrType): Source Path
        dst_path (AnyPathStrType): Destination Path (file or folder)

    Returns:
        AnyPathType: Path to copied vector

    Example:
        >>> new_aoi_path = copy("in/aoi.shp", "out/aoi.shp")
        >>> new_aoi_path
        PosixPath('out/aoi.shp')
        >>> list(new_aoi_path.parent.glob("*"))
        [PosixPath('out/aoi.dbf'), PosixPath('out/aoi.prj'), PosixPath('out/aoi.shp'), PosixPath('out/aoi.shx')]
    """
    src_path = AnyPath(src_path)
    dst_path = AnyPath(dst_path)

    if not dst_path.is_file():
        dst_path = files.copy(src_path, dst_path)

        # Add files that come with shape
        shp_co_files = [
            file
            for file in src_path.parent.glob(f"{path.get_filename(src_path)}.*")
            if file.suffix in SHP_CO_FILES
        ]
        for co_file in shp_co_files:
            files.copy(co_file, dst_path.with_suffix(co_file.suffix))

    return dst_path


def read(
    vector_path: AnyPathStrType,
    crs: Any = None,
    archive_regex: str = None,
    window: Any = None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Read any vector:

    - if KML/KMZ: sets correctly the drivers and open layered KML (you may need :code:`ogr2ogr` to make it work !)
    - if archive (only zip or tar), use a regex to look for the vector inside the archive. You can use this `site <https://regexr.com/>`_ to build your regex.
    - if GML: manages the empty errors

    Handles a lot of exceptions and have fallback mechanisms with :code:`ogr2ogr` (if in :code:`PATH`)

    Handles both on disk and cloud-stored vectors.

    Adds two attributes to your vector:

    - :code:`path`: File path
    - :code:`name`: File name

    Args:
        vector_path (AnyPathStrType): Path to vector to read. In case of archive, path to the archive.
        crs: Wanted CRS of the vector. If None, using naive or origin CRS.
        archive_regex (str): [Archive only] Regex for the wanted vector inside the archive
        window (Any): Anything that can be returned as a bbox (i.e. path, gpd.GeoPandas, Iterable, ...).
            In case of an iterable, assumption is made it corresponds to geographic bounds. Mimics :code:`rasters.read(..., window=)`. If given, :code:`bbox` is ignored.
        **kwargs: Additional arguments used in gpd.read_file.
            You can also give :code:`file_list`, the list of files of the archive to get the vector from, as this operation is expensive when done with large archives stored on the cloud.
            You can also set :code:`compute_sindex=False` to avoid computing the spatial index of the vector.

    Returns:
        gpd.GeoDataFrame: Read vector as a GeoDataFrame

    Examples:
        >>> # Usual
        >>> path = 'D:/path/to/vector.geojson'
        >>> vec = vectors.read(path, crs=WGS84)
                               Name  ...                                           geometry
        0  Sentinel-1 Image Overlay  ...  POLYGON ((0.85336 42.24660, -2.32032 42.65493,...
        >>>
        >>> # Attributes
        >>> vec.attrs["path"]
        'D:/path/to/vector.geojson'
        >>>
        >>> vec.attrs["name"]
        'vector.geojson'

        >>> # Archive
        >>> arch_path = 'D:/path/to/zip.zip'
        >>> vectors.read(arch_path, archive_regex=r".*map-overlay.kml")
                               Name  ...                                           geometry
        0  Sentinel-1 Image Overlay  ...  POLYGON ((0.85336 42.24660, -2.32032 42.65493,...
    """
    # Default values
    gpd_vect_path = str(vector_path)
    arch_path = None

    # -- Manage window and convert it to a bbox
    if window is not None:
        try:
            bbox = read(window)
        except (FileNotFoundError, TypeError):
            # Convert ndarray to tuple
            bbox = tuple(window) if isinstance(window, np.ndarray) else window

        kwargs["bbox"] = bbox

    # -- Manage the path formatting (create the path to be read by GeoPandas, the archive path if needed, ...)
    try:
        vector_path = AnyPath(vector_path)

        # Manage formatted archive file (fsspec style for example)
        if "!" in str(vector_path):
            split_vect = str(vector_path).split("!")
            archive_regex = ".*{}".format(split_vect[1].replace(".", r"\."))
            vector_path = AnyPath(split_vect[0])

        # Manage archive case
        if vector_path.suffix in [".tar", ".zip"]:
            prefix = vector_path.suffix[-3:]
            file_list = kwargs.pop(
                "file_list", path.get_archived_file_list(vector_path)
            )

            try:
                regex = re.compile(archive_regex)
                arch_path = list(filter(regex.match, file_list))[0]

                # Different template if on cloud or not... (only tested with S3)
                if path.is_cloud_path(vector_path):
                    gpd_vect_path = f"{prefix}+{vector_path}!{arch_path}"
                else:
                    gpd_vect_path = f"{prefix}://{vector_path}!{arch_path}"
            except IndexError as exc:
                raise FileNotFoundError(
                    f"Impossible to find vector {archive_regex} in {path.get_filename(vector_path)}"
                ) from exc
        # Don't read tar.gz archives (way too slow)
        elif vector_path.suffixes == [".tar", ".gz"]:
            raise TypeError(
                ".tar.gz files are too slow to be read from inside the archive. Please extract them instead."
            )
    except AnyPathTypeError:
        pass

    # Check existence of the file (here and not before to handle fsspec cases with '!')
    if not AnyPath(vector_path).exists():
        raise FileNotFoundError(f"Non existing vector: {vector_path}")

    # Read vector
    vect = _read_vector_core(gpd_vect_path, vector_path, arch_path, crs, **kwargs)

    # Add some attributes
    vect.attrs["path"] = str(vector_path)
    vect.attrs["name"] = path.get_filename(vector_path)

    # Generate spatial index for optimization
    with contextlib.suppress(AttributeError):
        if kwargs.get("compute_sindex", True) and not vect.has_sindex:
            vect.sindex  # noqa

    return vect


def _read_vector_core(
    gpd_vect_path: str, raw_path: AnyPathStrType, arch_path: str, crs, **kwargs
):
    """
    Read vector (core function) with correctly formatted paths.

    Handles a lot of exceptions, reads KML, and have fallback mechanisms with ogr2ogr (if in PATH)

    Args:
        gpd_vect_path (str): Resolved vector path (readable by geopandas)
        raw_path (AnyPathStrType): Path to vector to read. In case of archive, path to the archive.
        arch_path (str): If archived vector, path to the vector file inside the archive (from the root of the archive)
        crs: Wanted CRS of the vector. If None, using naive or origin CRS.
        **kwargs: Other arguments

    Returns:
        gpd.GeoDataFrame: Read vector as a GeoDataFrame
    """
    tmp_dir = None

    # -- Open vector
    try:
        # Discard some weird error concerning a NULL pointer that outputs a ValueError (as we already except it)
        fiona_logger = logging.getLogger("fiona")
        fiona_logger.setLevel(logging.CRITICAL)

        # Manage KML driver
        if gpd_vect_path.endswith(".kml") or gpd_vect_path.endswith(".kmz"):
            vect = _read_kml(gpd_vect_path, raw_path, arch_path, tmp_dir, **kwargs)
        else:
            vect = gpd.read_file(gpd_vect_path, **kwargs)

        # Manage naive geometries
        try:
            if vect.crs and crs:
                vect = vect.to_crs(crs)
        except AttributeError:
            # Pyogrio don't create crs columns for dbf files for example
            pass

        # Set fiona logger back to what it was
        fiona_logger.setLevel(logging.INFO)
    except DataSourceError:
        raise
    except (ValueError, GEOSException, IndexError) as ex:
        if "Use a.any() or a.all()" in str(ex):
            raise
        # Do not print warning for null layer
        elif "Null layer" not in str(ex):
            LOGGER.warning(ex)
        vect = gpd.GeoDataFrame(geometry=[], crs=crs)
    except CPLE_AppDefinedError as ex:
        # CPLE_AppDefinedError is not a pyogrio exception and this is therefore too broad
        if is_geopandas_1_0() and kwargs.get("engine") != "fiona":
            raise ex

        # Last try to read this vector
        # Needs ogr2ogr here
        if shutil.which("ogr2ogr"):
            # Open as geojson
            tmp_dir = tempfile.TemporaryDirectory()
            vect_path_gj = ogr2geojson(raw_path, tmp_dir.name, arch_path)
            vect = gpd.read_file(vect_path_gj, **kwargs)
            vect.crs = None
        else:
            # Do not print warning for null layer
            if "Null layer" not in str(ex):
                LOGGER.warning(ex)
            vect = gpd.GeoDataFrame(geometry=[], crs=crs)

    # Clean if needed
    if tmp_dir:
        tmp_dir.cleanup()

    return vect


def _read_kml(
    gpd_vect_path: str,
    raw_path: AnyPathStrType,
    arch_path: str = None,
    tmp_dir=None,
    **kwargs,
) -> gpd.GeoDataFrame:
    """
    Reader of KML data

    Args:
        gpd_vect_path (str): Resolved vector path (readable by geopandas)
        raw_path (AnyPathStrType): Path to vector to read. In case of archive, path to the archive.
        arch_path: If archived vector, path to the vector file inside the archive (from the root of the archive)
        tmp_dir: Temporary directory
        **kwargs: Additional arguments used in gpd.read_file

    Returns:
        gpd.GeoDataFrame: KML as a geopandas GeoDataFrame

    """
    vect = gpd.GeoDataFrame()
    driver = "KML" if gpd_vect_path.endswith(".kml") else "KMZ"
    engine = None

    # Errors reading KML and KMZ with pyogrio for now (v0.11.0 still buggy)
    # https://github.com/geopandas/pyogrio/issues/543
    # https://github.com/geopandas/pyogrio/issues/444
    use_pyogrio = is_geopandas_1_0()
    from importlib.metadata import version

    if misc.compare_version("pyogrio", "0.11.1", "<="):
        engine = "fiona"
        use_pyogrio = False

    if use_pyogrio:
        try:
            vect = gpd.read_file(gpd_vect_path, driver=driver, engine=engine, **kwargs)
        except DataSourceError:
            LOGGER.error(
                f"Error in reading {path.get_filename(gpd_vect_path)}. geopandas: {version('geopandas')},  pyogrio: {version('pyogrio')}, engine: {engine}"
            )
    else:
        import fiona

        set_kml_driver()

        # Document tags in KML file are separate layers for GeoPandas.
        # When you try to get the KML content, you actually get the first layer.
        # So you need for loop for iterating over layers.
        # https://gis.stackexchange.com/questions/328525/geopandas-read-file-only-reading-first-part-of-kml/328554
        for layer in fiona.listlayers(gpd_vect_path):
            try:
                vect_layer = gpd.read_file(
                    gpd_vect_path, driver=driver, layer=layer, engine=engine, **kwargs
                )
                if not vect_layer.empty:
                    # KML files are always in WGS84 (and does not contain this information)
                    vect_layer.crs = EPSG_4326
                    vect = pd.concat([vect, vect_layer])
            except ValueError:
                pass  # Except Null Layer

    # Workaround for archived KML -> they may be empty
    # Convert KML to GeoJSON
    # Needs ogr2ogr here
    if vect.empty:
        if shutil.which("ogr2ogr"):
            LOGGER.debug(
                "Impossible to open your KML file with Python. Using 'ogr2ogr' to convert it into a more readable format."
            )
            # Open the geojson
            if not tmp_dir:
                tmp_dir = tempfile.TemporaryDirectory()

            vect_path_gj = ogr2geojson(raw_path, tmp_dir.name, arch_path)
            vect = gpd.read_file(vect_path_gj, **kwargs)
        else:
            # Try reading it in a basic manner
            LOGGER.warning(
                "Missing `ogr2ogr` in your PATH, your KML may be incomplete. "
                "(KML files can contain unsupported data structures, nested folders etc.)"
            )
            try:
                vect = gpd.read_file(gpd_vect_path, **kwargs)
            except Exception:
                # Force set CRS to empty vector
                vect.crs = EPSG_4326

    return vect


def ogr2geojson(
    vector_path: AnyPathStrType,
    out_dir: AnyPathStrType,
    arch_vect_path: str = None,
) -> str:
    """
    Wrapper of ogr2ogr function, converting the input vector to GeoJSON.

    Args:
        vector_path (AnyPathStrType): Path to vector to read. In case of archive, path to the archive.
        out_dir (AnyPathStrType): Output directory
        arch_vect_path: If archived vector, path to the vector file inside the archive (from the root of the archive)

    Returns:
        str: Converted file

    Args:
        vector_path (AnyPathStrType): Path to vector to read. In case of archive, path to the archive.
        out_dir (AnyPathStrType): Output directory
        arch_vect_path: If archived vector, path to the vector file inside the archive (from the root of the archive)

    Returns:
        str: Converted file
    """
    assert shutil.which("ogr2ogr")  # Needs ogr2ogr here

    # Convert to strings to make it work with CLI
    vector_path = AnyPath(vector_path)

    # archived vector_path are extracted in a tmp folder so no need to be downloaded
    if vector_path.suffix == ".zip":
        with zipfile.ZipFile(vector_path, "r") as zip_ds:
            vect_path = zip_ds.extract(arch_vect_path, out_dir)
    elif vector_path.suffix == ".tar":
        with tarfile.open(vector_path, "r") as tar_ds:
            tar_ds.extract(arch_vect_path, out_dir)
            vect_path = os.path.join(out_dir, arch_vect_path)
    else:
        # vector_path should be downloaded to work with 'ogr2ogr'
        if path.is_cloud_path(vector_path):
            vector_path = AnyPath(vector_path).fspath
        vect_path = vector_path

    vect_path_gj = os.path.join(
        out_dir,
        os.path.basename(vect_path).replace(path.get_ext(vect_path), "geojson"),
    )
    cmd_line = [
        "ogr2ogr",
        "-fieldTypeToString DateTime",  # Disable warning
        "-f GeoJSON",
        strings.to_cmd_string(vect_path_gj),  # dst
        strings.to_cmd_string(vect_path),  # src
    ]
    try:
        misc.run_cli(cmd_line)
    except RuntimeError as ex:
        raise RuntimeError(f"Something went wrong with ogr2ogr: {ex}") from ex

    return vect_path_gj


@contextmanager
def utm_crs(gdf: gpd.GeoDataFrame) -> None:
    """
    Change temporary the CRS of a vector, ie when computing area based statistics / features (centroid....) which need a meter-based CRS.

    WARNING:
        The modifications (other than CRS) on the yielded GeoDataFrame will be kept!

    Args:
        gdf (str): GeoDataFrame to convert

    Example:
        >>> vect = vectors.read(vectors_path().joinpath("aoi.kml"))
        >>> with vectors.utm_crs(vect) as utm_vect:
        >>>     utm_centroid = utm_vect.centroid
        >>>     utm_vect["centroid_utm"] = utm_centroid
        >>> vect["centroid_utm"].equals(c2)
        True
    """
    src_crs = None
    if not gdf.crs.is_projected:
        src_crs = gdf.crs
        gdf.to_crs(gdf.estimate_utm_crs(), inplace=True)
    try:
        yield gdf
    finally:
        if src_crs is not None:
            gdf.to_crs(src_crs, inplace=True)
