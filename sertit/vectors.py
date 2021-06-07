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
Vectors tools

You can use this only if you have installed sertit[full] or sertit[vectors]
"""
import logging
import os
from typing import Any, Generator, Union

import numpy as np
import pandas as pd
from fiona.errors import UnsupportedGeometryTypeError

try:
    import geopandas as gpd
    from shapely import wkt
    from shapely.geometry import MultiPolygon, Polygon, box
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "Please install 'geopandas' to use the rasters package."
    ) from ex

from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)

WGS84 = "EPSG:4326"


def corresponding_utm_projection(lon: float, lat: float) -> str:
    """
    Find the EPSG code of the UTM projection from a lon/lat in WGS84.

    ```python
    >>> corresponding_utm_projection(lon=7.8, lat=48.6)  # Strasbourg
    'EPSG:32632'
    ```

    Args:
        lon (float): Longitude (WGS84)
        lat (float): Latitude (WGS84)

    Returns:
        str: EPSG string

    """
    # EPSG code begins with 32
    # Then 6 if north, 7 if south -> (np.sign(lat) + 1) / 2 * 100 == 1 if lat > 0 (north), 0 if lat < 0 (south)
    # Then EPSG code with usual formula np.floor((180 + lon) / 6) + 1)
    epsg = int(32700 - (np.sign(lat) + 1) / 2 * 100 + np.floor((180 + lon) / 6) + 1)
    return f"EPSG:{epsg}"


def from_polygon_to_bounds(
    polygon: Union[Polygon, MultiPolygon]
) -> (float, float, float, float):
    """
    Convert a `shapely.polygon` to its bounds, sorted as `left, bottom, right, top`.

    ```python
    >>> poly = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
    >>> from_polygon_to_bounds(poly)
    (0.0, 0.0, 1.0, 1.0)
    ```

    Args:
        polygon (MultiPolygon): polygon to convert

    Returns:
        (float, float, float, float): left, bottom, right, top
    """
    left = polygon.bounds[0]  # xmin
    bottom = polygon.bounds[1]  # ymin
    right = polygon.bounds[2]  # xmax
    top = polygon.bounds[3]  # ymax

    assert left < right
    assert bottom < top

    return left, bottom, right, top


def from_bounds_to_polygon(
    left: float, bottom: float, right: float, top: float
) -> Polygon:
    """
    Convert the bounds to a `shapely.polygon`.

    ```python
    >>> poly = from_bounds_to_polygon(0.0, 0.0, 1.0, 1.0)
    >>> print(poly)
    'POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))'
    ```

    Args:
        left (float): Left coordinates
        bottom (float): Bottom coordinates
        right (float): Right coordinates
        top (float): Top coordinates

    Returns:
        Polygon: Polygon corresponding to the bounds

    """
    return box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))


def get_geodf(
    geometry: Union[Polygon, list, gpd.GeoSeries], crs: str
) -> gpd.GeoDataFrame:
    """
    Get a GeoDataFrame from a geometry and a crs

    ```python
    >>> poly = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
    >>> geodf = get_geodf(poly, crs=WGS84)
    >>> print(geodf)
                                                geometry
    0  POLYGON ((0.00000 0.00000, 0.00000 1.00000, 1....
    ```

    Args:
        geometry (Union[Polygon, list]): List of Polygons, or Polygon or bounds
        crs (str): CRS of the polygon

    Returns:
        gpd.GeoDataFrame: Geometry as a geodataframe
    """
    if isinstance(geometry, list):
        if isinstance(geometry[0], Polygon):
            pass
        else:
            try:
                geometry = [from_bounds_to_polygon(*geometry)]
            except TypeError as ex:
                raise TypeError(
                    "Give the extent as 'left', 'bottom', 'right', and 'top'"
                ) from ex
    elif isinstance(geometry, Polygon):
        geometry = [geometry]
    elif isinstance(geometry, gpd.GeoSeries):
        geometry = geometry.geometry
    else:
        raise TypeError("geometry should be a list or a Polygon.")

    return gpd.GeoDataFrame(geometry=geometry, crs=crs)


def set_kml_driver() -> None:
    """
    Set KML driver for Fiona data (use it at your own risks !)

    ```python
    >>> path = "path\\to\\kml.kml"
    >>> gpd.read_file(path)
    fiona.errors.DriverError: unsupported driver: 'LIBKML'

    >>> set_kml_driver()
    >>> gpd.read_file(path)
                   Name  ...                                           geometry
    0  CC679_new_AOI2_3  ...  POLYGON Z ((45.03532 32.49765 0.00000, 46.1947...
    [1 rows x 12 columns]
    ```

    """
    drivers = gpd.io.file.fiona.drvsupport.supported_drivers
    if "LIBKML" not in drivers:
        drivers["LIBKML"] = "rw"
    if "KML" not in drivers:  # Just in case
        drivers["KML"] = "rw"


def get_aoi_wkt(aoi_path: str, as_str: bool = True) -> Union[str, Polygon]:
    """
    Get AOI formatted as a WKT from files that can be read by Fiona (like shapefiles, ...)
    or directly from a WKT file. The use of KML has been forced (use it at your own risks !).

    See: https://fiona.readthedocs.io/en/latest/fiona.html#fiona.open

    It is assessed that:

    - only **one** polygon composes the AOI (as only the first one is read)
    - it should be specified in lat/lon (WGS84) if a WKT file is provided
    ```python
    >>> path = "path\\to\\vec.geojson"  # OK with ESRI Shapefile, geojson, WKT, KML...
    >>> get_aoi_wkt(path)
    'POLYGON Z ((46.1947755465253067 32.4973553439109324 0.0000000000000000, 45.0353174370802520 32.4976496856158974
    0.0000000000000000, 45.0355748149750283 34.1139970085580018 0.0000000000000000, 46.1956059695554089
    34.1144793800670882 0.0000000000000000, 46.1947755465253067 32.4973553439109324 0.0000000000000000))'
    ```

    Args:
        aoi_path (str): Absolute or relative path to an AOI.
            Its format should be WKT or any format read by Fiona, like shapefiles.
        as_str (bool): If True, return WKT as a str, otherwise as a shapely geometry

    Returns:
        Union[str, Polygon]: AOI formatted as a WKT stored in lat/lon
    """
    if not os.path.isfile(aoi_path):
        raise FileNotFoundError(f"AOI file {aoi_path} does not exist.")

    if aoi_path.endswith(".wkt"):
        try:
            with open(aoi_path, "r") as aoi_f:
                aoi = wkt.load(aoi_f)
        except Exception as ex:
            raise ValueError("AOI WKT cannot be read") from ex
    else:
        try:
            if aoi_path.endswith(".kml"):
                set_kml_driver()

            # Open file
            aoi_file = gpd.read_file(aoi_path)

            # Check if a conversion to lon/lat is needed
            if aoi_file.crs.srs != WGS84:
                aoi_file = aoi_file.to_crs(WGS84)

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


def get_wider_exterior(vector: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """TODO"""

    # Get the footprint max (discard small holes stored in other polygons)
    wider = vector[vector.area == np.max(vector.area)]

    # Only select the exterior of this footprint(sometimes some holes persist)
    if not wider.empty:
        poly = Polygon(list(wider.exterior.iat[0].coords))
        wider = gpd.GeoDataFrame(geometry=[poly], crs=wider.crs)

        # Resets index as we only got one polygon left which should have index 0
        wider.reset_index(inplace=True)

    return wider


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


def shapes_to_gdf(shapes: Generator, crs: str):
    """TODO"""
    # Convert results to pandas (because of invalid geometries) and save it
    pd_results = pd.DataFrame(shapes, columns=["geometry", "raster_val"])

    if not pd_results.empty:
        # Convert to proper polygons(correct geometries)
        pd_results.geometry = pd_results.geometry.apply(_to_polygons)

    # Convert to geodataframe with correct geometry
    return gpd.GeoDataFrame(pd_results, geometry=pd_results.geometry, crs=crs)


def open_gml(gml_path: str, crs: Any = WGS84) -> gpd.GeoDataFrame:
    """
    Overload to `gpd.read_file` managing empty GML files that usually throws an exception.

    Use crs=None to open naive geometries.

    Args:
        gml_path (str): GML path
        crs (Union[str, CRS, None]): Default CRS

    Returns:
        gpd.GeoDataFrame: GML vector or empty geometry
    """
    # Read the GML file
    try:
        # Discard some weird error concerning a NULL pointer that outputs a ValueError (as we already except it)
        fiona_logger = logging.getLogger("fiona._env")
        fiona_logger.setLevel(logging.CRITICAL)

        # Read mask
        mask = gpd.read_file(gml_path)

        # Manage naive geometries
        if mask.crs and crs:
            mask = mask.to_crs(crs)

        # Set fiona logger back to what it was
        fiona_logger.setLevel(logging.INFO)
    except (ValueError, UnsupportedGeometryTypeError):
        mask = gpd.GeoDataFrame(geometry=[], crs=crs)

    return mask
