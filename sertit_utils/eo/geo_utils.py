""" Geo tools """
import logging
import os
from typing import Union
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon, box
from rasterio import crs

WGS84 = "EPSG:4326"

LOGGER = logging.getLogger('sertit_utils')


def corresponding_utm_projection(lon: float, lat: float) -> str:
    """
    Find the EPSG code of the UTM projection from a lon/lat in WGS84.

    ```python
    corresponding_utm_projection(lon=48.6, lat=7.8)  # Strasbourg
    # >> "EPSG:32632"
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
    return "EPSG:{}".format(epsg)


def from_polygon_to_bounds(polygon: Union[Polygon, MultiPolygon]) -> (float, float, float, float):
    """
    Convert a `shapely.polygon` to its bounds, sorted as `left, bottom, right, top`.

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


def from_bounds_to_polygon(left: float, bottom: float, right: float, top: float) -> Polygon:
    """
    Convert the bounds to a `shapely.polygon`.

    Args:
        left (float): Left coordinates
        bottom (float): Bottom coordinates
        right (float): Right coordinates
        top (float): Top coordinates

    Returns:
        Polygon: Polygon corresponding to the bounds

    """
    return box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))


def get_geodf(geometry: Union[Polygon, list, gpd.GeoSeries], geom_crs: Union[crs.CRS, str]) -> gpd.GeoDataFrame:
    """
    Get a GeoDataFrame from a geometry and a crs
    Args:
        geometry (Union[Polygon, list]): List of Polygons, or Polygon or bounds
        geom_crs (Union[crs.CRS, str]): CRS of the polygon

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
                raise TypeError("Give the extent as 'left', 'bottom', 'right', and 'top'") from ex
    elif isinstance(geometry, Polygon):
        geometry = [geometry]
    elif isinstance(geometry, gpd.GeoSeries):
        geometry = [from_bounds_to_polygon(*geometry.values)]
    else:
        raise TypeError("geometry should be a list or a Polygon.")

    return gpd.GeoDataFrame(geometry=geometry, crs=geom_crs)


def set_kml_driver():
    """
    Set KML driver for Fiona data (use it at your own risks !)
    """
    drivers = gpd.io.file.fiona.drvsupport.supported_drivers
    if 'LIBKML' not in drivers:
        drivers['LIBKML'] = 'rw'
    if 'KML' not in drivers:  # Just in case
        drivers['KML'] = 'rw'


def get_aoi_wkt(aoi_path, as_str=True):
    """
    Get AOI formatted as a WKT from files that can be read by Fiona (like shapefiles, ...)
    or directly from a WKT file. The use of KML has been forced (use it at your own risks !).

    See: https://fiona.readthedocs.io/en/latest/fiona.html#fiona.open

    It is assessed that:

    - only **one** polygon composes the AOI (as only the first one is read)
    - it should be specified in lat/lon (WGS84) if a WKT file is provided

    Args:
        aoi_path (str): Absolute or relative path to an AOI.
            Its format should be WKT or any format read by Fiona, like shapefiles.
        as_str (bool): If True, return WKT as a str, otherwise as a shapely geometry

    Returns:
        str: AOI formatted as a WKT stored in lat/lon
    """
    if not os.path.isfile(aoi_path):
        raise Exception("AOI file {} does not exist.".format(aoi_path))

    if aoi_path.endswith('.wkt'):
        try:
            with open(aoi_path, 'r') as aoi_f:
                aoi = wkt.load(aoi_f)
        except Exception as ex:
            raise Exception('AOI WKT cannot be read: {}'.format(ex))
    else:
        try:
            # Open file
            aoi_file = gpd.read_file(aoi_path)

            # Check if a conversion to lon/lat is needed
            if aoi_file.crs.srs != "epsg:4326":
                aoi_file = aoi_file.to_crs('epsg:4326')

            # Get envelope polygon
            geom = aoi_file['geometry']
            if len(geom) > 1:
                LOGGER.warning("Your AOI contains several polygons. Only the first will be treated !")
            polygon = geom[0].convex_hull

            # Convert to WKT
            aoi = wkt.loads(str(polygon))

        except Exception as ex:
            raise Exception('AOI cannot be read by Fiona: {}'.format(ex))

    # Convert to string if needed
    if as_str:
        aoi = wkt.dumps(aoi)

    LOGGER.debug('Specified AOI in WKT: %s', aoi)
    return aoi
