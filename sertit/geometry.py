# -*- coding: utf-8 -*-
# Copyright 2023, SERTIT-ICube - France, https://sertit.unistra.fr/
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
Geometry tools

You can use this only if you have installed sertit[full] or sertit[vectors]
"""
import logging

import numpy as np
from tqdm import tqdm

from sertit.types import AnyPolygonType

try:
    import geopandas as gpd
    from shapely.geometry import Polygon, box
except ModuleNotFoundError as ex:
    raise ModuleNotFoundError(
        "Please install 'geopandas' to use the rasters package."
    ) from ex

from sertit import vectors
from sertit.logs import SU_NAME

LOGGER = logging.getLogger(SU_NAME)


def from_polygon_to_bounds(polygon: AnyPolygonType) -> (float, float, float, float):
    """
    Convert a :code:`shapely.polygon` to its bounds, sorted as :code:`left, bottom, right, top`.

    .. code-block:: python

        >>> poly = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
        >>> from_polygon_to_bounds(poly)
        (0.0, 0.0, 1.0, 1.0)

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
    Convert the bounds to a :code:`shapely.polygon`.

    .. code-block:: python

        >>> poly = from_bounds_to_polygon(0.0, 0.0, 1.0, 1.0)
        >>> print(poly)
        'POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))'

    Args:
        left (float): Left coordinates
        bottom (float): Bottom coordinates
        right (float): Right coordinates
        top (float): Top coordinates

    Returns:
        Polygon: Polygon corresponding to the bounds

    """
    return box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))


def get_wider_exterior(vector: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Get the wider exterior of a MultiPolygon as a Polygon

    Args:
        vector (gpd.GeoDataFrame): Polygon to simplify

    Returns:
        vector: gpd.GeoDataFrame: Wider exterior
    """
    vector = vector.explode(index_parts=True)

    # Get the footprint max (discard small holes stored in other polygons)
    wider = vector[vector.area == np.max(vector.area)]

    # Only select the exterior of this footprint(sometimes some holes persist)
    if not wider.empty:
        poly = Polygon(list(wider.exterior.iat[0].coords))
        wider = gpd.GeoDataFrame(geometry=[poly], crs=wider.crs)

        # Resets index as we only got one polygon left which should have index 0
        wider.reset_index(inplace=True)

    return wider


def make_valid(gdf: gpd.GeoDataFrame, verbose=False) -> gpd.GeoDataFrame:
    """
    Repair geometries from a dataframe.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to repair
        verbose (bool): Verbose invalid geometries

    Returns:
        gpd.GeoDataFrame: Repaired geometries
    """
    try:
        geos_logger = logging.getLogger("shapely.geos")
        previous_level = geos_logger.level
        if verbose:
            logging.debug("Invalid geometries:\n" f"\t{gdf[~gdf.is_valid]}")
        else:
            geos_logger.setLevel(logging.CRITICAL)

        # Discard self-intersection and null geometries
        from shapely.validation import make_valid

        gdf.geometry = gdf.geometry.apply(make_valid)

        if not verbose:
            geos_logger.setLevel(previous_level)
    except ImportError:
        import shapely

        LOGGER.warning(
            f"make_valid not available in shapely (version {shapely.__version__} < 1.8). "
            f"The obtained vector may be broken !"
        )

    return gdf


def simplify_footprint(
    footprint: gpd.GeoDataFrame, resolution: float, max_nof_vertices: int = 50
) -> gpd.GeoDataFrame:
    """
    Simplify footprint.

    Set a number of maximum vertices and this function will try to simplify the footprint to have less than this number of vertices.
    The tolerance will grow to try to respect this number of vertices.

    This function will loop over a number of pixels of tolerence [1, 2, 4, 8, 16, 32, 64] (tolerance of gpd.simplify == resolution * tol_pix)
    If in the end, the number of vertices is still too high, a warning will be emitted.

    Args:
        footprint (gpd.GeoDataFrame): Footprint to be simplified
        resolution (float): Corresponding resolution
        max_nof_vertices (int): Maximum number of vertices of the wanted footprint

    Returns:
        gpd.GeoDataFrame: Simplified footprint
    """
    # Number of pixels of tolerance
    tolerance = [1, 2, 4, 8, 16, 32, 64]

    # Process only if given footprint is too complex (too many vertices)
    def simplify_geom(value):
        nof_vertices = len(value.exterior.coords)
        if nof_vertices > max_nof_vertices:
            for tol in tolerance:
                # Simplify footprint
                value = value.simplify(
                    tolerance=tol * resolution, preserve_topology=True
                )

                # Check if OK
                nof_vertices = len(value.exterior.coords)
                if nof_vertices <= max_nof_vertices:
                    break
        return value

    footprint = footprint.explode(index_parts=True)
    footprint.geometry = footprint.geometry.apply(simplify_geom)

    return footprint


def fill_polygon_holes(
    gpd_results: gpd.GeoDataFrame, threshold: float = None
) -> gpd.GeoDataFrame:
    """
    Fill holes over a given threshold on the hole area (in meters) for all polygons of a GeoDataFrame.
    If the threshold is set to None, every hole is filled.

    Args:
        gpd_results (gpd.GeoDataFrame): Geodataframe filled whith drilled polygons
        threshold (float): Holes area threshold, in meters. If set to None, every hole is filled.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with filled holes
    """

    def _fill_polygon(polygon: Polygon, threshold: float = None):
        """
        Function used in apply in fill_polygon_holes
        """
        if threshold is not None:
            if threshold > 0 and len(polygon.interiors) > 0:
                new_interiors = list(polygon.interiors)
                for interior in polygon.interiors:
                    interior_poly = Polygon(np.array(interior.coords))
                    if interior_poly.area < threshold:
                        new_interiors.remove(interior)
                return Polygon(polygon.exterior, new_interiors)
            else:
                return polygon
        else:
            return Polygon(polygon.exterior)

    # Ensure the geometries are valid
    gpd_results = make_valid(gpd_results)

    # Check if vector is projected, if not convert it
    with vectors.utm_crs(gpd_results) as utm_results:
        # Explode the multipolygons
        utm_results = utm_results.explode(index_parts=False)

        # Keep only the exterior
        tqdm.pandas(desc="Processing objects")
        utm_results.geometry = utm_results.geometry.progress_apply(
            _fill_polygon, args=(threshold,)
        )

        # Dissolve and explode to remove polygons in polygons
        utm_results = utm_results.dissolve()
        gpd_results = utm_results.explode(index_parts=False)

    # Write back to file
    return gpd_results
