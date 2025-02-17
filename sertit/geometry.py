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
Geometry tools

You can use this only if you have installed sertit[full] or sertit[vectors]
"""

import logging

import geopandas as gpd
import numpy as np
import shapely
from shapely import ops
from shapely.errors import GeometryTypeError
from shapely.geometry import Polygon, box
from tqdm import tqdm

from sertit import misc, vectors
from sertit.logs import SU_NAME
from sertit.types import AnyPolygonType

LOGGER = logging.getLogger(SU_NAME)


def from_polygon_to_bounds(polygon: AnyPolygonType) -> (float, float, float, float):
    """
    Convert a :code:`shapely.polygon` to its bounds, sorted as :code:`left, bottom, right, top`.

    Args:
        polygon (MultiPolygon): polygon to convert

    Returns:
        (float, float, float, float): left, bottom, right, top

    Example:
        >>> poly = Polygon(((0., 0.), (0., 1.), (1., 1.), (1., 0.), (0., 0.)))
        >>> from_polygon_to_bounds(poly)
        (0.0, 0.0, 1.0, 1.0)
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

    Args:
        left (float): Left coordinates
        bottom (float): Bottom coordinates
        right (float): Right coordinates
        top (float): Top coordinates

    Returns:
        Polygon: Polygon corresponding to the bounds

    Example:
        >>> poly = from_bounds_to_polygon(0.0, 0.0, 1.0, 1.0)
        >>> print(poly)
        'POLYGON ((1 0, 1 1, 0 1, 0 0, 1 0))'

    """
    return box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))


def get_wider_exterior(vector: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Get the wider exterior of a MultiPolygon as a Polygon

    Args:
        vector (gpd.GeoDataFrame): Polygon to simplify

    Returns:
        vector: gpd.GeoDataFrame: Wider exterior

    Example:
        >>> # Open a raw footprint
        >>> footprint_raw = vectors.read("footprint_raw.geojson")
                                                           geometry
        0         MULTIPOLYGON (((491053.524 5616778.498, 491262...
        1         MULTIPOLYGON (((491314.496 5616444.620, 491295...
        2         MULTIPOLYGON (((490783.440 5616102.457, 490923...
        >>>
        >>> # Get the wider exterior
        >>> get_wider_exterior(footprint_raw)
                                                        geometry
        0      POLYGON ((491053.524 5616778.498, 491262.302 5...
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

    Better to use :code:`gpd.make_valid` if you can.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame to repair
        verbose (bool): Verbose invalid geometries

    Returns:
        gpd.GeoDataFrame: Repaired geometries

    Example:
        >>> # Open a raw  vector with invalid geometries
        >>> raw = vectors.read("raw.geojson")
                                                           geometry
        0         MULTIPOLYGON (((491053.524 5616778.498, 491262...
        1         MULTIPOLYGON (((491314.496 5616444.620, 491295...
        2         MULTIPOLYGON (((490783.440 5616102.457, 490923...
        >>>
        >>> # Get the valid geometries
        >>> make_valid(raw)
                                                           geometry
        1         MULTIPOLYGON (((491314.496 5616444.620, 491295...
    """
    try:
        geos_logger = logging.getLogger("shapely.geos")
        previous_level = geos_logger.level
        if verbose:
            logging.debug(f"Invalid geometries:\n\t{gdf[~gdf.is_valid]}")
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
            f"'make_valid' not available in 'shapely' (version {shapely.__version__} < 1.8). "
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

    Examples:
        >>> # Open a raw footprint
        >>> footprint_raw = vectors.read("footprint_raw.geojson")
        >>> len(footprint_raw.get_coordinates())
        64757
        >>>
        >>> # Get the simplified footprint
        >>> simplified = simplify_footprint(footprint_raw, 20)
        >>> len(simplified.get_coordinates())
        29
    """
    # Number of pixels of tolerance
    tolerance = [1, 2, 4, 8, 16, 32, 64, 128, 256]

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

        # WARNING if nof_vertices > max_nof_vertices
        nof_vertices = len(value.exterior.coords)
        if nof_vertices > max_nof_vertices:
            LOGGER.warning(
                f"The number of vertices ({nof_vertices}) of your simplified footprint is higher than {max_nof_vertices}."
                f"However, it cannot be simplified further according to the given resolution ({resolution})."
            )

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

    Example:
        >>> # Open a polygon with holes
        >>> holes = vectors.read("holes.geojson")
        >>> # This polygons has interiors features, it has holes
        >>> holes.interiors
        3    [LINEARRING (491328.9981955575 5616655.8234532...
        dtype: object
        >>> no_holes = fill_polygon_holes(holes)
        Processing objects: 100%|██████████| 1/1 [00:00<00:00, 897.18it/s]
        >>> no_holes.interiors
        0    []
        dtype: object
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


def line_merge(lines: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
    """
    :code:`shapely.line_merge` algorithm applied to a GeoDataFrame.

    See the corresponding documentation for more insights about the details of this function.

    Args:
        lines (gpd.GeoDataFrame): MultiLineString as a GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame composed of (Multi)LineStrings formed by combining the lines of the input GeoDataFrame

    Example:
        >>> import geopandas as gpd
        >>> from sertit import geometry
        >>> lines = gpd.read("my_lines.shp")
        >>> poly = gpd.read("my_poly.shp")
                                                    geometry
        0  POLYGON ((491460.248 5616687.073, 491460.248 5...
        >>> geometry.split(poly, splitter=lines)
        The lines are discontinuous and don't intersect totally the input polygon: nothing is splitted
                                                    geometry
        0  POLYGON ((491460.248 5616687.073, 491460.248 5...

        >>> lines = geometry.line_merge(lines)
        >>> geometry.split(poly, splitter=lines)
        The lines are now merged into one and now intersect totally the input polygon: split is done
                                                            geometry
        0  POLYGON ((491460.248 5616687.073, 491460.248 5...
        0  POLYGON ((491055.017 5616255.823, 491053.998 5...
    """
    merge_lines = shapely.line_merge(lines.dissolve().geometry.values, **kwargs)
    return gpd.GeoDataFrame(geometry=merge_lines, crs=lines.crs).explode(
        ignore_index=True
    )


def split(polygons: gpd.GeoDataFrame, splitter: gpd.GeoDataFrame):
    """
    Split polygons with polygons or lines.

    :code:`shapely.ops.split` algorithm applied to GeoDataFrames.

    Be careful: lines have to cut the whole polygon to work!
    Use :code:`geometry.line_merge: to merge your lines if needed.

    Args:
        polygons (gpd.GeoDataFrame): Polygons to split
        splitter (gpd.GeoDataFrame): Splitter to split the polygons

    Returns:
        gpd.GeoDataFrame: Split GeoDataFrame

    Example:
        >>> import geopandas as gpd
        >>> from sertit import geometry
        >>> lines = gpd.read("my_lines.shp")
        >>> poly = gpd.read("my_poly.shp")
                                                    geometry
        0  POLYGON ((491460.248 5616687.073, 491460.248 5...
        >>> split_poly = geometry.split(poly, splitter=lines)
                                                            geometry
        0  POLYGON ((491460.248 5616687.073, 491460.248 5...
        0  POLYGON ((491055.017 5616255.823, 491053.998 5...
    """
    out = polygons.dropna(axis=1).geometry
    for _, split in splitter.iterrows():
        # Compute the boundary of the splitter polygon (to get a LineString)
        if split.geometry.area > 0:
            boundary = split.geometry.boundary
        else:
            boundary = split.geometry

        # Explode to prevent FeatureCollections
        try:
            # LineStrings
            out = (
                out.map(lambda geom: ops.split(geom, boundary).geoms).explode().dropna()  # noqa: B023
            )
        except GeometryTypeError:
            # MultiLineStrings
            for line in boundary.geoms:
                out = (
                    out.map(lambda geom: ops.split(geom, line).geoms).explode().dropna()  # noqa: B023
                )

    return gpd.GeoDataFrame(geometry=out.explode(), crs=polygons.crs)


def intersects(
    input: gpd.GeoDataFrame, other: gpd.GeoDataFrame, buffer_on_input: float = None
) -> gpd.GeoDataFrame:
    """
    Select the polygons of the input GeoDataFrame that intersects the other one and return them.

    A buffer can be added on the input GeoDataFrame to be sure the intersection is ok.
    For example, when dealing with polygons and points on their borders, sometimes the points fall a tiny bit outside the polygon boundary and is not considred as intersecting.

    :code:`gpd.intersects` algorithm applied to whole GeoDataFrames.

    Args:
        input (gpd.GeoDataFrame): Input GeoDataFrame from which the polygons will be selected
        other (gpd.GeoDataFrame): Other GeoDataFrame from that will intersect the first one
        buffer_on_input (float): Buffer size on thhe input GeoDataFrame

    Returns:
        gpd.GeoDataFrame: Polygons of the input that intersects the other GeoDataFrame

    Examples:
        >>> water = vectors.read("water.geojson")
        >>> lakes = vectors.read("lakes.geojson")
        >>> intersects(water, lakes)
                                                    geometry
        2  POLYGON ((490733.035 5616749.035, 490936.972 5...
        3  POLYGON ((491254.800 5616242.894, 491175.035 5...

        >>> water = vectors.read("water.geojson")
        >>> samples = vectors.read("samples.geojson")
        >>>
        >>> # Some samples are on the polygon boundary but for some reason fall a tiny bit outside and don't intersect the water
        >>> intersects(water, samples)
                                                    geometry
        2  POLYGON ((490733.035 5616749.035, 490936.972 5...
        >>>
        >>> # Some samples are on the polygon boundary but for some reason fall a tiny bit outside and don't intersect the water
        >>> # Use the buffer to handle this case. Use it carefully!
        >>> intersects(water, samples, buffer_on_input=0.1)
                                                    geometry
        2  POLYGON ((490733.035 5616749.035, 490936.972 5...
        3  POLYGON ((491254.800 5616242.894, 491175.035 5...
    """
    if buffer_on_input is not None:
        input_to_intersect = buffer(input, buffer_on_input)
    else:
        input_to_intersect = input

    return input[
        input_to_intersect.geometry.map(lambda x: x.intersects(other.geometry).any())
    ]


def buffer(vector: gpd.GeoDataFrame, buffer_m: float, **kwargs) -> gpd.GeoDataFrame:
    """
    Add a buffer on a vector.

    :code:`gpd.buffer` algorithm returning a GeoDataFrame instead of a GeoSeries.

    Args:
        vector (gpd.GeoDataFrame): Input vector
        buffer_m (int): Buffer size in meters.
        **kwargs: Other buffer arguments

    Returns:
        gpd.GeoDataFrame: Buffered vector

    Example:
        >>> lines = vectors.read(r"lines.shp")
        >>> lines.area
        0    0.0
        1    0.0
        dtype: float64
        >>> lines_buffer = buffer(lines, 10)
        >>> lines_buffer.area
        0    5695.213033
        1    5898.723719
        dtype: float64

    """
    vector_bfd = vector.copy()
    vector_bfd.geometry = vector.buffer(buffer_m, **kwargs)
    return vector_bfd


def nearest_neighbors(
    src_gdf: gpd.GeoDataFrame,
    candidates_gdf: gpd.GeoDataFrame,
    method: str,
    k_neighbors: int = None,
    radius: float = None,
    **kwargs,
) -> (np.ndarray, np.ndarray):
    """
    For each point in src_gdf, find the closest point in candidates_gdf and return them with their distances (in the crs coordinates).

    Closest points are:

    - if method == :code:`k_neighbors`: the k closest neighbors
    - if method == :code:`radius`: the neighbors inside this radius (in the crs coordinates, better done with projected geometries)

    Args:
        src_gdf (gpd.GeoDataFrame): Source geodataframe
        candidates_gdf (gpd.GeoDataFrame): Candidates geodataframe
        method (str): 'k_neighbors' or 'radius'
        k_neighbors (int): Number of neighbors to be looked for
        radius (float): Radius in which to find the neighbors
        **kwargs: Other args for the query

    Returns:
        (np.ndarray, np.ndarray): closest samples, distances

    Examples:
        >>> from sertit import geometry, vectors
        >>> src = vectors.read("src.shp")
        >>> candidates = vectors.read("candidates.shp")
        >>> # There is only one point in the neighborhood of each src, the others are further than 100m
        >>>
        >>> # Radius method
        >>> nearest_neighbors(src, candidates, method="radius", radius=100)
        [array([13]) array([12]) array([0])], [array([39.62574458]) array([50.37121574]) array([90.98648454])]
        >>>
        >>> # k_neighbors method
        >>> nearest_neighbors(src, candidates, method="k_neighbors", k_neighbors=1)
        [array([13]) array([12]) array([0])], [array([39.62574458]) array([50.37121574]) array([90.98648454])]
    """
    # Parse coordinates from points and insert them into a numpy array as RADIANS
    src = [(val.xy[0][0], val.xy[1][0]) for val in src_gdf.geometry.values]
    candidates = [
        (val.xy[0][0], val.xy[1][0]) for val in candidates_gdf.geometry.values
    ]

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in the crs coordinates)
    if method == "k_neighbors":
        closest_samples, distances = _get_k_nearest(
            src_points=src, candidates=candidates, k_neighbors=k_neighbors, **kwargs
        )
    else:
        closest_samples, distances = _get_radius_nearest(
            src_points=src, candidates=candidates, radius=radius, **kwargs
        )

    return closest_samples, distances


def _get_k_nearest(src_points: list, candidates: list, k_neighbors: int, **kwargs):
    """
    For each point in src_gdf, find the nearest k points in candidates_gdf and return them with their distance.

    Args:
        src_points (list): Source points
        candidates (list): Candidate points
        k_neighbors (int): Number of neighbors to be looked for
        **kwargs: Other args for the query

    Returns:
        (np.ndarray, np.ndarray): closest samples, distances
    """
    try:
        from sklearn.neighbors import BallTree
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'sklearn' the 'geometry.nearest_neighbors' function."
        ) from ex

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15)

    # Find the closest points and distances
    closest_dist, closest = tree.query(
        src_points, k=min(k_neighbors, len(candidates)), **kwargs
    )

    # Return indices and distances
    return closest, closest_dist


def _get_radius_nearest(
    src_points: list, candidates: list, radius: float, **kwargs
) -> (np.ndarray, np.ndarray):
    """
    For each point in src_gdf, find the points in candidates_gdf inside the given radius and return them with their distance.

    Args:
        src_points (list): Source points
        candidates (list): Candidate points
        radius (float): Radius for the search
        **kwargs: Other args for the query

    Returns:
        (np.ndarray, np.ndarray): closest samples, distances
    """
    try:
        from sklearn.neighbors import BallTree
    except ModuleNotFoundError as ex:
        raise ModuleNotFoundError(
            "Please install 'sklearn' the 'geometry.nearest_neighbors' function."
        ) from ex

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15)

    # Find the closest points and distances
    closest, closest_dist = tree.query_radius(
        src_points,
        r=radius,
        return_distance=True,
        **misc.select_dict(kwargs, ["count_only", "sort_results"]),
    )

    # Return indices and distances
    return closest, closest_dist


def force_2d(vect: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Have the force_2d function even with geopandas < 1.0.0"""
    try:
        return vect.force_2d()
    except AttributeError:
        vect.geometry = vect.geometry.apply(shapely.force_2d)
        return vect


def force_3d(vect: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Have the force_3d function even with geopandas < 1.0.0"""
    try:
        return vect.force_3d()
    except AttributeError:
        vect.geometry = vect.geometry.apply(shapely.force_3d)
        return vect
