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
"""Script testing vector functions"""

from ci.script_utils import KAPUT_KWARGS, geometry_path, s3_env, vectors_path
from sertit import ci, geometry, vectors
from sertit.geometry import (
    buffer,
    fill_polygon_holes,
    get_wider_exterior,
    intersects,
    line_merge,
    nearest_neighbors,
    split,
)

ci.reduce_verbosity()


@s3_env
def test_get_wider_exterior():
    """Test get_wider_exterior"""
    footprint_raw_path = geometry_path().joinpath("footprint_raw.geojson")
    footprint_path = geometry_path().joinpath("footprint.geojson")
    ci.assert_geom_equal(
        get_wider_exterior(vectors.read(footprint_raw_path)),
        vectors.read(footprint_path),
    )


@s3_env
def test_simplify_footprint():
    """Test simplify footprint"""
    complicated_footprint_path = geometry_path().joinpath(
        "complicated_footprint_spot6.geojson"
    )
    max_nof_vertices = 40
    complicated_footprint = vectors.read(complicated_footprint_path)
    ok_footprint = geometry.simplify_footprint(
        complicated_footprint, resolution=1.5, max_nof_vertices=max_nof_vertices
    )
    assert len(ok_footprint.geometry.exterior.iat[0].coords) < max_nof_vertices

    # Just to test
    nof_vertices_complicated = len(
        complicated_footprint.explode(index_parts=True).geometry.exterior.iat[0].coords
    )
    assert nof_vertices_complicated > max_nof_vertices


@s3_env
def test_geometry_fct():
    """Test other geometry functions"""
    kml_path = vectors_path().joinpath("aoi.kml")
    env = vectors.read(kml_path).envelope[0]
    from_env = geometry.from_bounds_to_polygon(*geometry.from_polygon_to_bounds(env))
    assert env.bounds == from_env.bounds


@s3_env
def test_make_valid():
    """Test make valid"""
    broken_geom_path = geometry_path().joinpath("broken_geom.shp")
    broken_geom = vectors.read(broken_geom_path)
    assert len(broken_geom[~broken_geom.is_valid]) == 1
    valid = geometry.make_valid(broken_geom, verbose=True)
    assert len(valid[~valid.is_valid]) == 0
    assert len(valid) == len(broken_geom)


@s3_env
def test_fill_polygon_holes():
    """Test fill_polygon_holes"""
    water_path = geometry_path().joinpath("water.geojson")
    water_none_path = geometry_path().joinpath("water_filled_none.geojson")
    water_0_path = geometry_path().joinpath("water_filled_0.geojson")
    water_1000_path = geometry_path().joinpath("water_filled_1000.geojson")
    water = vectors.read(water_path)

    ci.assert_geom_equal(fill_polygon_holes(water), vectors.read(water_none_path))
    ci.assert_geom_equal(fill_polygon_holes(water, 0), vectors.read(water_0_path))
    ci.assert_geom_equal(
        fill_polygon_holes(water, threshold=1000), vectors.read(water_1000_path)
    )


@s3_env
def test_split():
    """Test split"""
    water_path = geometry_path().joinpath("water.geojson")
    water = vectors.read(water_path)

    # No MultiLineStrings
    footprint_path = geometry_path().joinpath("footprint_split.geojson")
    water_split_path = geometry_path().joinpath("water_split.geojson")
    ci.assert_geom_equal(
        split(water, vectors.read(footprint_path)), vectors.read(water_split_path)
    )

    # With MultiLineStrings
    footprint_raw_path = geometry_path().joinpath("footprint_raw.geojson")
    water_split_raw_path = geometry_path().joinpath("water_split_raw.geojson")
    ci.assert_geom_equal(
        split(water, vectors.read(footprint_raw_path)),
        vectors.read(water_split_raw_path),
    )

    # Test with lines as splitter (with and without line_merge)
    # Without line_merge: doesn't split anything
    lines_raw_path = geometry_path().joinpath("lines.shp")
    lines_raw = vectors.read(lines_raw_path)
    ci.assert_geom_equal(
        split(water, lines_raw),
        water,
    )

    # With line_merge: works
    water_split_line_path = geometry_path().joinpath("water_split_line.geojson")
    lines_merged = line_merge(lines_raw)
    ci.assert_geom_equal(
        split(water, lines_merged),
        vectors.read(water_split_line_path),
    )


@s3_env
def test_intersects():
    """Test intersects"""
    water_path = geometry_path().joinpath("water.geojson")
    lakes_path = geometry_path().joinpath("lakes.geojson")

    inter = intersects(vectors.read(lakes_path), vectors.read(water_path))
    ci.assert_val(inter.index, [2, 3], "Index")


@s3_env
def test_buffer():
    """Test buffer"""
    water_path = geometry_path().joinpath("water.geojson")
    water = vectors.read(water_path)
    buffer_true = water.copy()
    buffer_true.geometry = water.buffer(10)

    ci.assert_geom_equal(buffer(water, 10), buffer_true)


@s3_env
def test_nearest_neighbors():
    """Test nearest_neighbors"""
    src_path = geometry_path().joinpath("source.geojson")
    candidates_path = geometry_path().joinpath("candidates.geojson")

    src = vectors.read(src_path)
    candidates = vectors.read(candidates_path)

    # Radius
    radius = 100
    closest, distances = nearest_neighbors(
        src, candidates, method="radius", radius=radius, **KAPUT_KWARGS
    )
    for curr_closest, curr_dist in zip(closest, distances):
        ci.assert_val(len(curr_closest), 1, "length")
        assert curr_dist[0] < radius, (
            f"distance superior to radius: {curr_dist[0]} > {radius}"
        )

    # Nof neighbors
    nof_neighbors = 1
    closest, distances = nearest_neighbors(
        src, candidates, method="k_neighbors", k_neighbors=nof_neighbors
    )
    for curr_closest, curr_dist in zip(closest, distances):
        ci.assert_val(len(curr_closest), nof_neighbors, "length")
        assert curr_dist[0] < radius, (
            f"distance superior to wanted distance: {curr_dist[0]} > {radius}"
        )

    # Ensure it works with k > nof candidates
    closest, distances = nearest_neighbors(
        src, candidates, method="k_neighbors", k_neighbors=len(candidates) + 10
    )


def test_force_2_or_3d():
    """Force 2D or 3D"""
    aoi_kml = vectors.read(vectors_path().joinpath("aoi.kml"))

    assert aoi_kml.has_z.any()

    # Check equality
    ci.assert_geom_equal(
        aoi_kml, geometry.force_3d(geometry.force_2d(aoi_kml)), ignore_z=False
    )

    # Check Z type
    assert not geometry.force_2d(aoi_kml).has_z.any()
    assert geometry.force_3d(geometry.force_2d(aoi_kml)).has_z.any()
