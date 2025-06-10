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

import logging
import os
import tempfile
import warnings

import geopandas as gpd
import pytest
from rasterio import CRS
from shapely import wkt

from ci.script_utils import KAPUT_KWARGS, files_path, s3_env, vectors_path
from sertit import ci, files, path, vectors
from sertit.logs import SU_NAME
from sertit.vectors import EPSG_4326, DataSourceError

LOGGER = logging.getLogger(SU_NAME)

ci.reduce_verbosity()


def _assert_attributes(vec: gpd.GeoDataFrame, vec_path):
    ci.assert_val(vec.attrs["path"], str(vec_path), "path")
    ci.assert_val(vec.attrs["name"], path.get_filename(vec_path), "name")


@s3_env
def test_vectors():
    """Test geo functions"""
    shp_path = vectors_path().joinpath("aoi.shp")
    kml_path = vectors_path().joinpath("aoi.kml")
    wkt_path = vectors_path().joinpath("aoi.wkt")
    utm_path = vectors_path().joinpath("aoi.geojson")
    ci.assert_geom_equal(shp_path, utm_path)  # Test shp

    # Test 3D vectors
    # with pytest.raises(AssertionError):
    #     ci.assert_geom_equal(shp_path, utm_path, ignore_z=False)

    # KML
    vectors.set_kml_driver()  # An error will occur afterwards if this fails (we are attempting to open a KML file)

    # KML to WKT
    aoi_str_test = vectors.get_aoi_wkt(kml_path, as_str=True)
    aoi_str = (
        "POLYGON Z ((46.1947755465253067 32.4973553439109324 0.0000000000000000, "
        "45.0353174370802520 32.4976496856158974 0.0000000000000000, "
        "45.0355748149750283 34.1139970085580018 0.0000000000000000, "
        "46.1956059695554089 34.1144793800670882 0.0000000000000000, "
        "46.1947755465253067 32.4973553439109324 0.0000000000000000))"
    )
    assert aoi_str == aoi_str_test

    aoi = vectors.get_aoi_wkt(kml_path, as_str=False)

    # WKT to WKT
    aoi2 = vectors.get_aoi_wkt(wkt_path, as_str=False)

    # UTM to WKT
    aoi3 = vectors.get_aoi_wkt(utm_path, as_str=False)

    assert aoi.equals(aoi2)  # No reprojection, should be equal
    assert aoi.equals_exact(
        aoi3, tolerance=0.5 * 10**6
    )  # Reprojection, so almost equal
    assert wkt.dumps(aoi) == aoi_str

    # UTM and bounds
    aoi = vectors.read(kml_path, **KAPUT_KWARGS)
    _assert_attributes(aoi, kml_path)

    with pytest.deprecated_call():
        assert (
            vectors.corresponding_utm_projection(aoi.centroid.x, aoi.centroid.y)
            == "EPSG:32638"
        )
        assert CRS.from_string("EPSG:32638") == vectors.to_utm_crs(
            aoi.centroid.x, aoi.centroid.y
        )

    env = aoi.envelope[0]

    # Test kwargs (should be slightly not equal toi AOI to prove bbox does sth)
    with pytest.raises(AssertionError):
        ci.assert_geom_equal(vectors.read(kml_path, bbox=env).geometry, aoi.envelope)

    # GeoDataFrame
    geodf = vectors.get_geodf(env, aoi.crs)  # GeoDataFrame from Polygon
    ci.assert_geom_equal(geodf.geometry, aoi.envelope)
    ci.assert_geom_equal(
        vectors.get_geodf(geodf.geometry, aoi.crs), geodf
    )  # GeoDataFrame from Geoseries
    ci.assert_geom_equal(
        vectors.get_geodf([env], aoi.crs), geodf
    )  # GeoDataFrame from list of poly

    with pytest.raises(TypeError):
        vectors.get_geodf([1, 2, 3, 4, 5], aoi.crs)
    with pytest.raises(TypeError):
        vectors.get_geodf([1, 2], aoi.crs)


@s3_env
def test_kml():
    """Test KML files"""
    LOGGER.debug("Open GEARTH_POLY.kml")
    # Just check there is no issue when opening this file
    kml_path = vectors_path().joinpath("GEARTH_POLY.kml")
    kml = vectors.read(kml_path)
    _assert_attributes(kml, kml_path)
    assert not kml.empty

    # Check equivalence between two vector types (complex vector)
    LOGGER.debug("Open EMSR680_AOI03_DEL_PRODUCT_observedEventA_v1.kml")
    kml_path = vectors_path().joinpath(
        "EMSR680_AOI03_DEL_PRODUCT_observedEventA_v1.kml"
    )
    json_path = vectors_path().joinpath(
        "EMSR680_AOI03_DEL_PRODUCT_observedEventA_v1.json"
    )

    # Read vectors
    kml = vectors.read(kml_path).explode(ignore_index=True)
    json = vectors.read(json_path).explode(ignore_index=True)

    # Check attributes
    assert not kml.empty
    _assert_attributes(kml, kml_path)
    _assert_attributes(json, json_path)

    # Check if equivalent
    ci.assert_geom_almost_equal(json, kml, decimal=6)

    # Just check there is no issue when opening this file
    LOGGER.debug("Open ICEYE_X2_QUICKLOOK_SC_124020_20210827T162211.kml")
    kml_path = vectors_path().joinpath(
        "ICEYE_X2_QUICKLOOK_SC_124020_20210827T162211.kml"
    )
    kml = vectors.read(kml_path)
    _assert_attributes(kml, kml_path)
    assert not kml.empty


@s3_env
def test_kmz():
    """Test KMZ files"""
    kmz_path = vectors_path().joinpath("AOI_kmz.kmz")
    gj_path = vectors_path().joinpath("AOI_kmz.geojson")

    # Read vectors
    kmz = vectors.read(kmz_path)
    gj = vectors.read(gj_path)

    # Check attributes
    _assert_attributes(kmz, kmz_path)
    _assert_attributes(gj, gj_path)

    # Check if equivalent
    assert all(
        gj.geometry.geom_equals_exact(
            kmz.to_crs(gj.crs).geometry, tolerance=0.5 * 10**-6
        )
    )


@s3_env
def test_gml():
    """Test GML functions"""
    empty_gml = vectors_path().joinpath("empty.GML")
    not_empty_gml = vectors_path().joinpath("not_empty.GML")
    naive_gml = vectors_path().joinpath("naive.GML")
    not_empty_true_path = vectors_path().joinpath("not_empty_true.geojson")

    # Empty
    empty_gdf = vectors.read(empty_gml, crs=EPSG_4326)
    assert empty_gdf.empty
    assert empty_gdf.crs == EPSG_4326
    _assert_attributes(empty_gdf, empty_gml)

    # Not empty
    not_empty_true = vectors.read(not_empty_true_path)
    not_empty = vectors.read(not_empty_gml, crs=not_empty_true.crs)
    ci.assert_geom_equal(not_empty, not_empty_true)

    # Check attrs
    _assert_attributes(not_empty_true, not_empty_true_path)
    _assert_attributes(not_empty, not_empty_gml)

    # Naive
    naive = vectors.read(naive_gml)
    assert naive.crs is None
    _assert_attributes(naive, naive_gml)


@s3_env
def test_write():
    vect_paths = [
        vectors_path().joinpath("aoi.shp"),
        vectors_path().joinpath("aoi.kml"),
        vectors_path().joinpath("aoi.geojson"),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for vect_path in vect_paths:
            vect = vectors.read(vect_path)
            vect_out_path = os.path.join(tmp_dir, os.path.basename(vect_path))
            vectors.write(vect, vect_out_path)
            vect_out = vectors.read(vect_out_path)

            ci.assert_geom_equal(vect_out, vect)


@s3_env
def test_copy():
    shpfile = vectors_path().joinpath("aoi.shp")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Assert normal copy will fail
        with pytest.raises(DataSourceError):
            gpd.read_file(files.copy(shpfile, tmp_dir))

        # Assert vector copy will open in geopandas
        gpd.read_file(vectors.copy(shpfile, tmp_dir))


@s3_env
def test_utm_context_manager():
    """Test UTM context manager."""
    warning_msg = r"Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect.*$"

    # Open a geographic vector
    vect = vectors.read(vectors_path().joinpath("aoi.kml"))

    # Check the vector is geographic
    assert vect.crs.is_geographic

    # Check that centroid function warns possible issues
    with pytest.warns(UserWarning, match=warning_msg):
        c = vect.centroid
        assert c.crs.is_geographic

    # Convert it
    with vectors.utm_crs(vect) as utm_vect:
        # Check the vector is projected
        assert utm_vect.crs.is_projected

        # Check no warning is now launched
        # https://docs.pytest.org/en/7.0.x/how-to/capture-warnings.html#additional-use-cases-of-warnings-in-tests
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c2 = utm_vect.centroid
            assert c2.crs.is_projected

        # Add the centroid as a new column
        utm_vect["centroid_utm"] = c2

    # Check the CRS is back to normal
    assert vect.crs.is_geographic

    # Assert the column still exists
    assert vect["centroid_utm"].equals(c2)


def test_read_archived():
    """Test archived vectors"""
    landsat = "LM05_L1TP_200030_20121230_20200820_02_T2_CI"
    map_overlay = "map-overlay.kml"
    map_overlay_regex = ".*{}".format(map_overlay.replace(".", r"\."))
    map_overlay_extracted_path = files_path() / landsat / map_overlay
    zip_landsat = files_path() / f"{landsat}.zip"
    tar_landsat = files_path() / f"{landsat}.tar"

    map_overlay_extracted = vectors.read(map_overlay_extracted_path)

    ci.assert_geom_equal(
        map_overlay_extracted, vectors.read(f"{zip_landsat}!{landsat}/{map_overlay}")
    )
    ci.assert_geom_equal(
        map_overlay_extracted,
        vectors.read(zip_landsat, archive_regex=map_overlay_regex),
    )
    ci.assert_geom_equal(
        map_overlay_extracted,
        vectors.read(tar_landsat, archive_regex=map_overlay_regex),
    )

    file_list = path.get_archived_file_list(tar_landsat)
    ci.assert_geom_equal(
        map_overlay_extracted,
        vectors.read(tar_landsat, archive_regex=map_overlay_regex, file_list=file_list),
    )


def test_window():
    """Test read with window"""
    aoi_path = vectors_path() / "areaOfInterestA.geojson"
    vect_path = vectors_path() / "hydrographyA.geojson"

    aoi = vectors.read(aoi_path)
    vect = vectors.read(vect_path)
    vect_aoi = vectors.read(vect_path, window=aoi)
    vect_aoi_path = vectors.read(vect_path, window=aoi_path)
    vect_aoi_bounds = vectors.read(vect_path, window=aoi.total_bounds)
    vect_aoi_bbox = vectors.read(vect_path, bbox=tuple(aoi.total_bounds))

    # Tests
    ci.assert_geom_equal(vect_aoi, vect_aoi_path)
    ci.assert_geom_equal(vect_aoi, vect_aoi_bounds)
    ci.assert_geom_equal(vect_aoi, vect_aoi_bbox)

    with pytest.raises(AssertionError):
        ci.assert_geom_equal(vect, vect_aoi)

    # Manage exception by default and with reversed engine
    if vectors.is_geopandas_1_0():
        # Default engine is pyogrio, so test here fiona
        ex = TypeError
        with pytest.raises(ValueError):
            vectors.read(vect_path, bbox=aoi.total_bounds, engine="fiona")
    else:
        # Default engine is fiona, so test here pyogrio
        ex = ValueError
        with pytest.raises(TypeError):
            vectors.read(vect_path, bbox=aoi.total_bounds, engine="pyogrio")

    # Test default engine
    with pytest.raises(ex):
        vectors.read(vect_path, bbox=aoi.total_bounds)


def test_read_gdb():
    """Test read from GDB"""
    gdb_path = vectors_path() / "EMSR712_04WESERRIVERSOUTH_DelineationMap_MONIT_06.gdb"
    layer = "B1_observed_event_a"

    sertit_gdb = vectors.read(gdb_path, layer=layer)
    gpd_gdb = gpd.read_file(str(gdb_path), layer=layer)

    ci.assert_geom_equal(sertit_gdb, gpd_gdb)
    _assert_attributes(sertit_gdb, gdb_path)


def test_read_dbf():
    """Test read from GDB"""
    # DataFrame DBF (just check it works)
    dbf_path = vectors_path() / "a0_source.dbf"
    vectors.read(dbf_path)

    # GeoDataFrame DBF
    dbf_path = vectors_path() / "aoi.dbf"

    fiona = vectors.read(dbf_path, engine="fiona")
    pyogrio = vectors.read(dbf_path, engine="pyogrio")

    ci.assert_geom_equal(fiona, pyogrio)
    _assert_attributes(fiona, dbf_path)
    _assert_attributes(pyogrio, dbf_path)
