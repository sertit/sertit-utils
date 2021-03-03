""" Script testing the geo_utils """
import os
import geopandas as gpd
from shapely import wkt
from CI.SCRIPTS import script_utils
from sertit import vectors

GEO_DATA = os.path.join(script_utils.get_ci_data_path(), "geo_utils")


# TODO: use geopandas.testing.assert_geoseries_equal() and assert_geodataframe_equal()

def test_vectors():
    """ Test geo functions """
    kml_path = os.path.join(GEO_DATA, "aoi.kml")
    wkt_path = os.path.join(GEO_DATA, "aoi.wkt")
    utm_path = os.path.join(GEO_DATA, "aoi.geojson")

    # KML
    vectors.set_kml_driver()  # An error will occur afterwards if this fails (we are attempting to open a KML file)

    # KML to WKT
    aoi_str_test = vectors.get_aoi_wkt(kml_path, as_str=True)
    aoi_str = 'POLYGON Z ((46.1947755465253067 32.4973553439109324 0.0000000000000000, ' \
              '45.0353174370802520 32.4976496856158974 0.0000000000000000, ' \
              '45.0355748149750283 34.1139970085580018 0.0000000000000000, ' \
              '46.1956059695554089 34.1144793800670882 0.0000000000000000, ' \
              '46.1947755465253067 32.4973553439109324 0.0000000000000000))'
    assert aoi_str == aoi_str_test

    aoi = vectors.get_aoi_wkt(kml_path, as_str=False)

    # WKT to WKT
    aoi2 = vectors.get_aoi_wkt(wkt_path, as_str=False)

    # UTM to WKT
    aoi3 = vectors.get_aoi_wkt(utm_path, as_str=False)

    assert aoi.equals(aoi2)  # No reprojection, shoul be equal
    assert aoi.almost_equals(aoi3)  # Reprojection, so almost equal
    assert wkt.dumps(aoi) == aoi_str

    # UTM and bounds
    aoi = gpd.read_file(kml_path)
    assert vectors.corresponding_utm_projection(aoi.centroid.x, aoi.centroid.y) == "EPSG:32638"
    env = aoi.envelope[0]
    assert env.bounds == vectors.from_bounds_to_polygon(*vectors.from_polygon_to_bounds(env)).bounds
    assert vectors.get_geodf(env, aoi.crs).bounds.equals(aoi.envelope.bounds)
