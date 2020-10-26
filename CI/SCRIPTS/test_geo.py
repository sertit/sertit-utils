""" Script testing the geo_utils """
import os
import geopandas as gpd
from shapely import wkt
from CI.SCRIPTS import script_utils
from sertit_utils.eo import geo_utils


def test_geo():
    """ Test geo functions """
    aoi_path = os.path.join(script_utils.get_ci_data_path(), "aoi.kml")

    # KML
    geo_utils.set_kml_driver()  # An error will occur afterwards if this fails (we are attempting to open a KML file)

    # AOI WKT
    aoi = geo_utils.get_aoi_wkt(aoi_path, as_str=False)
    aoi_str = 'POLYGON Z ((46.1947755465253067 32.4973553439109324 0.0000000000000000, ' \
              '45.0353174370802520 32.4976496856158974 0.0000000000000000, ' \
              '45.0355748149750283 34.1139970085580018 0.0000000000000000, ' \
              '46.1956059695554089 34.1144793800670882 0.0000000000000000, ' \
              '46.1947755465253067 32.4973553439109324 0.0000000000000000))'

    assert wkt.dumps(aoi) == aoi_str

    # UTM and bounds
    aoi = gpd.read_file(aoi_path)
    assert geo_utils.corresponding_utm_projection(aoi.centroid.x, aoi.centroid.y) == "EPSG:32638"
    env = aoi.envelope[0]
    assert env.bounds == geo_utils.from_bounds_to_polygon(*geo_utils.from_polygon_to_bounds(env)).bounds
    assert geo_utils.get_geodf(env, aoi.crs).bounds.equals(aoi.envelope.bounds)
