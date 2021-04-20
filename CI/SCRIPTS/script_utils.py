import os
from enum import unique

from sertit.misc import ListEnum


@unique
class Polarization(ListEnum):
    """ SAR Polarizations """
    hh = "HH"
    vv = "VV"
    vh = "VH"
    hv = "HV"


def get_proj_path():
    """ Get project path """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_ci_data_path():
    """ Get CI DATA path """
    return os.path.join(get_proj_path(), "CI", "DATA")


RASTER_DATA = os.path.join(get_ci_data_path(), "rasters")
GEO_DATA = os.path.join(get_ci_data_path(), "vectors")
FILE_DATA = os.path.join(get_ci_data_path(), "files")
