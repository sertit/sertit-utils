import os
import rasterio
import numpy as np


def get_proj_path():
    """ Get project path """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_ci_data_path():
    """ Get CI DATA path """
    return os.path.join(get_proj_path(), "CI", "DATA")


def assert_raster_equals(path_1: str, path_2: str) -> None:
    """
    Assert that 2 raster are equal
    Args:
        path_1 (str): Raster 1
        path_2 (str): Raster 2
    """
    with rasterio.open(path_1) as dst_1:
        with rasterio.open(path_2) as dst_2:
            assert dst_1.meta == dst_2.meta
            np.testing.assert_array_equal(dst_1.read(), dst_2.read())
