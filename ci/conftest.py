import numpy as np
import pytest
import rasterio

from ci.script_utils import KAPUT_KWARGS, LANDSAT_NAME, files_path, rasters_path
from sertit import path, rasters_rio, vectors

# PATHS


@pytest.fixture
def landsat_prod():
    return files_path() / LANDSAT_NAME


@pytest.fixture
def landsat_zip():
    return files_path() / f"{LANDSAT_NAME}.zip"


@pytest.fixture
def landsat_tar():
    return files_path() / f"{LANDSAT_NAME}.tar"


@pytest.fixture
def landsat_tar_gz():
    return files_path() / f"{LANDSAT_NAME}.tar.gz"


@pytest.fixture
def landsat_7z():
    return files_path() / f"{LANDSAT_NAME}.7z"


# RASTERS


@pytest.fixture
def raster_path():
    return rasters_path().joinpath("raster.tif")


@pytest.fixture
def dem_path():
    return rasters_path().joinpath("dem.tif")


@pytest.fixture
def mask_path():
    return rasters_path().joinpath("raster_mask.geojson")


@pytest.fixture
def mask(mask_path):
    return vectors.read(mask_path)


@pytest.fixture
def ds_name(raster_path):
    with rasterio.open(str(raster_path)) as ds:
        return path.get_filename(ds.name)


@pytest.fixture
def ds_dtype(raster_path):
    with rasterio.open(str(raster_path)) as ds:
        return getattr(np, ds.meta["dtype"])


@pytest.fixture
def raster_meta(raster_path):
    return rasters_rio.read(raster_path, **KAPUT_KWARGS)
