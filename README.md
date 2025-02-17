[![pypi](https://img.shields.io/pypi/v/sertit.svg)](https://pypi.python.org/pypi/sertit)
[![Conda](https://img.shields.io/conda/vn/conda-forge/sertit.svg)](https://anaconda.org/conda-forge/sertit)
[![Tests](https://github.com/sertit/sertit-utils/actions/workflows/test.yml/badge.svg)](https://github.com/sertit/sertit-utils/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sertit/sertit-utils/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5082060.svg)](https://doi.org/10.5281/zenodo.5082060)

# Sertit-Utils

Library gathering functions for all SERTIT's projects.

Find the API documentation [**here**](https://sertit-utils.readthedocs.io/latest/).

## Installing

### Pip
For installing this library to your environment, please type this: `pip install sertit[full]`

`[full]` will allow you to use the whole library, but you will need to install also `rioxarray` and `geopandas`
(with GDAL installation issues on Windows, so please install them from wheels that you can
find [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)).

However, if you do not need everything, you can type instead:

- *nothing*, and you won't need `rasterio`, `rioxarray`: `pip install sertit --extra-index-url ...`
- `[rasters_rio]`, and you won't need `rioxarray`: `pip install sertit[rasters_rio] --extra-index-url ...`
- `[rasters]`: `pip install sertit[rasters] --extra-index-url ...`
- `[colorlog]`: `pip install sertit[colorlog] --extra-index-url ...` to have `colorlog` installed
- `[dask]`: `pip install sertit[dask] --extra-index-url ...` to have `dask` installed

### Conda

You can install it via conda (but you will automatically have the full version):

`conda config --env --set channel_priority strict`

`conda install -c conda-forge sertit`
