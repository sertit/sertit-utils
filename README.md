[![pypi](https://img.shields.io/pypi/v/sertit.svg)](https://pypi.python.org/pypi/sertit)
[![Conda](https://img.shields.io/conda/vn/conda-forge/sertit.svg)](https://anaconda.org/conda-forge/sertit)
[![Tests](https://github.com/sertit/sertit-utils/actions/workflows/test.yml/badge.svg)](https://github.com/sertit/sertit-utils/actions/workflows/test.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Apache](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sertit/sertit-utils/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5082060.svg)](https://doi.org/10.5281/zenodo.5082060)

# 🧰 Sertit-Utils

Library gathering functions for all SERTIT's projects.

Find the API documentation [**here**](https://sertit-utils.readthedocs.io/latest/).

## 🛠️ Installation

### Pip

For installing this library to your environment, please type this: `pip install sertit[full]`

`[full]` will allow you to use the whole library, but you will need to install also all the optional libraries, notably `rioxarray`, `geopandas` and `dask` (see the [pyproject file](https://github.com/sertit/sertit-utils/blob/main/pyproject.toml) for more details)

However, if you do not need everything, you can type instead:

- `pip install sertit`, if you don't need any optional libraries
- `pip install sertit[geo]` to have the geospatial stack (`geopandas`, `geoutils`, `rasterio`, `rioxarray`, `xdem`) installed
- `pip install sertit[colorlog]` to have `colorlog` installed
- `pip install sertit[dask]` to have `dask` installed

### Conda

You can install it via conda (but you will automatically have the full version):

`conda install -c conda-forge sertit`

## 🚀 Contributing

Sertit-Utils welcomes contributions, human-written or carefully reviewed AI-assisted:

- **Humans**: [CONTRIBUTING.md](https://github.com/sertit/sertit-utils/blob/main/CONTRIBUTING.md): quick start, test suite, PR conventions
- **Coding agents**: [AGENTS.md](https://github.com/sertit/sertit-utils/blob/main/AGENTS.md): hard limits, task navigation, verification & self-review
- **Technical reference**: [ARCHITECTURE.md](https://github.com/sertit/sertit-utils/blob/main/ARCHITECTURE.md): modules, build & test, pitfalls
- **AI-assisted contribution policy**: [AI_POLICY.md](https://github.com/sertit/sertit-utils/blob/main/AI_POLICY.md)

## 📝 License

**Sertit-Utils** is licensed under Apache License v2.0. See LICENSE file for details.

## 🖋️ Authors

**Sertit-Utils** has been created by [ICube-SERTIT](https://sertit.unistra.fr/).