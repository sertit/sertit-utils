# SERTIT Utils

Library gathering utils functions for all SERTIT's projects.

## Documentation

An HTML documentation is provided to document the code.
It can be found in `docs`. 
To consult it, just open the `index.html` file in a web browser (you need first to clone this project)
To generate the HTML documentation, just type `pdoc sentineldownload -o docs\html -f --html`

## CORE
### File
File gathering file-related functions:
- paths
- file extraction
- file name
- copy/remove
- find files
- JSON/pickles
- hash

### Log
- Init simple logger
- Create complex logger (file and stream + color)
- Shutdown logger

### Sys
- Set up PROJ environment variables
- Run a command line
- Get a function name
- Change current directory (`cd`) as a context manager
- Test if in docker

### Type
- Conversion from string to bool, logging level, list, list of dates...
- Function on lists: convert a list to a dict, remove empty values...
- Function on dicts: nested set, check mandatory keys, find by key
- Case conversion (`snake_case` to/from `CamelCase`) 

## EO

### Geo
- Load an AOI as WKT
- Get UTM projection from lat/lon
- Manage bounds and polygons
- Get `geopandas.Geodataframe` from polygon and CRS

### Raster
Basically, these functions are overloads of rasterio's functions:
- Get extent of a raster
- Read/write overload of rasterio functions
- Masking with masked array
- Collocation (superimpose)
- Vectorization
- Get the path of the BEAM-DIMAP image that can be read by rasterio

## Generation
In order to generate a distribution, just upgrade the version in `setup.py` and run the command `python setup.py sdist bdist_wheel`.
To upload the pypi package, just type `twine upload --config-file .pypirc --repository gitlab dist\*` from the root of this project.
