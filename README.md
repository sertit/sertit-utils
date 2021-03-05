Library gathering utils functions for all SERTIT's projects.

# What is in it ?
## Files

File gathering file-related functions:

- paths
- Create archive
- Add folder to zip file
- file extraction
- file name
- copy/remove
- find files
- JSON/pickles
- hash

## Logs
- Init simple logger
- Create complex logger (file and stream + color)
- Shutdown logger

## Misc
- Function on lists: convert a list to a dict, remove empty values...
- Function on dicts: nested set, check mandatory keys, find by key
- Run a command line
- Get a function name
- Test if in docker
- Change current directory (`cd`) as a context manager

## Strings
- Conversion from string to bool, logging level, list, list of dates...
- Convert the string to be usable in command line
- Case conversion (`snake_case` to/from `CamelCase`) 

## Vectors
- Load an AOI as WKT
- Get UTM projection from lat/lon
- Manage bounds and polygons
- Get `geopandas.Geodataframe` from polygon and CRS

## Rasters
Basically, these functions are overloads of rasterio's functions:

- Get extent and footprint of a raster
- Read/write overload of rasterio functions
- Masking and cropping with masked array
- Collocation (superimpose)
- Sieving
- Vectorization and get nodata vector
- Merge rasters (as GTiff and VRT)
- Get the path of the BEAM-DIMAP image that can be read by rasterio
- Manage bit arrays

## Network
- Standard [Exponential Backoff](https://en.wikipedia.org/wiki/Exponential_backoff) algorithm 

# Installing

For installing this library to your environment, please type this :
- in your console: `pip install sertit[full] --extra-index-url https://gitlab-deploy-token:4ieKmsaqk4zLfM3WLxF4@code.sertit.unistra.fr/api/v4/projects/134/packages/pypi/simple`
- type this in your `requirement.txt`:  
    `--extra-index-url https://gitlab-deploy-token:4ieKmsaqk4zLfM3WLxF4@code.sertit.unistra.fr/api/v4/projects/134/packages/pypi/simple sertit[full]`

:bulb:
`[full]` will allow you to use the whole library, but you will need to install also`rasterio` and `geopandas` 
(with GDAL installation issues on Windows, so please install them from wheels that you can find [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)). 
However, if you do not need everything, you can type instead:
- *nothing*, and you won't need neither `rasterio` nor `geopandas`: `pip install sertit --extra-index-url ...`
- *vectors*, and you won't need `rasterio`: `pip install sertit[vectors] --extra-index-url ...`
- *rasters*, and you won't need `geopandas`: `pip install sertit[rasters] --extra-index-url ...` 

# Documentation

An HTML documentation is provided to document the code.
It can be found:
- online [here](https://sertit.pages.sertit.unistra.fr/sertit-utils/),
- on git, in `docs`.  
  To consult it, just open the `index.html` file in a web browser (you need first to clone this project)  
  To generate the HTML documentation, just type `pdoc sertit -o docs\html -f --html -c sort_identifiers=False`

# Packaging
In order to generate a distribution, just upgrade the version in `setup.py` and run the command `python setup.py sdist bdist_wheel`.
To upload the pypi package, just type `twine upload --config-file .pypirc --repository gitlab dist\*` from the root of this project.
