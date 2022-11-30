# Release History

## 1.XX.Y (20YY-MM-DD)

## 1.20.3 (2022-11-30)

- FIX: Ensure that attributes and encoding are propagated through `rasters` functions
- FIX: Allow the user to pass tags in `rasters(_rio).write`

## 1.20.2 (2022-11-29)

- FIX: Add other double extensions to discard in `files.get_filename`
- FIX: Add the possibility to use other double extensions in `files.get_filename`

## 1.20.1 (2022-11-29)

- FIX: Get proper filename in `files.get_filename` for point-separated names of folder/files

## 1.20.0 (2022-11-29)

- **ENH: Add a `xml.dict_to_xml` function**
- CI: Updating versions of pre-commit hooks

## 1.19.6 (2022-11-28)

- FIX: KML reading `vectors.read` has better log if `ogr2ogr` isn't available in user's PATH.
- FIX: Added a fallback using geopandas raw rfeading of KML file if `ogr2ogr` isn't available in user's PATH.
- CI: Using actions/checkout@v3

## 1.19.5 (2022-11-21)

- FIX: Fix `files.to_abspath` new feature...

## 1.19.4 (2022-11-21)

- FIX: Allow the user to choose if `files.to_abspath` raise a FileNotFoundError if the file doesn't exist.

## 1.19.3 (2022-11-21)

- FIX: Force vector conversion to dataset CRS in `rasters(_rio).rasterize`
- DOC: Add a History page
- CI: Correct gitlab-ci file

## 1.19.2 (2022-10-31)

- FIX: Add predictors for compression in `rasters(_rio).write`: 3 for floating point data, 2 for others. (see https://kokoalberti.com/articles/geotiff-compression-optimization-guide/)
- CI: Update script versions in GitHub actions

## 1.19.1 (2022-10-28)

- FIX: Fixing nodata value in `rasters.write`

## 1.19.0 (2022-10-10)

- **ENH: Add a `files.get_ext` function to mirror `get_filename`**
- **ENH: Add a `vectors.ogr2geojson` function to convert tricky vector files to GeoJSON using OGR fallbacks**
- FIX: Handling GML CRS errors (i.e. with `urn:ogc:def:derivedCRSType`) with some GDAL versions

## 1.18.3 (2022-10-07)

- FIX: Fixing absolute paths for non-existing files in `rasters(_rio).merge_vrt` (again)

## 1.18.2 (2022-10-07)

- FIX: Fixing absolute paths for non-existing files in `rasters(_rio).merge_vrt`

## 1.18.1 (2022-10-07)

- FIX: Enabling the option of absolute or relative paths in `rasters(_rio).merge_vrt`
- FIX: Fix issue with too long command line with `rasters(_rio).merge_vrt` (VRT with too much files)

## 1.18.0 (2022-09-28)

- **ENH: Add a `xml` folder grouping some helpers for `lxml.etree`** such as:
    - `read`
    - `read_archive`
    - `write`
    - `add`
    - `remove`
    - `update_attrib`
    - `update_txt`
    - `update_txt_fct`
    - `convert_to_xml`
    - `df_to_xml`

## 1.17.1 (2022-09-12)

- FIX: Set `BIGTIFF=YES` when needed in memfile (`IF_NEEDED` is not sufficient)
- OPTIM: Reduce memory usage when passing xarrays to `rasters_rio` functions

## 1.17.0 (2022-09-12)

- FIX: Allow BIGTIFF in memfiles
- FIX: Do not import `rasterio` and `geopandas` for `ci` functions that don't need it
- FIX: Fixing pandas FutureWarning `The frame.append method is deprecated and will be removed from pandas in a future version.`
- DEPS: Drop support of Python 3.7

## 1.16.1 (2022-08-30)

- FIX: Do not call fiona drivers through geopandas in `vectors.set_kml_driver`

## 1.16.0 (2022-08-26)

- **ENH: Adding useful AXA utils functions (`display.scale_to_uint8` and `strings.is_uuid`) [#2](https://git.unistra.fr/sertit/sertit-utils/-/issues/2)**
- OTHER: Log stack path when missing index in `rasters.read`

## 1.15.0 (2022-08-24)

- **ENH: Creating a `vectors.make_valid` function based on shapely's [#6](https://git.unistra.fr/sertit/sertit-utils/-/issues/6)**
- **ENH: Creating a `rasters_rio.reproject_match` function based on rioxarray's [#3](https://git.unistra.fr/sertit/sertit-utils/-/issues/3)**
- **ENH: Creating a `rasters(_rio).rasterize` function base on rasterio's [#5](https://git.unistra.fr/sertit/sertit-utils/-/issues/5)**
- OPTIM: Do not copy array whith `rasters.set_nodata`

## 1.14.1(2022-05-11)

- FIX: Manage `nodata` keyword in `rasters_rio`

## 1.14.0(2022-04-26)

- **ENH: Add `get_archived_path` function in `files`**
- FIX: Add `errno.EINVAL` error in `files.is_writable`
- DOC: Remove `Use it like OTB SuperImpose` from documentation
- DOC: Update theme

## 1.13.2(2022-04-13)

- FIX: Add other loggers to `ci.reduce_verbosity`

## 1.13.1(2022-04-11)

- FIX: Add checks for indexes in `rasters.read`
- FIX: Fix bug with nodata in `rasters._vectorize` and rasters without nodata set

## 1.13.0 (2022-03-17)

- **ENH: Adding a `reduce_verbosity` function in CI**
- CI: Log debug when tests
- DOC: Copyright to 2022
- DOC: Some updates

## 1.12.2 (2022-02-24)

- CI: Test code only if files have changed
- CI: Publishing wheel from GitHub instead of Gitlab

## 1.12.1 (2022-02-24)

- OPT: Do not `export_grid_mapping` when using `rioxarray.open_rasterio`
- FIX: `vectors.shapes_to_gdf`: Fix geometry when converting to geopandas
- FIX: `rasters_rio.collocate` returns a masked_array if a masked_array is given as input
- FIX: Use `--no-binary fiona,rasterio` directly in `requirements.txt`
- FIX: Remove warnings
- CI: Clean `gitlab-ci`
- REPO: Setting GitHub as the main repository and using new Gitlab runners

## 1.12.0 (2021-12-07)

- **ENH: Adding a `assert_geom_almost_equal` function**
- FIX: Better logs for CI functions

## 1.11.1 (2021-12-06)

- FIX: Using `dill` instead of `pickle` as it works on more python types

## 1.11.0 (2021-12-02)

- ENH: Add `read_archived_file` and `read_archived_html` functions
- FIX: Fixing ValueError in `rasters_rio.path_or_arr_or_dst_wrapper` (ambiguous test)

## 1.10.0 (2021-11-26)

- **BREAKING CHANGE**: Removing `to_np` function (useless)
- **BREAKING CHANGE**: Removing useless `dtype` argument from the `rasqters.sieve` function
- FIX: Correcting the `sieving` function that misused nodata values
- FIX: Correcting the `rasters_rio.write` function that modified the array instead of just writing it

## 1.9.0 (2021-09-28)

- ENH: Adding `slope` and `hillshade` functions (to bypass `gdaldem` CLI)

## 1.8.1 (2021-09-22)

- FIX: Handling more cases in `files.is_writable`

## 1.8.0 (2021-09-22)

- ENH: Adding `files.is_writable` function to test if a directory is writable or not
- DOC: Using readthedocs instead of github docs

## 1.7.4 (2021-09-14)

- FIX: Fixing python version in environment.yml
- FIX: Fixing driver to `GTiff` in `rasters.write`
- CI: Fixing dissolve with shapely < 1.8

## 1.7.3 (2021-09-08)

- FIX: Checking path existence in `vectors.read`
- FIX: Repair geometries in `vectorize`
- FIX: Do not modify in place the input in `merge_vrt` (`str` transformed in `Path`)
- CI: Stop writing vector on disk

## 1.7.2 (2021-09-06)

- FIX: Managing dask arrays with rasterio `sieve`
- CI: Testing properly `rasters` functions with dask

## 1.7.1 (2021-09-03)

- ENH: Adding  `ArcPyLogHandler` in `arcpy`
- FIX: Updating `init_conda_arcpy_env`
- CI: Testing `rasters.read` with chunks

## 1.7.0 (2021-08-30)

- ENH: Adding a function managing `arcpy` environment

## 1.6.0 (2021-08-26)

- ENH: Enabling Dask and ensure the functions are Dask-compatible
- FIX: Fixing typo in `snap.get_gpt_cli` function (`tileHeight`)
- CI: Do not lint on tags
- CI: Test with Dask local cluster

## 1.5.0 (2021-08-18)

- ENH: Making `add_to_zip` work with cloud zips
- BREAKING CHANGE: `add_to_zip` outputs the completed path
- FIX: `environment.yml` to respect the stricter use of `file:` syntax.
  See [here](https://stackoverflow.com/questions/68571543/using-a-pip-requirements-file-in-a-conda-yml-file-throws-attributeerror-fileno)
  for more information.
- FIX: Use `numpy>=1.21.2` to avoid a bug in `rasterio.merge` with `min`/`max` options. See [here](https://github.com/mapbox/rasterio/issues/2245#issuecomment-900585934) for more information.
- CI: Do not run pytests on tags and discard `except` keywords

## 1.4.8 (2021-07-29)

- ENH: Adding `ci.assert_raster_max_mismatch` allowing a mismatch between two rasters' pixels

## 1.4.7 (2021-07-28)

- FIX: Fixing the management of shapefiles on the cloud (caching the .shp and all other files)
- FIX: `ci.assert_geom_equal` manages correctly GeoSeries
- CI: renaming `build` step to `lint`
- CI: Optimizing the lib installation

## 1.4.6 (2021-07-19)

- `rasters.write` and `rasters_rio.write`:
    - Manage correctly BigTiffs with LZW compression
    - Use the maximum number of available threads to compress

## 1.4.5 (2021-07-16)

- Fix: clumsy metadata management in `rasters_rio.merge_gtiff`
- ENH: We can use paths when testing with `ci.assert_geom_equal`

## 1.4.4 (2021-07-13)

- Fix: Fixing a bug when using relative path with a start that is not an exact root
- Adding a DOI and a .coveragerc

## 1.4.3 (2021-07-05)

- Fix: JSON can serialize Pathlib objects
- Fix: `vectors.read` forces CRS to WGS84 for KML

## 1.4.2 (2021-07-02)

- By default, using `BIGTFF=IF_NEEDED` when writing files on disk
- Bug resolution when passing a rasterio dataset info `rasters` functions
- Bug resolution for pathlib paths with `vectors.read`
- Type hints updates

## 1.4.1 (2021-06-29)

- `vectors.read`:
    - Manage IO[bytes] and other inputs instead of only path in vectors.read and set KML vectors to WGS84
    - Manage Null Layer exception
- [CI] Updating CI to really test S3 data

## 1.4.0 (2021-06-28)

- Handling S3 compatible storage data
- [vectors] Adding a read function handling KML, GML, archived vectors...
- [API break] `files.read_archived_vector` is removed (ise `vectors.read` instead)
- [API break] Using pathlib objects instead of str
- CI: Updates

## 1.3.15 (2021-06-16)

- Adding a `display` file
- Adding a scaling function in `display`

## 1.3.14.post4 (2021-06-07)

- Managing naive geometries in `vectors.open_gml`

## 1.3.14.post3 (2021-06-07)

- Converting GML GeoDataFrame to the wanted CRS in `vectors.open_gml`

## 1.3.14.post2 (2021-06-07)

- Popping `_FillValue` from xarray attribute (wrongly set there by sth) in `rasters_rio.write`

## 1.3.14.post1 (2021-06-03)

- Compressing to `LZW` by default in `rasters.write` and `rasters_rio.write`

## 1.3.14.post0 (2021-06-02)

- Setting original dtype all the time in `rasters.read`

## 1.3.14 (2021-05-31)

- Add a `as_list` keyword to `files.get_archived_rio_path()`
- Add a `vectors.open_gml` overloading `gpd.read_file` for GML vectors in order to avoid exceptions for empty geometries

## 1.3.13.post1 (2021-05-27)

- Copy the `encoding` dict before setting the nodata

## 1.3.13.post0 (2021-05-27)

- Correct the original dtype in `rasters.read()`
- Keep xarray attributes in `rasters.read()`
- Pass the `encoding` dict in `rasters.set_nodata()`

## 1.3.13 (2021-05-26)

- [rasters] Adding the possibility to specify an index
- [rasters_rio] Adding the possibility to use all `rio.read()` function arguments
- [CI] Adding weekly tests with tox for py3.7, py3.8, py3.9 on Linux and Windows

## 1.3.12 (2021-05-20)

- Using xarray 0.18+ and rioxarray 0.4+

## 1.3.11.post4 (2021-05-04)

- Fixing bug when array has no nodata in `rasters.to_np`
- No need to set np.nan in xarray.where (default value)

## 1.3.11.post3 (2021-05-04)

- Missing `psutil` in setup.py and setting minimum versions

## 1.3.11.post2 (2021-05-04)

- Correctly manage nodata in `rasters.sieve`
- Bug correction in `rasters.where`

## 1.3.11.post1 (2021-05-04)

- Bad nodata setting in `rasters.paint`

## 1.3.11 (2021-05-04)

- Adding a function `rasters.paint` to fill a value where stands a polygon
- Set the nodata after `rasters.mask`
- Setting `from_disk=True` by default in `rasters.crop`
- Bug correction in `rasters.where` when setting nodata

## 1.3.10.post2 (2021-04-30)

- Fixing the original dtype management in the `rasters` decorator and in the `rasters.vectorize`

## 1.3.10 (2021-04-30)

- Fixing the parameter `dissolve` in `vectorize` function (invalid geometries...)

## 1.3.9 (2021-04-30)

- Adding a parameter `dissolve` to `vectorize`, allowing the user to retrieve a unique polygon

## 1.3.8 (2021-04-30)

- JSON Encoder converts int32 to int for some system that needs it
- `rasters.where`: convert type only if needed and output a xarray.DataArray if master_xda is passed
- Adding a parameter `keep_values` to `vectorize`, allowing the user to discard the specified values

## 1.3.7 (2021-04-29)

- Fixing regression in `rasters_rio.unpackbits`
- Fixing regression in `ci.assert_raster_almost_equal` and always checking transform to 10-9

## 1.3.6 (2021-04-29)

- Optimizing `rasters.read` (very slow function, maybe we need to take a look at that)
- Optimizing `rasters.set_nodata` and in `rasters_rio.unpackbits`
- Other minor optimizations

## 1.3.5 (2021-04-29)

- Managing exotic dtypes in `rasters.write`
- Adding a `rasters.where` function preserving metadata and nodata
- Fixing the case with `rasters.set_metadata` with `xarray` without CRS

## 1.3.4 (2021-04-28)

- Setting default nodata according to the dtype in `rasters.write`:
    - uint8: 255
    - int8: -128
    - uint16, uint32, int32, int64, uint64: 65535
    - int16, float32, float64, float128, float: -9999

## 1.3.3 (2021-04-27)

- [`rasters.read`]
    - Coarsen instead of reprojecting if possible (faster)
    - Load as float32 if possible
- Updates in CI to automatically update documentation on Github on new tags

## 1.3.2 (2021-04-27)

- Allowing `gpd.GeoDataFrames` in `crop`/`mask`
- **API break**: `rasters_rio.write(array, path, meta)` becomes `rasters_rio.write(array, meta, path)` !

## 1.3.1 (2021-04-27)

- Do not lose attributes when using `rasters.set_nodata`
- Discard locks when reading `xarrays` (in `rasters.read`)

## 1.3.0 (2021-04-26)

- Going Open Source
