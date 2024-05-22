# Release History

## 1.37.1 (2024-05-22)
- FIX: Don't set endpoint_url in s3 if environment variable is empty

## 1.37.0 (2024-04-22)

- **ENH: Add AWS profile feature to s3 module**
- FIX: Fix `files.extract_file` when there are only files in the root of the zip archive ([#14](https://github.com/sertit/sertit-utils/pull/14))
- FIX: FIX metadata handling with `rasters_rio.read` when reading with indexes

## 1.36.1 (2024-03-26)

- FIX: Fix `vector._read_vector_core` when we convert a GML file from S3 to geojson with ogr2ogr ([#12](https://github.com/sertit/sertit-utils/pull/12))
- FIX: Fix `files.extract_file` when there is a file in the root of the zip archive ([#11](https://github.com/sertit/sertit-utils/pull/11))
- FIX: Fix `geometry.nearest_neighbors` when k is bigger than the number of candidates
- FIX: Add a `buffer_on_input` in `geometry.intersects` to manage edge cases when points on polygons boundary aren't seen as intersecting
- FIX: `rasters.read` accepts `xarray` objects as input
- FIX: Sanitize imports
- DOC: Update some examples in documentation

## 1.36.0 (2024-02-27)

- **BREAKING CHANGE**: Rename `feature_layer_to_path` to `gp_layer_to_path`
- **BREAKING CHANGE**: Rename `rasters(_rio).get_nodata_mask` to `rasters(_rio).get_data_mask` to better fit with its behaviour (1 where data is valid, 0 elsewhere)
- **ENH: Add `geometry.intersects` to find polygons intersecting with other polygons (wrapper of `geopandas.intersects` that works only row-wise)**
- **ENH: Add `geometry.line_merge` to use `shapely.line_merge` on GeoDataFrames**
- **ENH: Add `geometry.buffer` (using `gpd.buffer`) to return a GeoDataFrame instead of a GeoSeries**
- **ENH: Add `geometry.nearest_neighbors` to get the nearest neighbors from each point of a Point GeoDataFrame in another one (two methods, `k_neighbors` and `radius`). Needs `sklearn`.**
- FIX: Ignore when trying to split polygons with points
- FIX: Make `ci.assert_val` work for Iterables
- DOC: Numerous documentation updates to better use Sphinx syntax

## 1.35.0 (2024-02-08)

- **ENH: Add `arcpy.feature_layer_to_path` to retrieve the path from an ArcGIS feature layer**
- **ENH: Add a class `ArcPyLogger` to better handle logs in Arcgis Pro tools**
- **ENH: Add `geometry.split` to split polygons with polygons**
- FIX: Fix `rasters.sieve` with integer data

## 1.34.2 (2024-01-23)

- FIX: Update `setup.py` to have all the needed dependencies listed in `requirements.txt` ([#9](https://github.com/sertit/sertit-utils/issues/9))
- FIX: Fix `vectors.read` with `engine=pyogrio` when opening vectors without geometries (like `dbf` files)

## 1.34.1 (2024-01-17)

- FIX: Update `arcpy.init_conda_arcpy_env` to fix new issues (`GDAL` `InvalidVersion` when writing on disk with `geopandas` using `fiona`'s engine)

## 1.34.0 (2024-01-15)

- **BREAKING CHANGE**: Set default `chunks` to `auto` in `rasters.read`
- **ENH: Add types for ArcGis GDB in `arcpy`**
- FIX: Allow folders to be opened in `vectors.read` (to open GDBs)
- OPTIM: Geopandas now handles correctly S3 paths, so don't download S3-stored vectors anymore

## 1.33.0 (2024-01-02)

- **ENH: Mirror `window` in `vectors.read` (from `rasters.read`), enhancing `gpd.read_file(bbox=...)`**
- FIX: Allow kwargs in `rasters.collocate`
- DOC: Update copyright to 2024

## 1.32.4 (2023-12-07)

- FIX: Fix requester_pays option in `s3` module
- FIX: Use `total_bounds` when computing the window in `rasters(_rio).read()`

## 1.32.3 (2023-11-28)

- FIX: Fixing additional arguments passed to the `s3` decorator

## 1.32.2 (2023-11-22)

- FIX: Fixing the return of s3 environment decorators
- CI: Enabling pre-commit.ci and dependabot bots

## 1.32.1 (2023-11-14)

- FIX: Add the support of `no_sign_request` in `s3` functions

## 1.32.0 (2023-11-13)

- **BREAKING CHANGE**: Change the order of `files.save_json` function to fit `files.save_obj`. Older order is deprecated.
- **ENH: Allow to pass \*\*kwargs in `files.save_json` and `files.save_obj`**
- **ENH: Allow to pass \*\*kwargs for S3 environments, in order to add options such as _requester pays_**
- FIX: Use `EPSG_4326` instead of `WGS84` for sake of naming accuracy (this is not the same thing!) (`WGS84` stays available though)
- FIX: Return `False` instead of failin in `path.is_cloud_path` if it cannot be converted to `AnyPath`
- FIX: Fix the custom JSON encoder to handle sets
- FIX: Handles correctly multi-layered KMZ in `vectors.read`

## 1.31.0 (2023-10-30)

- **ENH: Add `s3.temp_s3` and `unistra.unistra_s3` context managers to manage s3 environments inside Python code**
- FIX: Fix `rasters.read` whith given indexes order
- DEPS : only import `vectors` inside functions for `ci` module (in order not to have to install rasterio if these functions are not needed)
- DEPS : don't import anything from `rasterio` inside `vectors` module (in order not to have to install rasterio if these functions are not needed)
- DEPS: Remove as many mention as possible to `cloudpathlib`

## 1.30.1 (2023-10-20)

- FIX: Reorder raster in `rasters.read` whith the given indexes order
- FIX: Allow to write with any driver in `rasters(_rio).write`
- FIX: Create proper variables for environment variables in `snap`
- FIX: Normalize geometries before testing within `ci`
- FIX: `files.get_archived_rio_path` returns the result of `path.get_archived_rio_path` instead of incorrect one.
- CI: Update pre-commit hooks

## 1.30.0 (2023-10-04)

- **BREAKING CHANGE**: Creating a `path` module where following `files` functions have been transferred (original have been deprecated): `get_root_path`, `listdir_abspath`, `to_abspath`, `real_rel_path`, `get_archived_file_list`, `get_archived_path`, `get_archived_rio_path`, `get_filename`, `get_ext`, `find_files`, `get_file_in_dir`, `is_writable`.
- **ENH: Add a `s3` modules handling other endpoints than Unistra's**
- **ENH: Add deprecation for `ci` functions handled in other modules (such as `s3_env`, `define_s3_client`, `get_db2_path`, `get_db2_path`, `get_db2_path`)**
- FIX: Allow `unistra.s3_env` to wrap functions with arguments
- FIX: Manage the case with fsspec path given to `vectors.read`
- CI: Better testing of kwargs handling in `vectors.read`
- Update README

## 1.29.1 (2023-09-26)

- **ENH: Add a function covering `vectors.corresponding_utm_projection`'s usecase (converting lat/lon to UTM CRS): `vectors.to_utm_crs`. Returns directly a CRS instead of a string. Keep the deprecation for `corresponding_utm_projection` but not for the same reason.**

## 1.29.0 (2023-09-25)

- **BREAKING CHANGE**: Creating a `geometry` module where following `vectors` functions have been transferred: `from_polygon_to_bounds`, `from_bounds_to_polygon`, `get_wider_exterior`, `make_valid`. The function `fill_polygon_holes` has been created.
- **ENH: Add a `vectors.utm_crs` context manager allowing the user to compute seamlessly geographic-based operation (such as centroids, area computation...)**
- **ENH: Add a `sertit.types` containing aliases to common typings**
- **ENH: Add kwargs in `vectors.read`**
- **ENH: Handles unchecked attributes in `ci.assert_xr_encoding_attrs`**
- **ENH: Add more types**
- **ENH: Add a new module `unistra` used to handle functions referring to Unistra's environment**
- FIX: Return an AssertionError text in `ci.assert_xr_encoding_attrs`
- FIX: Fix `display.scale_to_uint8` with numpy masked arrays
- DEPR: Deprecation of `vectors.corresponding_utm_projection` in favor to [geopandas' `estimate_utm_crs`](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.estimate_utm_crs.html)

## 1.28.3 (2023-07-20)

- FIX: Fixing Windows archived rasterio path (switching to `vsizip`/`vsitar` for all platform)

## 1.28.2 (2023-07-07)

- FIX: Allow collocating rasters (only if they are given as DataArrays) from different dtypes
- FIX: Make VRT relative in `merge_vrt` if possible

## 1.28.1 (2023-06-22)

- FIX: Add a workaround for a weird bug with dask's `reshape`

## 1.28.0 (2023-06-15)

- **ENH: Add a `vectors.copy` function to handle the copy of shapefiles**
- FIX: Fix debug message in `files.copy`

## 1.27.3 (2023-06-12)

- FIX: Add `stacklevel=3` when throwing deprecation warnings
- CI: Add a test to check if the deprecation warning is thrown

## 1.27.2 (2023-06-07)

- FIX: Allow users to read KMZ vectors (as it is now handled by fiona with LIBKML)
- FIX: Better manage non-existing file given as a window `rasters(_rio).read`

## 1.27.1 (2023-04-24)

- FIX: Don't manage `band` coordinate in `rasters.collocate`: keep as is
- FIX: Rename `reference_arr` into `reference` in `rasters.collocate`

## 1.27.0 (2023-04-24)

- **ENH: Add a `misc.unique` function to retrieve unique elements of a list while keeping the original order**
- FIX: Fix the correct number of bands (in the coordinates of the xr.DataArray) in `rasters.collocate`
- FIX: Changes names to reference/other objects in `rasters(_rio).collocate`

## 1.26.0 (2023-04-17)

- **ENH: Add a `vectors.write` function to automatically detect the driver from the filename (and add the KML driver if needed)**
- **ENH: Add shapely 2.0 functions in `ci.assert_geom_xxx` to handle more cases (2D/3D vectors, geomtry written in another way)**
- FIX: Use our own function for setting the nodata instead of using rioxarray's in `rasters.crop`
- FIX: Fix GeoPandas FutureWarning in `explode`: `Currently, index_parts defaults to True, but in the future, it will default to False to be consistent with Pandas.`
- INTERNAL: Simplify `rasters.read`
- DEPS: Pin Shapely >= 2.0
- DEPS: Dropping support of Python 3.8

## 1.25.0 (2023-04-04)

- **ENH: Add a function simplifying footprints**
- **ENH: Pass the `chunks` keyword to `open_rasterio` in `@path_xarr_dst`**
- **ENH: Add a wrapper to add deprecation warnings in `logs`**
- OPTIM: Better nodata management in `rasters.write`
- DEPS: Allow last versions of Dask
- DOC: Update README

## 1.24.4 (2023-03-07)

- OPTIM: Allowing the user to change SNAP tile size with the environment variable `SERTIT_UTILS_SNAP_TILE_SIZE`.
- OPTIM: Allowing the user to change SNAP  max cores with the environment variable `SERTIT_UTILS_MAX_CORES`.
- DOC: Document the usable environment variables

## 1.24.3 (2023-01-27)

- DEPS: Fixing deps (xarray is requiered by default)

## 1.24.2 (2023-01-25)

- FIX: Allowing the users to open `\\DS2\database0x` directories from `ci.get_dbx_path`

## 1.24.1 (2023-01-24)

- FIX: fixing infinite values in `ci.assert_raster_almost_equal_magnitude`

## 1.24.0 (2023-01-23)

- **ENH: Adding NODATA formalizations in `rasters(_rio)`: function to get the value from a dtype (`get_nodata_value` and global variables for `uint8`, `int8`, `uint16`, `float`)**
- **ENH: Allow to merge rasters with different projections**
- **ENH: Adding a `ci.assert_raster_almost_equal_magnitude` function that checks decimals from the scientific expression of the raster**
- CI: Don't run tests when only `__init__` or `__meta__` is updated
- CI: Move data elsewhere than git

## 1.23.0 (2023-01-09)

- **ENH: Adding `ci.assert_files_equal` function**

## 1.22.0 (2023-01-06)

- **ENH: Adding several CI functions: `ci.assert_val`, `ci.assert_xr_encoding_attrs`, `ci.s3_env`, `ci.define_s3_client`**
- **ENH: Making public `ci.assert_field`**
- FIX: factorizing some `ci` functions, better logs in other...
- FIX: Fixing `xml.read` with cloud path as str
- FIX: Fixing `ci.assert_dir_equal` if folders don't have files ordrered inside... (for some reason)
- FIX: Add `distributed` in loggers removed by `ci.reduce_verbosity`
- FIX: Use `sertit` LOGGER in `networks` module
- FIX: Manage `CloudPath` in `files.copy` function
- DOC: Changing copyright from 2022 to 2023
- CI: Better handling of logging display for pytest

## 1.21.2 (2022-12-14)

- FIX: Better handling of windows in `rasters(_rio).read`
- FIX: Expose `ci.assert_meta`

## 1.21.1 (2022-12-13)

- FIX: Fix new_shape retrieval when providing a size with height or width equal to the original dimension
- DOC: Add latest DOI link

## 1.21.0 (2022-12-13)

- **ENH: Add the possibility to load an image/geo window in `rasters(_rio).read`** ([#1](https://github.com/sertit/sertit-utils/issues/1))

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
