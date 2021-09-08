# Release History

## 1.7.3 (2021-09-08)

- FIX: Checking path existence in `vectors.read`
- FIX: Repair geometries in `vectorize`
- FIX: Do not modify in place the input in `merge_vrt` (`str` transformed in `Path`)

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
- FIX: Use `numpy>=1.21.2` to avoid a bug in `rasterio.merge` with `min`/`max` options.
  See [here](https://github.com/mapbox/rasterio/issues/2245#issuecomment-900585934) for more information.
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
