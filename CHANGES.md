# Release History

## 1.3.Z (2021-MM-DD)

## 1.3.11 (2021-05-04)

- Adding a function `rasters.paint` to fill a value where stands a polygon
- Set the nodata after `rasters.mask`

## 1.3.10.post3 (2021-05-03)

- Setting `from_disk=True` by default in `rasters.crop`

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
