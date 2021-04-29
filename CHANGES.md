# Release History

## X.Y.Z (YYYY-MM-DD)

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
